import os
import sys
from datetime import datetime
from typing import Dict

import monai
import torch
import yaml
from accelerate import Accelerator
from easydict import EasyDict
from monai.utils import ensure_tuple_rep
from objprint import objstr
from timm.optim import optim_factory

from src import utils
from src.loader import get_dataloader
from src.optimizer import LinearWarmupCosineAnnealingLR
from src.SlimUNETR.SlimUNETR import SlimUNETR
from src.utils import Logger, same_seeds


def train_one_epoch(
    model: torch.nn.Module,
    config: EasyDict,
    is_HepaticVessel: bool,
    loss_functions: Dict[str, torch.nn.modules.loss._Loss],
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
    post_trans: monai.transforms.Compose,
    accelerator: Accelerator,
    epoch: int,
    step: int,
):
    # train
    model.train()
    for i, image_batch in enumerate(train_loader):
        logits = model(image_batch["image"])
        total_loss = 0
        log = ""
        for name in loss_functions:
            alpth = 1
            loss = loss_functions[name](logits, image_batch["label"])
            accelerator.log({"Train/" + name: float(loss)}, step=step)
            total_loss += alpth * loss
        val_outputs = [post_trans(i) for i in logits]
        for metric_name in metrics:
            metrics[metric_name](y_pred=val_outputs, y=image_batch["label"])

        accelerator.backward(total_loss)
        optimizer.step()
        optimizer.zero_grad()
        accelerator.log(
            {
                "Train/Total Loss": float(total_loss),
            },
            step=step,
        )
        accelerator.print(
            f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] Training [{i + 1}/{len(train_loader)}] Loss: {total_loss:1.5f} {log}",
            flush=True,
        )
        step += 1
    scheduler.step(epoch)
    metric = {}
    if is_HepaticVessel == True:
        for metric_name in metrics:
            batch_acc = metrics[metric_name].aggregate()
            if accelerator.num_processes > 1:
                batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes
            metric.update(
                {
                    f"Train/mean {metric_name}": float(batch_acc.mean()),
                    f"Train/Hepatic Vessel {metric_name}": float(batch_acc[0]),
                    f"Train/Tumors {metric_name}": float(batch_acc[1]),
                }
            )
    else:
        for metric_name in metrics:
            batch_acc = metrics[metric_name].aggregate()
            if accelerator.num_processes > 1:
                batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes
            metric.update(
                {
                    f"Train/mean {metric_name}": float(batch_acc.mean()),
                    f"Train/TC {metric_name}": float(batch_acc[0]),
                    f"Train/WT {metric_name}": float(batch_acc[1]),
                    f"Train/ET {metric_name}": float(batch_acc[2]),
                }
            )
    accelerator.print(
        f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] Training metric {metric}"
    )
    accelerator.log(metric, step=epoch)
    return step


@torch.no_grad()
def val_one_epoch(
    model: torch.nn.Module,
    is_HepaticVessel: bool,
    loss_functions: Dict[str, torch.nn.modules.loss._Loss],
    inference: monai.inferers.Inferer,
    val_loader: torch.utils.data.DataLoader,
    config: EasyDict,
    metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
    step: int,
    post_trans: monai.transforms.Compose,
    accelerator: Accelerator,
    epoch: int,
):
    # val
    model.eval()
    for i, image_batch in enumerate(val_loader):
        logits = inference(image_batch["image"], model)
        total_loss = 0
        log = ""
        for name in loss_functions:
            loss = loss_functions[name](logits, image_batch["label"])
            accelerator.log({"Val/" + name: float(loss)}, step=step)
            log += f" {name} {float(loss):1.5f} "
            total_loss += loss
        val_outputs = [post_trans(i) for i in logits]
        for metric_name in metrics:
            metrics[metric_name](y_pred=val_outputs, y=image_batch["label"])
        accelerator.log(
            {
                "Val/Total Loss": float(total_loss),
            },
            step=step,
        )
        accelerator.print(
            f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] Validation [{i + 1}/{len(val_loader)}] Loss: {total_loss:1.5f} {log}",
            flush=True,
        )
        step += 1

    metric = {}
    for metric_name in metrics:
        batch_acc = metrics[metric_name].aggregate()
        if accelerator.num_processes > 1:
            batch_acc = (
                accelerator.reduce(batch_acc.to(accelerator.device))
                / accelerator.num_processes
            )
        metrics[metric_name].reset()
        if is_HepaticVessel == True:
            metric.update(
                {
                    f"Val/mean {metric_name}": float(batch_acc.mean()),
                    f"Val/Hepatic Vessel {metric_name}": float(batch_acc[0]),
                    f"Val/Tumors {metric_name}": float(batch_acc[1]),
                }
            )
        else:
            metric.update(
                {
                    f"Val/mean {metric_name}": float(batch_acc.mean()),
                    f"Val/TC {metric_name}": float(batch_acc[0]),
                    f"Val/WT {metric_name}": float(batch_acc[1]),
                    f"Val/ET {metric_name}": float(batch_acc[2]),
                }
            )
    accelerator.print(
        f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] Validation metric {metric}"
    )
    accelerator.log(metric, step=epoch)
    return (
        torch.Tensor([metric["Val/mean dice_metric"]]).to(accelerator.device),
        batch_acc,
        step,
    )


if __name__ == "__main__":
    # load yml
    config = EasyDict(
        yaml.load(open("config.yml", "r", encoding="utf-8"), Loader=yaml.FullLoader)
    )
    if config.is_brats2021 == True:
        config = config.brats2021
        data_flag = "brats2021"
        is_HepaticVessel = False
    elif config.is_brain2019 == True:
        config = config.brain2019
        data_flag = "brain2019"
        is_HepaticVessel = False
    else:
        config = config.hepatic_vessel2021
        data_flag = "hepatic_vessel2021"
        is_HepaticVessel = True

    same_seeds(50)
    logging_dir = os.getcwd() + "/logs/" + str(datetime.now())
    accelerator = Accelerator(
        cpu=False, log_with=["tensorboard"], project_dir=logging_dir
    )
    Logger(logging_dir if accelerator.is_local_main_process else None)
    accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])
    accelerator.print(objstr(config))

    accelerator.print("Load Model...")
    model = SlimUNETR(**config.slim_unetr)
    image_size = config.trainer.image_size

    accelerator.print("Load Dataloader...")
    train_loader, val_loader = get_dataloader(config, data_flag)

    inference = monai.inferers.SlidingWindowInferer(
        roi_size=ensure_tuple_rep(image_size, 3),
        overlap=0.5,
        sw_device=accelerator.device,
        device=accelerator.device,
    )
    metrics = {
        "dice_metric": monai.metrics.DiceMetric(
            include_background=True,
            reduction=monai.utils.MetricReduction.MEAN_BATCH,
            get_not_nans=False,
        ),
        # 'hd95_metric': monai.metrics.HausdorffDistanceMetric(percentile=95, include_background=True, reduction=monai.utils.MetricReduction.MEAN_BATCH, get_not_nans=False)
    }
    post_trans = monai.transforms.Compose(
        [
            monai.transforms.Activations(sigmoid=True),
            monai.transforms.AsDiscrete(threshold=0.5),
        ]
    )

    optimizer = optim_factory.create_optimizer_v2(
        model,
        opt=config.trainer.optimizer,
        weight_decay=config.trainer.weight_decay,
        lr=config.trainer.lr,
        betas=(0.9, 0.95),
    )
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=config.trainer.warmup,
        max_epochs=config.trainer.num_epochs,
    )
    loss_functions = {
        "focal_loss": monai.losses.FocalLoss(to_onehot_y=False),
        "dice_loss": monai.losses.DiceLoss(
            smooth_nr=0, smooth_dr=1e-5, to_onehot_y=False, sigmoid=True
        ),
    }

    step = 0
    best_eopch = -1
    val_step = 0
    starting_epoch = 0
    best_acc = 0
    best_class = []

    model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(
        model, optimizer, scheduler, train_loader, val_loader
    )

    # resume training
    if config.trainer.resume:
        model, starting_epoch, step, val_step = utils.resume_train_state(
            model, "{}".format(config.finetune.checkpoint), train_loader, accelerator
        )

    # Start Training
    accelerator.print("Start Training！")
    for epoch in range(starting_epoch, config.trainer.num_epochs):
        # train
        step = train_one_epoch(
            model,
            config,
            is_HepaticVessel,
            loss_functions,
            train_loader,
            optimizer,
            scheduler,
            metrics,
            post_trans,
            accelerator,
            epoch,
            step,
        )
        # val
        mean_acc, batch_acc, val_step = val_one_epoch(
            model,
            is_HepaticVessel,
            loss_functions,
            inference,
            val_loader,
            config,
            metrics,
            val_step,
            post_trans,
            accelerator,
            epoch,
        )

        accelerator.print(
            f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] lr = {scheduler.get_last_lr()} best acc: {best_acc}, mean acc: {mean_acc}, mean class: {batch_acc}"
        )

        # save model
        if mean_acc > best_acc:
            accelerator.save_state(
                output_dir=f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/best"
            )
            best_acc = mean_acc
            best_class = batch_acc
            best_eopch = epoch
        accelerator.save_state(
            output_dir=f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/epoch_{epoch}"
        )

    accelerator.print(f"best dice mean acc: {best_acc}")
    accelerator.print(f"best dice accs: {best_class}")
    sys.exit(1)
