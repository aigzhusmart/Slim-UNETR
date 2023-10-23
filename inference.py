import os
import sys
from datetime import datetime
from typing import Dict

import monai
import pytz
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
from src.utils import Logger, load_pretrain_model

best_acc = 0
best_class = []


def warm_up(
    model: torch.nn.Module,
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
    # warm_up
    model.train()
    accelerator.print(f"Start Warn Up!")
    for i, image_batch in enumerate(train_loader):
        logits = model(image_batch["image"])
        total_loss = 0
        log = ""
        for name in loss_functions:
            alpth = 1
            loss = loss_functions[name](logits, image_batch["label"])
            accelerator.log({"Train/" + name: float(loss)}, step=step)
            total_loss += alpth * loss
        accelerator.backward(total_loss)
        optimizer.step()
        optimizer.zero_grad()
        accelerator.log(
            {
                "Train/Total Loss": float(total_loss),
            },
            step=step,
        )
        step += 1
    scheduler.step(epoch)
    accelerator.print(f"Warn Up Over!")
    return step


@torch.no_grad()
def val_one_epoch(
    model: torch.nn.Module,
    config: EasyDict,
    is_HepaticVessel: bool,
    inference: monai.inferers.Inferer,
    val_loader: torch.utils.data.DataLoader,
    metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
    step: int,
    post_trans: monai.transforms.Compose,
    accelerator: Accelerator,
):
    # inference
    model.eval()
    dice_acc = 0
    dice_class = []
    hd95_acc = 0
    hd95_class = []
    for i, image_batch in enumerate(val_loader):
        logits = inference(image_batch["image"], model)
        val_outputs = [post_trans(i) for i in logits]
        for metric_name in metrics:
            metrics[metric_name](y_pred=val_outputs, y=image_batch["label"])
        accelerator.print(f"[{i + 1}/{len(val_loader)}] Validation Loading", flush=True)
        step += 1
    metric = {}

    if is_HepaticVessel == True:
        for metric_name in metrics:
            batch_acc = metrics[metric_name].aggregate()
            if accelerator.num_processes > 1:
                batch_acc = (
                    accelerator.reduce(batch_acc.to(accelerator.device))
                    / accelerator.num_processes
                )
            metrics[metric_name].reset()
            if metric_name == "dice_metric":
                metric.update(
                    {
                        f"Val/mean {metric_name}": float(batch_acc.mean()),
                        f"Val/Hepatic Vessel {metric_name}": float(batch_acc[0]),
                        f"Val/Tumors {metric_name}": float(batch_acc[1]),
                    }
                )
                dice_acc = torch.Tensor([metric["Val/mean dice_metric"]]).to(
                    accelerator.device
                )
                dice_class = batch_acc
            else:
                metric.update(
                    {
                        f"Val/mean {metric_name}": float(batch_acc.mean()),
                        f"Val/Hepatic Vessel {metric_name}": float(batch_acc[0]),
                        f"Val/Tumors {metric_name}": float(batch_acc[1]),
                    }
                )
                hd95_acc = torch.Tensor([metric["Val/mean hd95_metric"]]).to(
                    accelerator.device
                )
                hd95_class = batch_acc
    else:
        for metric_name in metrics:
            batch_acc = metrics[metric_name].aggregate()
            if accelerator.num_processes > 1:
                batch_acc = (
                    accelerator.reduce(batch_acc.to(accelerator.device))
                    / accelerator.num_processes
                )
            metrics[metric_name].reset()
            if metric_name == "dice_metric":
                metric.update(
                    {
                        f"Val/mean {metric_name}": float(batch_acc.mean()),
                        f"Val/TC {metric_name}": float(batch_acc[0]),
                        f"Val/WT {metric_name}": float(batch_acc[1]),
                        f"Val/ET {metric_name}": float(batch_acc[2]),
                    }
                )
                dice_acc = torch.Tensor([metric["Val/mean dice_metric"]]).to(
                    accelerator.device
                )
                dice_class = batch_acc
            else:
                metric.update(
                    {
                        f"Val/mean {metric_name}": float(batch_acc.mean()),
                        f"Val/TC {metric_name}": float(batch_acc[0]),
                        f"Val/WT {metric_name}": float(batch_acc[1]),
                        f"Val/ET {metric_name}": float(batch_acc[2]),
                    }
                )
                hd95_acc = torch.Tensor([metric["Val/mean hd95_metric"]]).to(
                    accelerator.device
                )
                hd95_class = batch_acc
    return dice_acc, dice_class, hd95_acc, hd95_class


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

    utils.same_seeds(50)
    logging_dir = os.getcwd() + "/logs/" + str(datetime.now())
    accelerator = Accelerator(cpu=False)
    Logger(logging_dir if accelerator.is_local_main_process else None)
    accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])
    accelerator.print(objstr(config))

    accelerator.print("load model...")
    model = SlimUNETR(**config.slim_unetr)
    image_size = config.trainer.image_size

    accelerator.print("load dataset...")
    train_loader, val_loader = get_dataloader(config, data_flag)

    inference = monai.inferers.SlidingWindowInferer(
        roi_size=ensure_tuple_rep(image_size, 3),
        overlap=0.5,
        sw_device=accelerator.device,
        device=accelerator.device,
    )
    loss_functions = {
        "focal_loss": monai.losses.FocalLoss(to_onehot_y=False),
        "dice_loss": monai.losses.DiceLoss(
            smooth_nr=0, smooth_dr=1e-5, to_onehot_y=False, sigmoid=True
        ),
    }
    metrics = {
        "dice_metric": monai.metrics.DiceMetric(
            include_background=True,
            reduction=monai.utils.MetricReduction.MEAN_BATCH,
            get_not_nans=False,
        ),
        "hd95_metric": monai.metrics.HausdorffDistanceMetric(
            percentile=95,
            include_background=True,
            reduction=monai.utils.MetricReduction.MEAN_BATCH,
            get_not_nans=False,
        ),
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
    step = 0
    best_eopch = -1
    val_step = 0

    # load pre-train model
    model = load_pretrain_model(
        f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/best/pytorch_model.bin",
        model,
        accelerator,
    )

    model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(
        model, optimizer, scheduler, train_loader, val_loader
    )

    # start inference
    accelerator.print("Start Val！")

    _ = warm_up(
        model,
        loss_functions,
        train_loader,
        optimizer,
        scheduler,
        metrics,
        post_trans,
        accelerator,
        0,
        step,
    )

    dice_acc, dice_class, hd95_acc, hd95_class = val_one_epoch(
        model,
        config,
        is_HepaticVessel,
        inference,
        val_loader,
        metrics,
        val_step,
        post_trans,
        accelerator,
    )
    accelerator.save_state(
        output_dir=f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/best/new/"
    )
    accelerator.print(f"dice acc: {dice_acc}")
    accelerator.print(f"dice class : {dice_class}")
    accelerator.print(f"hd95 acc: {hd95_acc}")
    accelerator.print(f"hd95 class : {hd95_class}")
    accelerator.save_state(
        output_dir=f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/best"
    )
    sys.exit(1)
