import os
import time
import argparse
from pathlib import Path

import torch
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

import utils.misc as misc
import utils.builder as builder
import utils.dist as dist

from engine import train_epoch, eval_epoch

from model.loss import BPPLoss

import math


def configure_optimizers(model, lr, aux_lr):

    parameters = {n for n, p in model.named_parameters() if not n.endswith(".quantiles") and p.requires_grad}
    aux_parameters = {n for n, p in model.named_parameters() if n.endswith(".quantiles") and p.requires_grad}
    # Make sure we don't have an intersection of parameters
    params_dict = dict(model.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=lr,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=aux_lr,
    )

    return optimizer, aux_optimizer


def train(args):

    dist.init_distributed_mode(args)
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()

    device = torch.device(args.device)

    args.seed = args.seed + global_rank
    misc.set_seed(args.seed)

    torch.backends.cudnn.benchmark = True

    global_batch_size = args.batch_size * num_tasks
    if not hasattr(args, "lr"):
        args.lr = args.blr * math.sqrt(global_batch_size / 64)  # TODO maybe modify

    if not hasattr(args, "aux_lr"):
        args.aux_lr = args.aux_blr * math.sqrt(global_batch_size / 64)

    print("Job directory:", os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:\n{}".format(misc.filter_args(args)).replace(", ", ",\n"))
    print("Global batch size: {}".format(global_batch_size))
    print("Learning rate: {}".format(args.lr))

    sampler_train = data.DistributedSampler(args.train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)

    train_dataloader = data.DataLoader(
        args.train_dataset,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
    )

    val_loader = data.DataLoader(
        args.val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
    )

    model = args.model
    criterion = BPPLoss()

    print("Model = {}".format(model))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.2f}M".format(n_params / 1e6))

    model.to(device)
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    else:
        model_without_ddp = model

    optimizer, aux_optimizer = configure_optimizers(model_without_ddp, args.lr, args.aux_lr)
    scheduler = args.scheduler(optimizer)
    print("Optimizer = {}".format(optimizer))
    print("Aux Optimizer = {}".format(aux_optimizer))
    # print("Scheduler = {}".format(scheduler))

    print("Len train dataset: {}".format(len(args.train_dataset)))
    print("Len val dataset: {}".format(len(args.val_dataset)))

    if args.resume:
        log_dir = os.path.join(args.resume, "logs")  # args.resume : run-xxxx
        ckpt_dir = os.path.join(args.resume, "checkpoints")
        ckp = torch.load(os.path.join(ckpt_dir, "model.pt"), map_location="cpu")

        model_without_ddp.load_state_dict(ckp["model"])  # TODO
        start_epoch = ckp["epoch"] + 1
        optimizer.load_state_dict(ckp["optimizer_state_dict"])
        aux_optimizer.load_state_dict(ckp["aux_optimizer_state_dict"])
        scheduler.load_state_dict(ckp["scheduler"])
        train_step = ckp["step"]
        best_bpp = ckp["best_bpp"]
        del ckp
        print("Resume from {}, start epoch {}".format(ckpt_dir, start_epoch))

    else:
        start_epoch = 0
        train_step = 0
        best_bpp = float("inf")
        args.output_dir = os.path.join(args.output_dir, "run-{}".format(time.strftime("%Y%m%d-%H%M%S")))

        if os.path.exists(args.output_dir):
            args.output_dir = misc.get_unique_dir(args.output_dir)

        log_dir = os.path.join(args.output_dir, "logs")
        ckpt_dir = os.path.join(args.output_dir, "checkpoints")
        if global_rank == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(ckpt_dir, exist_ok=True)

            if args.store:
                misc.save_script_dir(args.output_dir, exclude_dirs=[os.path.dirname(args.output_dir), "data"])

    if global_rank == 0:
        writer = SummaryWriter(log_dir=log_dir)
        writer.add_text("args", str(args).replace(", ", ",\n"))
    else:
        writer = None
    print("Experiment dir : {}".format(args.output_dir))
    print("Start training")
    try:

        for epoch in range(start_epoch, args.num_epochs):
            if args.distributed:
                train_dataloader.sampler.set_epoch(epoch)

            train_step = train_epoch(
                model,
                criterion,
                train_dataloader,
                optimizer,
                aux_optimizer,
                writer,
                train_step,
                clip_grad=args.clip_grad,
            )

            val_loss = eval_epoch(model, criterion, val_loader, epoch, writer)

            if not args.multistep:
                scheduler.step(val_loss)
            else:
                scheduler.step()

            dist.save_on_master(
                {
                    "epoch": epoch,
                    "model": model_without_ddp.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "aux_optimizer_state_dict": aux_optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": train_step,
                    "best_bpp": best_bpp,
                },
                os.path.join(ckpt_dir, "model.pt"),
            )
            if val_loss < best_bpp:

                dist.save_on_master(
                    {
                        "model": model.state_dict(),
                    },
                    os.path.join(ckpt_dir, "best_model.pt"),
                )
                print("New best bpp: {:.4f} -> {:.4f}. Saving model...".format(best_bpp, val_loss))
                best_bpp = val_loss

            print("Epoch: {}/ {}, loss:{:.4f}".format(epoch + 1, args.num_epochs, val_loss))

            if writer is not None:
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        if epoch == args.num_epochs - 1:
            print("Training stopped because reached maximum")

        if writer is not None:
            writer.close()
    except KeyboardInterrupt:
        print("Exiting from training early because of KeyboardInterrupt")


def get_args_parser():
    parser = argparse.ArgumentParser("Training", add_help=False)
    parser.add_argument(
        "--config", type=str, default="configs/rgb_channel_ctx_part4_bd_pw16.py", help="Path to the config file."
    )
    parser.add_argument("--store", action="store_true", help="Whether to store script.")
    parser.add_argument("--resume", type=str, default="", help="Resume from checkpoint.")
    parser.add_argument("--mute", action="store_true", help="Whether to be mute.")
    parser.add_argument("--dist_on_itp", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":

    args = get_args_parser()
    config = builder.load_config(args.config)
    args = builder.merge_config_args(config, args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train(args)
