import torch
import torch.distributed as dist
import os
import builtins
import datetime


def setup_for_distributed(is_master, mute=False):
    """
    This function disables printing when not in master process
    """

    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print("[{}] ".format(now), end="")  # print with time stamp
            builtin_print(*args, **kwargs)

    def no_print(*args, **kwargs):
        pass

    if mute:
        builtins.print = no_print
        return

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        args.world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        args.gpu = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        args.dist_url = "tcp://%s:%s" % (os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"])
        os.environ["LOCAL_RANK"] = str(args.gpu)
        os.environ["RANK"] = str(args.rank)
        os.environ["WORLD_SIZE"] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    else:
        print("Not using distributed mode")
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    # args.dist_backend = "nccl"
    print("| distributed init (rank {}): {}, gpu {}".format(args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


import torch.nn as nn
from itertools import chain
from typing import Any, Optional, Union
import copy


class CustomDP(nn.DataParallel):
    def __init__(self, module, device_ids=None, output_device=None, dim=0, no_scatter=False):
        super().__init__(module, device_ids, output_device, dim)
        self.replicas = [copy.deepcopy(module).to(f"cuda:{i}").eval() for i in device_ids]
        self.no_scatter = no_scatter

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        with torch.autograd.profiler.record_function("DataParallel.forward"):
            if not self.device_ids:
                return self.module(*inputs, **kwargs)

            for t in chain(self.module.parameters(), self.module.buffers()):
                if t.device != self.src_device_obj:
                    raise RuntimeError(
                        "module must have its parameters and buffers "
                        f"on device {self.src_device_obj} (device_ids[0]) but found one of "
                        f"them on device: {t.device}"
                    )
            if not self.no_scatter:

                inputs, module_kwargs = self.scatter(inputs, kwargs, self.device_ids)
            # for forward function without any inputs, empty list and dict will be created
            # so the module can be executed on one device which is the first one in device_ids
            else:
                inputs = tuple(tuple(inputs[i][j] for i in range(len(inputs))) for j in range(len(inputs[0])))
                module_kwargs = tuple({} for _ in self.device_ids)

            if not inputs and not module_kwargs:
                inputs = ((),)
                module_kwargs = ({},)

            if len(self.device_ids) == 1:
                return self.module(*inputs[0], **module_kwargs[0])
            outputs = self.parallel_apply(self.replicas, inputs, module_kwargs)
            return outputs

    def get_bit_depth_num(self):
        return self.module.get_bit_depth_num()
