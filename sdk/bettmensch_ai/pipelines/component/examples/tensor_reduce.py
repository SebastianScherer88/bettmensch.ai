from bettmensch_ai.pipelines.component import as_torch_ddp_component
from bettmensch_ai.pipelines.io import InputParameter, OutputParameter


def tensor_reduce(
    n_iter: InputParameter = 100,
    n_seconds_sleep: InputParameter = 10,
    duration: OutputParameter = None,
) -> None:
    """When decorated with the torch_component decorator, implements a
    bettmensch_ai.TorchComponent that runs a torch DDP across pods and nodes in
    your K8s cluster."""

    import time
    from datetime import datetime as dt

    import GPUtil
    import torch
    import torch.distributed as dist
    from bettmensch_ai.components import LaunchContext

    has_gpu = torch.cuda.is_available()
    ddp_context = LaunchContext()

    print(f"GPU present: {has_gpu}")

    if has_gpu:
        dist.init_process_group(backend="nccl")
    else:
        dist.init_process_group(backend="gloo")

    for i in range(1, n_iter + 1):

        time.sleep(n_seconds_sleep)

        GPUtil.showUtilization()

        a = torch.tensor([ddp_context.rank])

        print(f"{i}/{n_iter}: @{dt.now()}")
        print(f"{i}/{n_iter}: Backend {dist.get_backend()}")

        print(f"{i}/{n_iter}: Global world size: {ddp_context.world_size}")
        print(f"{i}/{n_iter}: Global worker process rank: {ddp_context.rank}")
        print(
            f"{i}/{n_iter}: This makes me worker process "
            f"{ddp_context.rank + 1}/{ddp_context.world_size} globally!"
        )

        print(f"{i}/{n_iter}: Local rank of worker: {ddp_context.local_rank}")
        print(
            f"{i}/{n_iter}: Local world size: {ddp_context.local_world_size}"
        )
        print(
            f"{i}/{n_iter}: This makes me worker process "
            f"{ddp_context.local_rank + 1}/{ddp_context.local_world_size} locally!"  # noqa: E501
        )

        print(f"{i}/{n_iter}: Node/pod rank: {ddp_context.group_rank}")

        if has_gpu:
            device = torch.device(f"cuda:{ddp_context.local_rank}")

            device_count = torch.cuda.device_count()
            print(f"{i}/{n_iter}: GPU count: {device_count}")

            device_name = torch.cuda.get_device_name(ddp_context.local_rank)
            print(f"{i}/{n_iter}: GPU name: {device_name}")

            device_property = torch.cuda.get_device_capability(device)
            print(f"{i}/{n_iter}: GPU property: {device_property}")

        else:
            device = torch.device("cpu")

        a_placed = a.to(device)
        print(f"{i}/{n_iter}: Pre-`all_reduce` tensor: {a_placed}")
        dist.all_reduce(a_placed)
        print(f"{i}/{n_iter}: Post-`all_reduce` tensor: {a_placed}")
        print("===================================================")

    if duration is not None:
        duration_seconds = n_iter * n_seconds_sleep
        duration.assign(duration_seconds)


tensor_reduce_torch_ddp_factory = as_torch_ddp_component(tensor_reduce)

if __name__ == "__main__":
    tensor_reduce()
