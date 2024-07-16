from bettmensch_ai.io import InputParameter, OutputParameter


def torch_ddp(
    n_iter: InputParameter = 100,
    n_seconds_sleep: InputParameter = 10,
    duration: OutputParameter = None,
) -> None:
    """When decorated with the torch_component decorator, implements a
    bettmensch_ai.TorchComponent that runs a torch DDP across pods and nodes in
    your K8s cluster."""

    import time
    from datetime import datetime as dt

    import torch
    import torch.distributed as dist

    has_gpu = torch.cuda.is_available()
    print(f"GPU present: {has_gpu}")

    if has_gpu:
        dist.init_process_group(backend="nccl")
    else:
        dist.init_process_group(backend="gloo")

    for i in range(1, n_iter + 1):

        time.sleep(n_seconds_sleep)

        a = torch.tensor([dist.get_rank()])

        print(f"{i}/{n_iter}: @{dt.now()}")
        print(f"{i}/{n_iter}: Backend {dist.get_backend()}")
        print(f"{i}/{n_iter}: World size {dist.get_world_size()}")
        print(f"{i}/{n_iter}: Rank {dist.get_rank()}")
        print(
            f"{i}/{n_iter}: This makes me worker process "
            f"{dist.get_rank() + 1}/{dist.get_world_size()} globally!"
        )

        if has_gpu:
            device = torch.device("cuda:0")

            device_count = torch.cuda.device_count()
            print(f"{i}/{n_iter}: GPU count: {device_count}")

            device_name = torch.cuda.get_device_name(0)
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


if __name__ == "__main__":
    torch_ddp()
