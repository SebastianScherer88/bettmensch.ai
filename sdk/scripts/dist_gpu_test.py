import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl")
print(f"Backend {dist.get_backend()}")
print(f"World size {dist.get_world_size()}")
print(f"Global rank {dist.get_global_rank()}")
print(f"Group rank {dist.get_group_rank()}")
print(f"Rank {dist.get_rank()}")
print(
    f"This makes me worker process {dist.get_rank() + 1}/{dist.get_world_size()} globally!"
)

has_gpu = torch.cuda.is_available()
print(f"GPU present: {has_gpu}")

a = torch.tensor([dist.get_rank()])

if has_gpu:
    device = torch.device("cuda:0")

    device_count = torch.cuda.device_count()
    print(f"GPU count: {device_count}")

    device_name = torch.cuda.get_device_name(0)
    print(f"GPU name: {device_name}")

    device_property = torch.cuda.get_device_capability(device)
    print(f"GPU property: {device_property}")
else:
    device = torch.device("cpu")

a.to(device)
print(f"Pre-`all_reduce` tensor: {a}")
dist.all_reduce(a)
print(f"Post-`all_reduce` tensor: {a}")
