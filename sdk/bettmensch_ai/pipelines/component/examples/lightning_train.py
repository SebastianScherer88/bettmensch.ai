from bettmensch_ai.pipelines.component import as_torch_ddp_component
from bettmensch_ai.pipelines.io import InputParameter, OutputParameter


def lightning_train(
    max_time: InputParameter = "00:00:00:30",
    duration: OutputParameter = None,
) -> None:
    """When decorated with the torch_component decorator, implements a
    bettmensch_ai.TorchComponent that runs a torch DDP across pods and nodes in
    your K8s cluster."""

    # imports - as per
    # https://github.com/Lightning-AI/pytorch-lightning/issues/17445
    from datetime import datetime as dt

    import lightning.pytorch as pl
    import torch
    from bettmensch_ai.pipelines.component.torch_ddp import (
        LaunchConfigSettings,
    )
    from lightning.pytorch.strategies import DDPStrategy

    start = dt.now()

    # STEP 1: DEFINE YOUR LIGHTNING MODULE
    class ToyExample(pl.LightningModule):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def training_step(self, batch):
            # Send the batch through the model and calculate the loss
            # The Trainer will run .backward(), optimizer.step(), .zero_grad(),
            # etc. for you
            loss = self.model(batch).sum()
            return loss

        def configure_optimizers(self):
            # Choose an optimizer or implement your own.
            return torch.optim.Adam(self.model.parameters())

    # Set up the model so it can be called in `training_step`.
    # This is a dummy model. Replace it with an LLM or whatever
    model = torch.nn.Linear(32, 2)
    pl_module = ToyExample(model)
    # Configure the dataset and return a data loader.
    train_dataloader = torch.utils.data.DataLoader(torch.randn(8, 32))

    # Explicitly specify the process group backend if you choose to
    has_gpu = torch.cuda.is_available()
    print(f"GPU present: {has_gpu}")
    process_group_backend = "nccl" if has_gpu else "gloo"
    accelerator = "gpu" if has_gpu else "cpu"

    ddp = DDPStrategy(process_group_backend=process_group_backend)

    # Configure the strategy on the Trainer & train model
    launch_settings = LaunchConfigSettings()
    trainer = pl.Trainer(
        strategy=ddp,
        accelerator=accelerator,
        num_nodes=launch_settings.max_nodes,
        devices=launch_settings.nproc_per_node,
        max_time=max_time,
    )

    trainer.fit(pl_module, train_dataloader)

    if duration is not None:
        duration.assign(dt.now() - start)


lightning_train_torch_ddp_factory = as_torch_ddp_component(lightning_train)

if __name__ == "__main__":
    lightning_train()
