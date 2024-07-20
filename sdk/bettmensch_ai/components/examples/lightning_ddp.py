from bettmensch_ai.components import lightning_component
from bettmensch_ai.io import InputParameter, OutputParameter


def lightning_ddp(
    max_time: InputParameter = "00:00:01:30",
    duration: OutputParameter = None,
) -> None:
    """When decorated with the torch_component decorator, implements a
    bettmensch_ai.TorchComponent that runs a torch DDP across pods and nodes in
    your K8s cluster."""

    # imports
    # import os
    from datetime import datetime as dt

    import lightning as pl

    # train.py
    import torch

    # import torch.nn.functional as F
    from bettmensch_ai.components.torch_utils import LaunchConfigSettings
    from lightning.pytorch.strategies import DDPStrategy

    # from torch import nn
    # from torch.utils.data import DataLoader
    # from torchvision import transforms
    # from torchvision.datasets import MNIST
    # import lightning as L

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

    # # pytorch modules
    # class Encoder(nn.Module):
    #     def __init__(self):
    #         super().__init__()
    #         self.l1 = nn.Sequential(
    #             nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3)
    #         )

    #     def forward(self, x):
    #         return self.l1(x)

    # class Decoder(nn.Module):
    #     def __init__(self):
    #         super().__init__()
    #         self.l1 = nn.Sequential(
    #             nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28)
    #         )

    #     def forward(self, x):
    #         return self.l1(x)

    # class LitAutoEncoder(pl.LightningModule):
    #     def __init__(self, encoder, decoder):
    #         super().__init__()
    #         self.encoder = encoder
    #         self.decoder = decoder

    #     def training_step(self, batch, batch_idx):
    #         # training_step defines the train loop.
    #         x, _ = batch
    #         x = x.view(x.size(0), -1)
    #         z = self.encoder(x)
    #         x_hat = self.decoder(z)
    #         loss = F.mse_loss(x_hat, x)
    #         return loss

    #     def configure_optimizers(self):
    #         optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    #         return optimizer

    # dataset = MNIST(
    #     os.getcwd(), download=True, transform=transforms.ToTensor()
    # )  # noqa: E501
    # train_loader = DataLoader(dataset)

    # # model
    # autoencoder = LitAutoEncoder(Encoder(), Decoder())

    # Explicitly specify the process group backend if you choose to
    # has_gpu = torch.cuda.is_available()
    # print(f"GPU present: {has_gpu}")
    # process_group_backend = "nccl" if has_gpu else "gloo"
    # accelerator = "gpu" if has_gpu else "cpu"

    # ddp = DDPStrategy(process_group_backend=process_group_backend)

    # # Configure the strategy on the Trainer & train model
    # launch_settings = LaunchConfigSettings()
    # trainer = pl.Trainer(
    #     strategy=ddp,
    #     accelerator=accelerator,
    #     num_nodes=launch_settings.max_nodes,
    #     devices=launch_settings.nproc_per_node,
    #     max_time=max_time,
    # )
    # trainer.fit(model=autoencoder, train_dataloaders=train_loader)

    # if duration is not None:
    #     duration.assign(dt.now() - start)


lightning_ddp_lightning_factory = lightning_component(lightning_ddp)

if __name__ == "__main__":
    lightning_ddp()
