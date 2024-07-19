from bettmensch_ai.components import torch_component
from bettmensch_ai.io import InputParameter


def lightning_ddp(
    process_group_backend: InputParameter = "gloo",  # gloo / nccl
    max_epochs: InputParameter = 1,
    accelerator: InputParameter = "cpu",  # cpu / gpu
) -> None:
    """When decorated with the torch_component decorator, implements a
    bettmensch_ai.TorchComponent that runs a torch DDP across pods and nodes in
    your K8s cluster."""

    # imports
    import os

    import lightning as L
    import torch
    import torch.nn.functional as F
    from lightning.pytorch.strategies import DDPStrategy
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.datasets import MNIST

    # pytorch modules
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Sequential(
                nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3)
            )

        def forward(self, x):
            return self.l1(x)

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Sequential(
                nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28)
            )

        def forward(self, x):
            return self.l1(x)

    class LitAutoEncoder(L.LightningModule):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder

        def training_step(self, batch, batch_idx):
            # training_step defines the train loop.
            x, _ = batch
            x = x.view(x.size(0), -1)
            z = self.encoder(x)
            x_hat = self.decoder(z)
            loss = F.mse_loss(x_hat, x)
            return loss

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
            return optimizer

    dataset = MNIST(
        os.getcwd(), download=True, transform=transforms.ToTensor()
    )  # noqa: E501
    train_loader = DataLoader(dataset)

    # model
    autoencoder = LitAutoEncoder(Encoder(), Decoder())

    # Explicitly specify the process group backend if you choose to
    ddp = DDPStrategy(process_group_backend=process_group_backend)

    # Configure the strategy on the Trainer & train model
    trainer = L.Trainer(
        strategy=ddp, max_epochs=max_epochs, accelerator=accelerator, devices=8
    )
    trainer.fit(model=autoencoder, train_dataloaders=train_loader)


lightning_ddp_torch_factory = torch_component(lightning_ddp)
