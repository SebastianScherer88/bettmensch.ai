import time

import GPUtil
import torch
import torch.distributed as dist
from bettmensch_ai.pipelines.component import LaunchContext
from bettmensch_ai.pipelines.component.examples.annotated_transformer.architecture import (  # noqa: E501
    EncoderDecoder,
    make_model,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR

from .data import create_dataloaders
from .optimizer import (
    DummyOptimizer,
    DummyScheduler,
    LabelSmoothing,
    SimpleLossCompute,
)
from .utils import Batch, SpecialTokens, TrainConfig, TrainState, rate


def run_epoch(
    data_iter,
    model: EncoderDecoder,
    loss_compute: SimpleLossCompute,
    optimizer: torch.optim.Adam,
    scheduler: LambdaLR,
    mode: str = "train",
    accum_iter: int = 1,
    train_state: TrainState = TrainState(),
):
    """Train a single epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        # loss_node = loss_node / accum_iter
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state


def train_worker(
    config: TrainConfig,
):
    ddp_context = LaunchContext()
    dist.init_process_group("nccl")

    # data
    preprocessor, (train_dataloader, valid_dataloader) = create_dataloaders(
        ddp_context.local_rank,
        config.dataset,
        config.source_language,
        config.target_language,
        config.max_padding,
        config.source_tokenizer_path,
        config.target_tokenizer_path,
        config.vocabularies_path,
        batch_size=config.batch_size // ddp_context.world_size,
    )

    # model
    print(
        f"Trainer process using GPU: {ddp_context.local_rank} for training",
        flush=True,
    )
    torch.cuda.set_device(ddp_context.local_rank)

    pad_idx = preprocessor.vocab_tgt[SpecialTokens.blank.value]
    d_model = 512
    model = make_model(
        preprocessor.vocab_src_size, preprocessor.vocab_tgt_size, N=6
    )
    model.cuda(ddp_context.local_rank)
    model = DDP(model, device_ids=[ddp_context.local_rank])
    module = model.module
    is_main_process = ddp_context.rank == 0

    criterion = LabelSmoothing(
        size=preprocessor.vocab_tgt_size, padding_idx=pad_idx, smoothing=0.1
    )
    criterion.cuda(ddp_context.local_rank)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.base_lr, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, d_model, factor=1, warmup=config.warmup
        ),
    )
    train_state = TrainState()

    for epoch in range(config.num_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        valid_dataloader.sampler.set_epoch(epoch)

        model.train()
        print(
            f"[GPU{ddp_context.local_rank}] Epoch {epoch} Training ====",
            flush=True,
        )
        _, train_state = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=config.accum_iter,
            train_state=train_state,
        )

        GPUtil.showUtilization()
        if is_main_process:
            file_path = f"{config.file_prefix}{epoch}.pt"
            torch.save(module.state_dict(), file_path)
        torch.cuda.empty_cache()

        print(
            f"[GPU{ddp_context.local_rank}] Epoch {epoch} Validation ====",
            flush=True,
        )
        model.eval()
        sloss = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
            model,
            SimpleLossCompute(criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )
        print(sloss)
        torch.cuda.empty_cache()

    if is_main_process:
        file_path = f"{config.file_prefix}final.pt"
        torch.save(module.state_dict(), file_path)
