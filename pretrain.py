"""
File: pretrain.py
-------------------
Pretrain the CodonTransformer model.

Data should be a json file with each line containing a dictionary with the following keys:
    - idx: an integer representing the index of the sequence
    - codons: a string representing the tokenized sequence
    - organism: an integer representing the organism index for that sequence  
"""

import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from transformers import (
    PreTrainedTokenizerFast,
    BigBirdConfig,
    BigBirdForMaskedLM
)

from CodonTransformer.CodonUtils import (
    TOKEN2MASK,
    NUM_ORGANISMS,
    MAX_LEN,
    IterableJSONData
)


DATASET_LEN = 1195714
NGPUS = 16
BATCH_SIZE = 6
MAX_EPOCHS = 5
NUM_WORKERS = 5
ACC = 1

TRAIN_BATCHES = DATASET_LEN / BATCH_SIZE / NGPUS * MAX_EPOCHS
GRAD_STEPS = TRAIN_BATCHES * ACC

WARMUP = int(0.1 * GRAD_STEPS)
DECAY = int(0.9 * GRAD_STEPS)

DEBUG = False


class MaskedTokenizerCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        tokenized = self.tokenizer(
            [ex["codons"] for ex in examples],
            return_attention_mask=True,
            return_token_type_ids=True,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )

        seq_len = tokenized["input_ids"].shape[-1]
        species_index = torch.tensor([[ex["organism"]] for ex in examples])
        tokenized["token_type_ids"] = species_index.repeat(1, seq_len)

        inputs = tokenized["input_ids"]
        targets = tokenized["input_ids"].clone()

        prob_matrix = torch.full(inputs.shape, 0.15)
        prob_matrix[torch.where(inputs < 5)] = 0.
        selected = torch.bernoulli(prob_matrix).bool()

        # 80% of the time, replace masked input tokens with respective mask tokens
        replaced = torch.bernoulli(torch.full(selected.shape, 0.8)).bool() & selected
        inputs[replaced] = torch.tensor(list((map(TOKEN2MASK.__getitem__, inputs[replaced].numpy()))))

        # 10% of the time, we replace masked input tokens with random vector.
        randomized = torch.bernoulli(torch.full(selected.shape, 0.1)).bool() & selected & ~replaced
        random_idx = torch.randint(26, 90, prob_matrix.shape, dtype=torch.long)
        inputs[randomized] = random_idx[randomized]

        tokenized["input_ids"] = inputs
        tokenized["labels"] = torch.where(selected, targets, -100)

        return tokenized


class plTrainHarness(pl.LightningModule):
    def __init__(self, model, warmup, decay):
        super().__init__()
        self.model = model
        self.warmup = warmup
        self.decay = decay

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.trainer.model.parameters(), lr=5e-5,
        )
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.,
            total_iters=self.warmup)
        linear_decay = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.,
            end_factor=0.01,
            total_iters=self.decay)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup, linear_decay],
            milestones=[self.warmup]
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def training_step(self, batch, batch_idx):
        # May be set to "full" automatically. Try "block_sparse" to prevent memory errors.
        model.bert.set_attention_type("block_sparse")
        outputs = self.model(**batch)
        (current_lr,) = self.lr_schedulers().get_last_lr()
        self.log_dict(
            dictionary={"loss": outputs.loss, "lr": current_lr},
            on_step=True,
            prog_bar=True
        )

        return outputs.loss


class EpochCheckpoint(pl.Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            # Save a checkpoint every 5 epochs, starting from epoch 0
            if (trainer.current_epoch + 1) % 5 == 0 or trainer.current_epoch == 0:
                checkpoint_path = f"checkpoints/epoch_{trainer.current_epoch}.ckpt"
                trainer.save_checkpoint(checkpoint_path)
                print(f"\nCheckpoint saved at {checkpoint_path}\n")


if __name__ == "__main__":
    pl.seed_everything(123)
    torch.set_float32_matmul_precision("medium")

    tokenizer_model_path = "CodonTransformerTokenizer.json"
    train_data_path = "pretrain_dataset.json"

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_model_path,
        bos_token="[CLS]",
        eos_token="[SEP]",
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
    )

    # BigBird model.
    config = BigBirdConfig(
        vocab_size=len(tokenizer),
        type_vocab_size=NUM_ORGANISMS,
        sep_token_id=2,
    )

    model = BigBirdForMaskedLM(config=config)

    harnessed_model = plTrainHarness(
        model, warmup=WARMUP, decay=DECAY
    )

    train_data = IterableJSONData(train_data_path, dist_env="slurm")

    data_loader = DataLoader(
        dataset=train_data,
        collate_fn=MaskedTokenizerCollator(tokenizer),
        batch_size=BATCH_SIZE,
        num_workers=0 if DEBUG else NUM_WORKERS,
        persistent_workers=False if DEBUG else True,
    )

    save_checkpoint = EpochCheckpoint()

    trainer = pl.Trainer(
        default_root_dir="checkpoints",
        strategy="ddp_find_unused_parameters_true",
        accelerator="gpu",
        devices=1 if DEBUG else -1,
        precision="16-mixed",
        max_epochs=MAX_EPOCHS,
        deterministic=False,
        enable_checkpointing=True,
        callbacks=[save_checkpoint],
        accumulate_grad_batches=ACC
    )

    trainer.fit(harnessed_model, data_loader)
