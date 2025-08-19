"""
File: finetune.py
-------------------
Finetune the CodonTransformer model.

The pretrained model is loaded directly from Hugging Face.
The dataset is a JSON file. You can use prepare_training_data from CodonData to
prepare the dataset. The repository README has a guide on how to prepare the
dataset and use this script.
"""

import argparse
import os
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BigBirdForMaskedLM

from CodonTransformer.CodonUtils import (
    MAX_LEN,
    TOKEN2MASK,
    IterableJSONData,
)


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
        prob_matrix[torch.where(inputs < 5)] = 0.0
        selected = torch.bernoulli(prob_matrix).bool()

        # 80% of the time, replace masked input tokens with respective mask tokens
        replaced = torch.bernoulli(torch.full(selected.shape, 0.8)).bool() & selected
        inputs[replaced] = torch.tensor(
            list((map(TOKEN2MASK.__getitem__, inputs[replaced].numpy())))
        )

        # 10% of the time, we replace masked input tokens with random vector.
        randomized = (
            torch.bernoulli(torch.full(selected.shape, 0.1)).bool()
            & selected
            & ~replaced
        )
        random_idx = torch.randint(26, 90, prob_matrix.shape, dtype=torch.long)
        inputs[randomized] = random_idx[randomized]

        tokenized["input_ids"] = inputs
        tokenized["labels"] = torch.where(selected, targets, -100)

        return tokenized


class plTrainHarness(pl.LightningModule):
    """Lightning module for training the model."""

    def __init__(
            self,
            model: torch.nn.Module,
            learning_rate: float,
            warmup_fraction: float,
            train_dataset: Optional[IterableJSONData] = None,
            batch_size: Optional[int] = None,
            n_gpus: Optional[int] = None,
            grad_accum: Optional[int] = None
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.warmup_fraction = warmup_fraction

        # Store dataset info for scheduler configuration
        self.train_dataset = train_dataset
        self.batch_size = batch_size
        self.n_gpus = n_gpus
        self.grad_accum = grad_accum

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        # Calculate total steps from dataset if possible
        if all(x is not None for x in [
            self.train_dataset,
            self.batch_size,
            self.n_gpus,
            self.grad_accum
        ]):
            total_steps = self.train_dataset.get_total_steps(
                batch_size=self.batch_size,
                n_gpus=self.n_gpus,
                grad_accum=self.grad_accum
            )
        else:
            total_steps = self.trainer.estimated_stepping_batches

        print(f"\nConfiguring OneCycleLR with total_steps: {total_steps}\n")

        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                total_steps=total_steps,
                pct_start=self.warmup_fraction,
                anneal_strategy='linear'
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        """Execute a single training step."""
        self.model.bert.set_attention_type("block_sparse")
        outputs = self.model(**batch)
        self.log_dict(
            dictionary={
                "loss": outputs.loss,
                "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
            },
            on_step=True,
            prog_bar=True,
        )
        return outputs.loss


class DumpStateDict(pl.callbacks.ModelCheckpoint):
    def __init__(self, checkpoint_dir, checkpoint_filename, every_n_train_steps):
        super().__init__(
            dirpath=checkpoint_dir, every_n_train_steps=every_n_train_steps
        )
        self.checkpoint_filename = checkpoint_filename

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        model = trainer.model.model
        torch.save(
            model.state_dict(), os.path.join(self.dirpath, self.checkpoint_filename)
        )


def main(args):
    """Finetune the CodonTransformer model."""
    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium")

    # Load tokenizer and model from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained("adibvafa/CodonTransformer")
    model = BigBirdForMaskedLM.from_pretrained("adibvafa/CodonTransformer")

    # Load the training data
    train_data = IterableJSONData(
        data_path=args.dataset_dir,
        train=True,
        dist_env="slurm" if not args.debug else None
    )

    # Print dataset info
    print(f"\nTotal examples in dataset: {train_data.total_examples}")
    print(f"Batch size per GPU: {args.batch_size}")
    print(f"Number of GPUs: {args.num_gpus}")
    print(f"Gradient accumulation steps: {args.accumulate_grad_batches}")

    total_steps = train_data.get_total_steps(
        batch_size=args.batch_size,
        n_gpus=args.num_gpus,
        grad_accum=args.accumulate_grad_batches
    )
    print(f"Calculated total training steps: {total_steps}\n")

    # Create harness
    harnessed_model = plTrainHarness(
        model=model,
        learning_rate=args.learning_rate,
        warmup_fraction=args.warmup_fraction,
        train_dataset=train_data,
        batch_size=args.batch_size,
        n_gpus=args.num_gpus,
        grad_accum=args.accumulate_grad_batches
    )

    # Create data loader
    data_loader = DataLoader(
        dataset=train_data,
        collate_fn=MaskedTokenizerCollator(tokenizer),
        batch_size=args.batch_size,
        num_workers=0 if args.debug else args.num_workers,
        persistent_workers=False if args.debug else True,
    )

    # Setup trainer and callbacks
    save_checkpoint = DumpStateDict(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_filename=args.checkpoint_filename,
        every_n_train_steps=args.save_every_n_steps,
    )

    # Early stopping callback
    early_stop = pl.callbacks.EarlyStopping(
        monitor='loss',
        min_delta=0.00,
        patience=3,
        verbose=True,
        mode='min'
    )

    trainer = pl.Trainer(
        default_root_dir=args.checkpoint_dir,
        strategy="ddp_find_unused_parameters_true",
        accelerator="gpu",
        devices=1 if args.debug else args.num_gpus,
        precision="16-mixed",
        max_epochs=args.max_epochs,
        deterministic=False,
        enable_checkpointing=True,
        callbacks=[save_checkpoint, early_stop],
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=1.0,  # Add gradient clipping
        log_every_n_steps=10,
    )

    # Finetune the model
    trainer.fit(harnessed_model, data_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune the CodonTransformer model.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory containing the dataset",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory where checkpoints will be saved",
    )
    parser.add_argument(
        "--checkpoint_filename",
        type=str,
        default="finetune.ckpt",
        help="Filename for the saved checkpoint",
    )
    parser.add_argument(
        "--batch_size", type=int, default=6, help="Batch size for training"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=15, help="Maximum number of epochs to train"
    )
    parser.add_argument(
        "--num_workers", type=int, default=5, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of batches to accumulate gradients",
    )
    parser.add_argument(
        "--num_gpus", type=int, default=4, help="Number of GPUs to use for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--warmup_fraction",
        type=float,
        default=0.1,
        help="Fraction of total steps to use for warmup",
    )
    parser.add_argument(
        "--save_every_n_steps",
        type=int,
        default=512,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed for reproducibility"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    main(args)

