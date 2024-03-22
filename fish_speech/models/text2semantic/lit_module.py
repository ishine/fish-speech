import itertools
import random
from dataclasses import dataclass
from typing import Any, Optional

import lightning as L
import loralib as lora
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRScheduler

import fish_speech.utils as utils
from fish_speech.models.text2semantic.llama import Transformer

log = utils.RankedLogger(__name__, rank_zero_only=True)


@dataclass
class LoraConfig:
    r: int
    lora_alpha: float
    lora_dropout: float = 0.0


class TextToSemantic(L.LightningModule):
    def __init__(
        self,
        ar_model: Transformer,
        nar_model: Transformer,
        optimizer: Any,
        lr_scheduler: Any,
        lora_config: Optional[LoraConfig] = None,
        save_lora_only: bool = False,
        use_dpo: bool = False,
        dpo_beta: float = 0.2,
    ):
        super().__init__()

        self.ar_model = ar_model
        self.nar_model = nar_model
        self.optimizer_builder = optimizer
        self.lr_scheduler_builder = lr_scheduler
        self.lora_config = lora_config
        self.save_lora_only = save_lora_only
        self.use_dpo = use_dpo  # We don't support reference model yet
        self.dpo_beta = dpo_beta

        if self.lora_config is not None:
            self.setup_lora()

    def setup_lora(self):
        # Replace the embedding layer with a LoRA layer
        self.ar_model.embeddings = lora.Embedding(
            num_embeddings=self.ar_model.embeddings.num_embeddings,
            embedding_dim=self.ar_model.embeddings.embedding_dim,
            padding_idx=self.ar_model.embeddings.padding_idx,
            r=self.lora_config.r,
            lora_alpha=self.lora_config.lora_alpha,
        )

        self.nar_model.embeddings = lora.Embedding(
            num_embeddings=self.nar_model.embeddings.num_embeddings,
            embedding_dim=self.nar_model.embeddings.embedding_dim,
            padding_idx=self.nar_model.embeddings.padding_idx,
            r=self.lora_config.r,
            lora_alpha=self.lora_config.lora_alpha,
        )

        # Replace output layer with a LoRA layer
        linears = [(self.ar_model, "output"), (self.nar_model, "output")]

        # Replace all linear layers with LoRA layers
        for layer in itertools.chain(self.ar_model.layers, self.nar_model.layers):
            linears.extend([(layer.attention, "wqkv"), (layer.attention, "wo")])
            linears.extend(
                [
                    (layer.feed_forward, "w1"),
                    (layer.feed_forward, "w2"),
                    (layer.feed_forward, "w3"),
                ]
            )

        for module, layer in linears:
            updated_linear = lora.Linear(
                in_features=getattr(module, layer).in_features,
                out_features=getattr(module, layer).out_features,
                bias=getattr(module, layer).bias,
                r=self.lora_config.r,
                lora_alpha=self.lora_config.lora_alpha,
                lora_dropout=self.lora_config.lora_dropout,
            )
            setattr(module, layer, updated_linear)

        # Mark only the LoRA layers as trainable
        lora.mark_only_lora_as_trainable(self.ar_model, bias="lora_only")
        lora.mark_only_lora_as_trainable(self.nar_model, bias="lora_only")

    def forward(self, x):
        return self.ar_model(x)

    def on_save_checkpoint(self, checkpoint):
        if self.lora_config is None or self.save_lora_only is False:
            return

        # Save only LoRA parameters
        state_dict = checkpoint["state_dict"]
        for name in list(state_dict.keys()):
            if "lora" not in name:
                state_dict.pop(name)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        # Get weight decay parameters
        weight_decay_parameters, other_parameters = [], []
        for name, param in self.named_parameters():
            if ".bias" in name or "norm.weight" in name or ".embeddings." in name:
                other_parameters.append(param)
            else:
                weight_decay_parameters.append(param)

        optimizer = self.optimizer_builder(
            [
                {"params": weight_decay_parameters},
                {"params": other_parameters, "weight_decay": 0.0},
            ]
        )

        # Print the parameters and their weight decay
        for i in optimizer.param_groups:
            log.info(
                f"Set weight decay: {i['weight_decay']} for {len(i['params'])} parameters"
            )

        lr_scheduler = self.lr_scheduler_builder(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }

    def get_accuracy(self, logits, labels):
        _, indices = logits.topk(5, dim=-1)
        correct = indices.eq(labels.unsqueeze(-1))
        correct[labels == -100] = 0
        correct = correct.sum()
        accuracy = correct / (labels != -100).sum()

        return accuracy

    # Copied from https://github.com/eric-mitchell/direct-preference-optimization/blob/main/trainers.py#L90
    def get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, codebook_size, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length, codebook_size)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        assert logits.shape[:-1] == labels.shape

        labels = labels.clone()
        loss_mask = labels != -100

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == -100] = 0

        per_token_logps = torch.gather(
            logits.log_softmax(-1), dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def _step(self, batch, batch_idx, stage: str):
        on_step = True if stage == "train" else False

        ############ Auto-regressive model ############
        # Do positive and negative samples in the same batch to speed up training
        ar_logits = self.ar_model(
            x=batch["inputs"][:, 0],
            key_padding_mask=batch["attention_masks"][:, 0],
        )
        ar_labels = batch["labels"][:, 0]

        # if self.use_dpo:
        #     # Firtst half is positive, second half is negative
        #     token_logits, negative_token_logits = token_logits.chunk(2)
        #     codebook_logits, negative_codebook_logits = codebook_logits.chunk(2)
        #     labels, negative_labels = labels.chunk(2)

        # Generate labels
        ar_loss = F.cross_entropy(
            ar_logits.reshape(-1, ar_logits.size(-1)),
            ar_labels.reshape(-1),
            ignore_index=-100,
        )

        # If we use dpo
        # if self.use_dpo:
        #     negative_codebook_labels = negative_labels[
        #         :, 1 : 1 + self.model.config.num_codebooks
        #     ].mT

        #     positive_codebook_logps = self.get_batch_logps(
        #         codebook_logits, codebook_labels
        #     )
        #     negative_codebook_logps = self.get_batch_logps(
        #         negative_codebook_logits, negative_codebook_labels
        #     )

        #     # TODO: implement the reference model, avoid screwing up the gradients
        #     dpo_loss = -F.logsigmoid(
        #         (positive_codebook_logps - negative_codebook_logps) * self.dpo_beta
        #     ).mean()

        #     chosen_rewards = self.dpo_beta * positive_codebook_logps.detach()
        #     rejected_rewards = self.dpo_beta * negative_codebook_logps.detach()
        #     reward_accuracy = (chosen_rewards > rejected_rewards).float().mean()
        #     chosen_rewards, rejected_rewards = (
        #         chosen_rewards.mean(),
        #         rejected_rewards.mean(),
        #     )

        #     loss = loss + dpo_loss

        #     self.log(
        #         f"{stage}/dpo_loss",
        #         dpo_loss,
        #         on_step=True,
        #         on_epoch=False,
        #         prog_bar=False,
        #         logger=True,
        #     )

        #     self.log(
        #         f"{stage}/chosen_rewards",
        #         chosen_rewards,
        #         on_step=True,
        #         on_epoch=False,
        #         prog_bar=False,
        #         logger=True,
        #     )

        #     self.log(
        #         f"{stage}/rejected_rewards",
        #         rejected_rewards,
        #         on_step=True,
        #         on_epoch=False,
        #         prog_bar=False,
        #         logger=True,
        #     )

        #     self.log(
        #         f"{stage}/reward_accuracy",
        #         reward_accuracy,
        #         on_step=True,
        #         on_epoch=False,
        #         prog_bar=False,
        #         logger=True,
        #     )

        self.log(
            f"{stage}/ar_loss",
            ar_loss,
            on_step=on_step,
            on_epoch=not on_step,
            prog_bar=True,
            logger=True,
            sync_dist=not on_step,
        )

        # Top-5 accuracy
        ar_accuracy = self.get_accuracy(ar_logits, ar_labels)
        self.log(
            f"{stage}/ar_top_5_accuracy",
            ar_accuracy,
            on_step=on_step,
            on_epoch=not on_step,
            prog_bar=True,
            logger=True,
            sync_dist=not on_step,
        )

        ############ Non-auto-regressive model ############
        bs, n_codebooks, n_tokens = batch["inputs"].size()
        nar_inputs = batch["inputs"][:, 1:].reshape(bs * (n_codebooks - 1), n_tokens)
        nar_attention_masks = batch["attention_masks"][:, 1:].reshape(
            bs * (n_codebooks - 1), n_tokens
        )
        nar_labels = batch["labels"][:, 1:].reshape(bs * (n_codebooks - 1), n_tokens)
        total_samples = nar_inputs.size(0)

        # Random select total_samples/(n_codebooks - 1) samples, to reduce the training time
        sample_indices = random.choices(
            range(0, total_samples), k=total_samples // (n_codebooks - 1)
        )
        nar_inputs = nar_inputs[sample_indices]
        nar_attention_masks = nar_attention_masks[sample_indices]
        nar_labels = nar_labels[sample_indices]

        nar_logits = self.nar_model(
            x=nar_inputs,
            key_padding_mask=nar_attention_masks,
        )

        nar_loss = F.cross_entropy(
            nar_logits.reshape(-1, nar_logits.size(-1)),
            nar_labels.reshape(-1),
            ignore_index=-100,
        )

        self.log(
            f"{stage}/nar_loss",
            nar_loss,
            on_step=on_step,
            on_epoch=not on_step,
            prog_bar=True,
            logger=True,
            sync_dist=not on_step,
        )

        # Top-5 accuracy
        nar_accuracy = self.get_accuracy(nar_logits, nar_labels)
        self.log(
            f"{stage}/nar_top_5_accuracy",
            nar_accuracy,
            on_step=on_step,
            on_epoch=not on_step,
            prog_bar=True,
            logger=True,
            sync_dist=not on_step,
        )

        return ar_loss + nar_loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")
