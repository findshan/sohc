# -*- coding: utf-8 -*-
# @File   : hcsasrec.py

"""
HCSASRec
################################################

A SASRec variant where residual connections are replaced by
Hyper-Connections (Zhu et al., 2024, arXiv:2409.19606).

Hyper-Connections generalise residual connections by introducing
a learnable (N+1)×(N+1) connection matrix that controls:
  - Width-connections: how the N hyper-hidden vectors are mixed
                       to produce the sub-layer input h0.
  - Depth-connections: how the sub-layer output is broadcast back
                       into the N hyper-hidden vectors.

Forward data flow:
::

    Item/Position Embedding  [B, L, D]
              │ expand(N)
              ▼
         H_0  [B, L, N, D]
              │
    ┌─────────┼─────────┐
    │  HyperConnection  │  × n_layers
    └─────────┼─────────┘
              ▼
         H_L  [B, L, N, D]
              │ sum over N
              ▼
    FinalLayerNorm  [B, L, D]
              │ gather last item
              ▼
    seq_output  [B, D]
"""

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import HyperTransformerEncoder
from recbole.model.loss import BPRLoss


class HCSASRec(SequentialRecommender):
    r"""SASRec with Hyper-Connections replacing standard residual connections.

    Config keys
    -----------
    expansion_rate : int
        Number of hyper-hidden vectors N.  Default: 4.
    dynamic_hc : bool
        Use dynamic (input-dependent) hyper-connections.  Default: True.
    hc_use_tanh : bool
        Apply tanh to the dynamic delta term.  Default: True.

    All standard SASRec keys (n_layers, n_heads, hidden_size, inner_size,
    hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps,
    initializer_range, loss_type) are also supported.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # ── Load hyperparameters ──────────────────────────────────────────
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]      # same as embedding_size
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]

        # HC-specific
        self.expansion_rate = config["expansion_rate"] if "expansion_rate" in config else 4
        self.dynamic_hc = config["dynamic_hc"] if "dynamic_hc" in config else True
        self.hc_use_tanh = config["hc_use_tanh"] if "hc_use_tanh" in config else True

        # ── Embeddings ────────────────────────────────────────────────────
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.input_layer_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.input_dropout = nn.Dropout(self.hidden_dropout_prob)

        # ── Encoder ───────────────────────────────────────────────────────
        self.hc_encoder = HyperTransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            expansion_rate=self.expansion_rate,
            dynamic_hc=self.dynamic_hc,
            hc_use_tanh=self.hc_use_tanh,
        )

        # Final LayerNorm applied to the summed output (before prediction)
        # Equivalent to the "normalization layer" before unembedding in the paper.
        self.final_layer_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        # ── Loss ──────────────────────────────────────────────────────────
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("loss_type must be 'BPR' or 'CE'.")

        # ── Weight initialisation ─────────────────────────────────────────
        # NOTE: HyperConnection static parameters (static_alpha, static_beta)
        #       are initialised in HyperConnection.__init__ following Eq. 14
        #       and must NOT be overwritten here.
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialise non-HC weights following SASRec convention."""
        # Skip HC static parameter tensors (already custom-initialised)
        # They are nn.Parameter directly on HyperConnection, not sub-modules.
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq: torch.Tensor, item_seq_len: torch.Tensor) -> torch.Tensor:
        """Encode an item sequence and return the last-item representation.

        Args:
            item_seq:     [B, L] – padded item ID sequence.
            item_seq_len: [B]    – actual (unpadded) lengths.

        Returns:
            seq_output: [B, D]
        """
        # ── Input embedding ───────────────────────────────────────────────
        pos_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        ).unsqueeze(0).expand_as(item_seq)                          # [B, L]

        item_emb = self.item_embedding(item_seq)                    # [B, L, D]
        pos_emb = self.position_embedding(pos_ids)                  # [B, L, D]
        input_emb = self.input_layer_norm(item_emb + pos_emb)
        input_emb = self.input_dropout(input_emb)                   # [B, L, D]

        # ── Expand to hyper-hidden matrix H_0 ────────────────────────────
        # H_0 = [h0, h0, ..., h0]^T  (N copies)  shape [B, L, N, D]
        H = input_emb.unsqueeze(2).expand(
            -1, -1, self.expansion_rate, -1
        ).clone()   # clone so each slice can diverge during backward

        # ── Causal attention mask ─────────────────────────────────────────
        extended_mask = self.get_attention_mask(item_seq)           # [B, 1, L, L]

        # ── Hyper-Connection encoder ──────────────────────────────────────
        hc_out = self.hc_encoder(H, extended_mask, output_all_encoded_layers=True)
        H_last = hc_out[-1]                                         # [B, L, N, D]

        # ── Aggregate: sum over N hyper-hidden vectors (Algorithm 1 L11) ──
        seq_output = H_last.sum(dim=2)                              # [B, L, D]
        seq_output = self.final_layer_norm(seq_output)              # [B, L, D]

        # ── Gather the last valid item ────────────────────────────────────
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)  # [B, D]
        return seq_output

    def calculate_loss(self, interaction) -> torch.Tensor:
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)           # [B, D]
        pos_items = interaction[self.POS_ITEM_ID]

        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_emb = self.item_embedding(pos_items)                # [B, D]
            neg_emb = self.item_embedding(neg_items)                # [B, D]
            pos_score = (seq_output * pos_emb).sum(dim=-1)          # [B]
            neg_score = (seq_output * neg_emb).sum(dim=-1)          # [B]
            return self.loss_fct(pos_score, neg_score)
        else:  # CE
            # Full softmax over all items
            all_item_emb = self.item_embedding.weight               # [n_items, D]
            logits = torch.matmul(seq_output, all_item_emb.T)       # [B, n_items]
            return self.loss_fct(logits, pos_items)

    def predict(self, interaction) -> torch.Tensor:
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)           # [B, D]
        test_emb = self.item_embedding(test_item)                   # [B, D]
        return (seq_output * test_emb).sum(dim=-1)                  # [B]

    def full_sort_predict(self, interaction) -> torch.Tensor:
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)           # [B, D]
        all_item_emb = self.item_embedding.weight                   # [n_items, D]
        return torch.matmul(seq_output, all_item_emb.T)             # [B, n_items]
