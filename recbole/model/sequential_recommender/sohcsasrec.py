# -*- coding: utf-8 -*-
# @File   : sohcsasrec.py

"""
SOHCSASRec
################################################

An advanced variant of SASRec incorporating Exact Special Orthogonal
Hyper-Connections via Cayley Transform (SOHC) and Dynamic Sigmoid Routing.
This architecture preserves isometric geometry in deeper networks, eliminating
multiplicative gradient decay and representation collapse.

Data Flow:
::

    Item/Position Embedding      [B, L, D]
               │
               ▼
+-----------------------------------------------------------+
| SOHCEncoder                                               |
|  (Stack of SOHCEncoderLayers)                             |
|                                                           |
| 1. Isometric Stream (Cayley + Newton-Schulz)              |
|    H_res ∈ SO(n) -> x_iso = x_l @ H_res                   |
| 2. Semantic Dynamic Routing:                              |
|    Gate_pre, Gate_post ∈ (0, 1)^d                         |
|    x_sem = Gate_post * F(Gate_pre * x_l)                  |
| 3. Additive Synthesis:                                    |
|    x_{l+1} = x_iso + x_sem                                |
+-----------------------------------------------------------+
               ▼
    FinalLayerNorm               [B, L, D]
               │ gather last item
               ▼
    seq_output                   [B, D]
"""

import torch
from torch import nn
from dataclasses import dataclass
from typing import Dict, Any

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import SOHCEncoder
from recbole.model.loss import BPRLoss


@dataclass
class SOHCConfig:
    """Dataclass encapsulating advanced SOHC-specific configurations."""
    ns_iterations: int = 7
    init_std: float = 0.02
    expansion_rate: int = 4   # N hyper-hidden channels (Wide-SOHC)


class SOHCSASRec(SequentialRecommender):
    r"""Exact Special Orthogonal Hyper-Connections SASRec.
    
    This model rigidly locks the residual representation mapping into SO(n) 
    using Cayley transforms and Newton-Schulz iterative approximations. 
    It leverages dynamic element-wise routing (Sigmoid) instead of standard 
    Softmax to balance inhibitory and excitatory contextual patterns.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # ── Load hyperparameters ──────────────────────────────────────────
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]
        
        # ── SOHC config extraction ─────────────────────────────────────────
        self.sohc_config = SOHCConfig(
            ns_iterations=config["ns_iterations"] if "ns_iterations" in config else 7,
            init_std=config["init_std"] if "init_std" in config else 0.02,
            expansion_rate=config["expansion_rate"] if "expansion_rate" in config else 4,
        )
        self.no_ortho = config["no_ortho"] if "no_ortho" in config else False
        self.static_routing = config["static_routing"] if "static_routing" in config else False
        self.use_standard_block = config["use_standard_block"] if "use_standard_block" in config else False
        self.use_softmax_routing = config["use_softmax_routing"] if "use_softmax_routing" in config else False

        # ── Embeddings ────────────────────────────────────────────────────
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(
            self.max_seq_length, self.hidden_size
        )
        self.input_layer_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.input_dropout = nn.Dropout(self.hidden_dropout_prob)

        # ── SOHC Encoder ───────────────────────────────────────────────
        self.sohc_encoder = SOHCEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            expansion_rate=self.sohc_config.expansion_rate,
            ns_iterations=self.sohc_config.ns_iterations,
            init_std=self.sohc_config.init_std,
            no_ortho=self.no_ortho,
            static_routing=self.static_routing,
            use_standard_block=self.use_standard_block,
            use_softmax_routing=self.use_softmax_routing,
        )

        # Final LayerNorm applied to sum output
        self.final_layer_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        # ── Loss Function ─────────────────────────────────────────────────
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("loss_type must be 'BPR' or 'CE'.")

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the weights properly. 
        Note: SOHC dynamic routing weights have custom normal definitions inside `layers.py`.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Ignore the custom init blocks created in layers.py
            if not getattr(module, '_is_custom_init', False):
                module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq: torch.Tensor, item_seq_len: torch.Tensor) -> torch.Tensor:
        """Encode an item sequence and return the contextual representation.

        Args:
            item_seq:     [B, L] - padded item ID sequence.
            item_seq_len: [B]    - actual (unpadded) lengths.

        Returns:
            seq_output:   [B, D] - valid ending representations for users.
        """
        # 1. Embedding
        pos_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        ).unsqueeze(0).expand_as(item_seq)

        item_emb = self.item_embedding(item_seq)
        pos_emb = self.position_embedding(pos_ids)
        input_emb = self.input_layer_norm(item_emb + pos_emb)
        input_emb = self.input_dropout(input_emb)         # [B, L, D]

        # 2. Expand to hyper-hidden matrix H_0 [B, L, N, D]
        N = self.sohc_config.expansion_rate
        H = input_emb.unsqueeze(2).expand(-1, -1, N, -1).clone()

        # 3. Causal Label Mask
        extended_mask = self.get_attention_mask(item_seq)

        # 4. Wide-SOHC Forward Prop
        sohc_out = self.sohc_encoder(
            H,
            extended_mask,
            output_all_encoded_layers=True
        )
        H_last = sohc_out[-1]                             # [B, L, N, D]

        # 5. Aggregate: sum N hyper-hidden vectors
        seq_output = H_last.sum(dim=2)                    # [B, L, D]

        # 6. Final Normalization & Gather Last Item
        seq_output = self.final_layer_norm(seq_output)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)  # [B, D]
        return seq_output

    def get_routing_data(self):
        """Capture dynamic gate weights from all SOHC layers."""
        routing_stats = {}
        for i, layer in enumerate(self.sohc_encoder.layers):
            # We capture the bias/weights or specific activations if needed
            # Here we capture the parameters that determine the routing
            routing_stats[f'layer_{i}_gate_pre_bias'] = layer.gate_pre.bias.detach().cpu().clone()
            routing_stats[f'layer_{i}_gate_post_bias'] = layer.gate_post.bias.detach().cpu().clone()
            # Also capture the actual SO(N) matrix being used
            with torch.no_grad():
                routing_stats[f'layer_{i}_orthogonal_matrix'] = layer._cayley_ortho().detach().cpu().clone()
        return routing_stats

    def calculate_loss(self, interaction: Dict[str, Any]) -> torch.Tensor:
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]

        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_emb = self.item_embedding(pos_items)
            neg_emb = self.item_embedding(neg_items)
            pos_score = (seq_output * pos_emb).sum(dim=-1)
            neg_score = (seq_output * neg_emb).sum(dim=-1)
            return self.loss_fct(pos_score, neg_score)
        else:  # CE
            # Full softmax over all internal items
            all_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, all_item_emb.T)
            return self.loss_fct(logits, pos_items)

    def predict(self, interaction: Dict[str, Any]) -> torch.Tensor:
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        
        seq_output = self.forward(item_seq, item_seq_len)
        test_emb = self.item_embedding(test_item)
        return (seq_output * test_emb).sum(dim=-1)

    def full_sort_predict(self, interaction: Dict[str, Any]) -> torch.Tensor:
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        
        seq_output = self.forward(item_seq, item_seq_len)
        all_item_emb = self.item_embedding.weight
        return torch.matmul(seq_output, all_item_emb.T)
