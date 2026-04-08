# -*- coding: utf-8 -*-
# @File   : mhcsasrec.py

"""
mHCSASRec
################################################

A SASRec variant where standard residual connections are replaced by
Manifold-Constrained Hyper-Connections (mHC, DeepSeek).

According to Zhu et al. (2025, arXiv:2512.24880):
mHC projects the residual connection space of Hyper-Connections onto a specific
manifold (doubly stochastic matrices) to restore the identity mapping property.
This is achieved using the Sinkhorn-Knopp algorithm, ensuring training stability
and unrestricted scalability without compromising performance.

Algorithm 1 Forward Pass of mHC:
::

    Input: x_l [B, L, n, c]
    x_vec = RMSNorm(Flatten(x_l)) [B, L, n*c]
    H_pre = Sigmoid(alpha * Linear(x_vec) + b) [B, L, 1, n]
    H_post = 2 * Sigmoid(alpha * Linear(x_vec) + b) [B, L, n, 1]
    H_res = SinkhornKnopp(alpha * Linear(x_vec) + b) [B, L, n, n]

    f_pre = (H_pre @ x_l).squeeze(2) [B, L, c]
    f_layer =  Attention/FFN (f_pre) [B, L, c]

    x_res = H_res @ x_l [B, L, n, c]
    x_proj = H_post @ f_layer.unsqueeze(2) [B, L, n, c]

    x_{l+1} = x_res + x_proj
"""

import math
import torch
from torch import nn
from typing import Tuple

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import _PureMultiHeadAttention, _PureFeedForward
from recbole.model.loss import BPRLoss


def sinkhorn_knopp(adj_matrix: torch.Tensor, max_iters: int = 20) -> torch.Tensor:
    """
    Apply Sinkhorn-Knopp algorithm to project matrix into Birkhoff Polytope (Doubly Stochastic Matrix).

    Args:
        adj_matrix (torch.Tensor): The pre-exponential adjacency matrix (logits) of shape [..., N, N].
        max_iters (int): The number of iterations to perform. Default is 20 (from paper Table 5).

    Returns:
        torch.Tensor: The doubly stochastic matrix of shape [..., N, N].
    """
    matrix = torch.exp(adj_matrix)
    for _ in range(max_iters):
        # Column Normalization (Tc): sum over rows (dim=-2)
        matrix = matrix / matrix.sum(dim=-2, keepdim=True).clamp(min=1e-8)
        # Row Normalization (Tr): sum over cols (dim=-1)
        matrix = matrix / matrix.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return matrix


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate RMSNorm
        # x: [B, L, n*c]
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_normed


class ManifoldHyperRouter(nn.Module):
    """
    Calculate Manifold-Constrained Hyper-Connections routers: H_pre, H_post, H_res.
    """
    def __init__(
        self,
        hidden_size: int,
        expansion_rate: int,
        sk_iters: int = 20,
        factor_init: float = 0.01,
        layer_norm_eps: float = 1e-8,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n = expansion_rate
        self.sk_iters = sk_iters
        self.flat_dim = self.n * self.hidden_size

        # 1. RMSNorm processing flattened input
        self.norm = RMSNorm(self.flat_dim, eps=layer_norm_eps)

        # 2. Linear Projections (No explicit N(0, 0.02) per paper, standard init)
        self.proj_pre = nn.Linear(self.flat_dim, self.n)
        self.proj_post = nn.Linear(self.flat_dim, self.n)
        self.proj_res = nn.Linear(self.flat_dim, self.n * self.n)

        # 3. Alpha Scaling Factors (Init to 0.01 as per Table 5)
        self.alpha_pre = nn.Parameter(torch.tensor(factor_init))
        self.alpha_post = nn.Parameter(torch.tensor(factor_init))
        self.alpha_res = nn.Parameter(torch.tensor(factor_init))

        self._initialize_strict_priors()

    def _initialize_strict_priors(self):
        """
        Enforce strict static bias initialization to maintain Identity Mapping Property initially.
        """
        # H_pre goes to 1/n through Sigmoid -> b_pre = ln(1 / (n-1))
        # Sigmoid(x) = 1/n => 1 / (1 + e^-x) = 1/n => 1 + e^-x = n => e^-x = n-1 => x = -ln(n-1)
        if self.n > 1:
            val_pre = -math.log(self.n - 1)
        else:
            val_pre = 0.0
        torch.nn.init.constant_(self.proj_pre.bias, val_pre)

        # H_post goes to 1 through 2*Sigmoid -> b_post = 0
        # 2*Sigmoid(x) = 1 => Sigmoid(x) = 0.5 => x = 0
        torch.nn.init.constant_(self.proj_post.bias, 0.0)

        # H_res goes to Identity (I) through Sinkhorn
        # Require diagonal >> off-diagonal before exponentiation
        res_bias = torch.zeros(self.n, self.n)
        res_bias.fill_(0.0)
        res_bias.fill_diagonal_(3.0) # strong diagonal prior
        self.proj_res.bias.data.copy_(res_bias.view(-1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, L, n, c]
        Returns:
            H_pre: [B, L, 1, n]
            H_post: [B, L, n, 1]
            H_res: [B, L, n, n]
        """
        B, L, N, C = x.shape
        # 1. Flatten and Norm
        x_vec = x.view(B, L, N * C) # [B, L, n*c]
        x_norm = self.norm(x_vec)

        # 2. Linear projection and scaling
        # (Weight * x + Bias_priors) * alpha -- Alpha scales weight dynamics but bias stays raw to keep prior
        # Wait, the paper: H_res = Sinkhorn( alpha * mat(x * phi) + b )
        # Meaning the linear output IS the dynamic part.
        # But nn.Linear includes both. So we apply alpha only to the weight dot product.
        # Let's adjust manually:
        
        dyn_pre = torch.matmul(x_norm, self.proj_pre.weight.t())
        logits_pre = self.alpha_pre * dyn_pre + self.proj_pre.bias

        dyn_post = torch.matmul(x_norm, self.proj_post.weight.t())
        logits_post = self.alpha_post * dyn_post + self.proj_post.bias

        dyn_res = torch.matmul(x_norm, self.proj_res.weight.t())
        logits_res = self.alpha_res * dyn_res + self.proj_res.bias

        # 3. Manifold Constraint Mappings
        H_pre = torch.sigmoid(logits_pre).unsqueeze(-2) # [B, L, 1, n]
        H_post = (2.0 * torch.sigmoid(logits_post)).unsqueeze(-1) # [B, L, n, 1]
        
        logits_res_mat = logits_res.view(B, L, N, N)
        H_res = sinkhorn_knopp(logits_res_mat, self.sk_iters) # [B, L, n, n]

        return H_pre, H_post, H_res


class ManifoldHyperConnectionLayer(nn.Module):
    """
    A single Hyper-Connection block combining mHC routing and transformer sub-layers (Attn + FFN).
    """
    def __init__(
        self,
        hidden_size: int,
        expansion_rate: int,
        n_heads: int,
        inner_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        hidden_act: str,
        layer_norm_eps: float,
        sk_iters: int,
        factor_init: float,
    ):
        super().__init__()
        self.router = ManifoldHyperRouter(
            hidden_size=hidden_size,
            expansion_rate=expansion_rate,
            sk_iters=sk_iters,
            factor_init=factor_init,
            layer_norm_eps=layer_norm_eps,
        )

        self.pure_attn = _PureMultiHeadAttention(
            n_heads=n_heads,
            hidden_size=hidden_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attn_dropout_prob=attn_dropout_prob,
            layer_norm_eps=layer_norm_eps,
            expansion_rate=expansion_rate, # Scaling 1/sqrt(n) inside
        )

        self.pure_ffn = _PureFeedForward(
            hidden_size=hidden_size,
            inner_size=inner_size,
            hidden_dropout_prob=hidden_dropout_prob,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            expansion_rate=expansion_rate, # Scaling 1/sqrt(n) inside
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, n, c]
            attention_mask: [B, 1, 1, L]
        """
        # 1. Routing Matrices
        # H_pre: [B, L, 1, n] | H_post: [B, L, n, 1] | H_res: [B, L, n, n]
        H_pre, H_post, H_res = self.router(x)

        # 2. Extract input for layer
        # [B, L, 1, n] @ [B, L, n, c] -> [B, L, 1, c] -> squeeze -> [B, L, c]
        f_pre = torch.matmul(H_pre, x).squeeze(-2)
        
        # 3. Process with Pure_Attention and Pure_FFN (Sub-Layer Computation F)
        # Note: Following standard transformers, attention needs its own minimal inner residual 
        # to process tokens properly over sequence before projection out.
        f_attn = self.pure_attn(f_pre, attention_mask) + f_pre 
        f_layer = self.pure_ffn(f_attn) # No FFN residual, handled by outer H_res
        
        # 4. Multiply and merge connections
        # x_res: [B, L, n, n] @ [B, L, n, c] -> [B, L, n, c]
        x_res = torch.matmul(H_res, x)
        
        # x_proj: [B, L, n, 1] @ [B, L, 1, c] -> [B, L, n, c]
        x_proj = torch.matmul(H_post, f_layer.unsqueeze(-2))
        
        return x_res + x_proj


class mHCTransformerEncoder(nn.Module):
    """
    Stack of ManifoldHyperConnectionLayer blocks.
    """
    def __init__(
        self,
        n_layers: int,
        hidden_size: int,
        expansion_rate: int,
        n_heads: int,
        inner_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        hidden_act: str,
        layer_norm_eps: float,
        sk_iters: int,
        factor_init: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ManifoldHyperConnectionLayer(
                    hidden_size=hidden_size,
                    expansion_rate=expansion_rate,
                    n_heads=n_heads,
                    inner_size=inner_size,
                    hidden_dropout_prob=hidden_dropout_prob,
                    attn_dropout_prob=attn_dropout_prob,
                    hidden_act=hidden_act,
                    layer_norm_eps=layer_norm_eps,
                    sk_iters=sk_iters,
                    factor_init=factor_init,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        H: torch.Tensor,
        attention_mask: torch.Tensor,
        output_all_encoded_layers: bool = True,
    ):
        all_layers = []
        for layer in self.layers:
            H = layer(H, attention_mask)
            if output_all_encoded_layers:
                all_layers.append(H)
        if not output_all_encoded_layers:
            all_layers.append(H)
        return all_layers


class mHCSASRec(SequentialRecommender):
    r"""SASRec with Manifold-Constrained Hyper-Connections replacing standard residual connections.

    Config keys
    -----------
    expansion_rate : int
        Number of hyper-hidden vectors N (n). Default: 4.
    sk_iters : int
        Iterations for Sinkhorn-Knopp algorithm. Default: 20.
    factor_init : float
        Initialization value for dynamic routing alpha scalars. Default: 0.01.

    All standard SASRec keys are supported.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # Hyperparameters
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

        # mHC-specific
        self.expansion_rate = config["expansion_rate"] if "expansion_rate" in config else 4
        self.sk_iters = config["sk_iters"] if "sk_iters" in config else 20
        self.factor_init = config["factor_init"] if "factor_init" in config else 0.01

        # Embeddings
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.input_layer_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.input_dropout = nn.Dropout(self.hidden_dropout_prob)

        # Encoder
        self.mhc_encoder = mHCTransformerEncoder(
            n_layers=self.n_layers,
            hidden_size=self.hidden_size,
            expansion_rate=self.expansion_rate,
            n_heads=self.n_heads,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            sk_iters=self.sk_iters,
            factor_init=self.factor_init,
        )

        self.final_layer_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("loss_type must be 'BPR' or 'CE'.")

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialise non-mHC weights following SASRec convention."""
        # Note: ManifoldHyperRouter has its own specific bias initializations which shouldn't be overridden
        if isinstance(module, ManifoldHyperRouter):
            return
        
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm) and not isinstance(module, RMSNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq: torch.Tensor, item_seq_len: torch.Tensor) -> torch.Tensor:
        pos_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        ).unsqueeze(0).expand_as(item_seq)

        item_emb = self.item_embedding(item_seq)                    # [B, L, D]
        pos_emb = self.position_embedding(pos_ids)                  # [B, L, D]
        input_emb = self.input_layer_norm(item_emb + pos_emb)
        input_emb = self.input_dropout(input_emb)                   # [B, L, D]

        # Expand to hyper-hidden matrix [B, L, n, c]
        H = input_emb.unsqueeze(-2).expand(
            -1, -1, self.expansion_rate, -1
        ).clone()

        extended_mask = self.get_attention_mask(item_seq)

        # Forward through layers
        mhc_out = self.mhc_encoder(H, extended_mask, output_all_encoded_layers=True)
        H_last = mhc_out[-1]                                        # [B, L, n, c]

        # Aggregate: sum over n hyper-hidden vectors
        seq_output = H_last.sum(dim=-2)                             # [B, L, c]
        seq_output = self.final_layer_norm(seq_output)              # [B, L, c]

        # Gather last item
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)  # [B, c]
        return seq_output

    def get_routing_data(self):
        """Capture mHC manifold routing parameters."""
        routing_stats = {}
        for i, layer in enumerate(self.mhc_encoder.layers):
            # Capture the scalar factors (alpha) and priors (bias)
            routing_stats[f'layer_{i}_alpha_res'] = layer.router.alpha_res.detach().cpu().clone()
            routing_stats[f'layer_{i}_alpha_pre'] = layer.router.alpha_pre.detach().cpu().clone()
            routing_stats[f'layer_{i}_alpha_post'] = layer.router.alpha_post.detach().cpu().clone()
            # Capture res bias which holds the identity prior
            routing_stats[f'layer_{i}_proj_res_bias'] = layer.router.proj_res.bias.detach().cpu().clone()
        return routing_stats

    def calculate_loss(self, interaction) -> torch.Tensor:
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
        else:
            all_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, all_item_emb.T)
            return self.loss_fct(logits, pos_items)

    def predict(self, interaction) -> torch.Tensor:
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_emb = self.item_embedding(test_item)
        return (seq_output * test_emb).sum(dim=-1)

    def full_sort_predict(self, interaction) -> torch.Tensor:
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        all_item_emb = self.item_embedding.weight
        return torch.matmul(seq_output, all_item_emb.T)
