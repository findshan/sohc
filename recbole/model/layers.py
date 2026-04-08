# -*- coding: utf-8 -*-
# @Time   : 2020/6/27 16:40
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : layers.py

# UPDATE:
# @Time   : 2022/7/16, 2020/8/24 14:58, 2020/9/16, 2020/9/21, 2020/10/9, 2021/05/01
# @Author : Zhen Tian, Yujie Lu, Xingyu Pan, Zhichao Feng, Hui Wang, Xinyan Fan
# @Email  : chenyuwuxinn@gmail.com, yujielu1998@gmail.com, panxy@ruc.edu.cn, fzcbupt@gmail.com, hui.wang@ruc.edu.cn, xinyan.fan@ruc.edu.cn

"""
recbole.model.layers
#############################
Common Layers in recommender system
"""

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch.nn.init import normal_

from recbole.utils import FeatureType, FeatureSource


class MLPLayers(nn.Module):
    r"""MLPLayers

    Args:
        - layers(list): a list contains the size of each layer in mlp layers
        - dropout(float): probability of an element to be zeroed. Default: 0
        - activation(str): activation function after each layer in mlp layers. Default: 'relu'.
                           candidates: 'sigmoid', 'tanh', 'relu', 'leekyrelu', 'none'

    Shape:

        - Input: (:math:`N`, \*, :math:`H_{in}`) where \* means any number of additional dimensions
          :math:`H_{in}` must equal to the first value in `layers`
        - Output: (:math:`N`, \*, :math:`H_{out}`) where :math:`H_{out}` equals to the last value in `layers`

    Examples::

        >>> m = MLPLayers([64, 32, 16], 0.2, 'relu')
        >>> input = torch.randn(128, 64)
        >>> output = m(input)
        >>> print(output.size())
        >>> torch.Size([128, 16])
    """

    def __init__(
        self,
        layers,
        dropout=0.0,
        activation="relu",
        bn=False,
        init_method=None,
        last_activation=True,
    ):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = bn
        self.init_method = init_method

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(
            zip(self.layers[:-1], self.layers[1:])
        ):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            if self.use_bn:
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            activation_func = activation_layer(self.activation, output_size)
            if activation_func is not None:
                mlp_modules.append(activation_func)
        if self.activation is not None and not last_activation:
            mlp_modules.pop()
        self.mlp_layers = nn.Sequential(*mlp_modules)
        if self.init_method is not None:
            self.apply(self.init_weights)

    def init_weights(self, module):
        # We just initialize the module with normal distribution as the paper said
        if isinstance(module, nn.Linear):
            if self.init_method == "norm":
                normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)


def activation_layer(activation_name="relu", emb_dim=None):
    """Construct activation layers

    Args:
        activation_name: str, name of activation function
        emb_dim: int, used for Dice activation

    Return:
        activation: activation layer
    """
    if activation_name is None:
        activation = None
    elif isinstance(activation_name, str):
        if activation_name.lower() == "sigmoid":
            activation = nn.Sigmoid()
        elif activation_name.lower() == "tanh":
            activation = nn.Tanh()
        elif activation_name.lower() == "relu":
            activation = nn.ReLU()
        elif activation_name.lower() == "leakyrelu":
            activation = nn.LeakyReLU()
        elif activation_name.lower() == "dice":
            activation = Dice(emb_dim)
        elif activation_name.lower() == "none":
            activation = None
    elif issubclass(activation_name, nn.Module):
        activation = activation_name()
    else:
        raise NotImplementedError(
            "activation function {} is not implemented".format(activation_name)
        )

    return activation


class FMEmbedding(nn.Module):
    r"""Embedding for token fields.

    Args:
        field_dims: list, the number of tokens in each token fields
        offsets: list, the dimension offset of each token field
        embed_dim: int, the dimension of output embedding vectors

    Input:
        input_x: tensor, A 3D tensor with shape:``(batch_size,field_size)``.

    Return:
        output: tensor,  A 3D tensor with shape: ``(batch_size,field_size,embed_dim)``.
    """

    def __init__(self, field_dims, offsets, embed_dim):
        super(FMEmbedding, self).__init__()
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = offsets

    def forward(self, input_x):
        input_x = input_x + input_x.new_tensor(self.offsets).unsqueeze(0)
        output = self.embedding(input_x)
        return output


class FLEmbedding(nn.Module):
    r"""Embedding for float fields.

    Args:
        field_dims: list, the number of float in each float fields
        offsets: list, the dimension offset of each float field
        embed_dim: int, the dimension of output embedding vectors

    Input:
        input_x: tensor, A 3D tensor with shape:``(batch_size,field_size,2)``.

    Return:
        output: tensor,  A 3D tensor with shape: ``(batch_size,field_size,embed_dim)``.
    """

    def __init__(self, field_dims, offsets, embed_dim):
        super(FLEmbedding, self).__init__()
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = offsets

    def forward(self, input_x):
        base, index = torch.split(input_x, [1, 1], dim=-1)
        index = index.squeeze(-1).long()
        index = index + index.new_tensor(self.offsets).unsqueeze(0)
        output = base * self.embedding(index)
        return output


class BaseFactorizationMachine(nn.Module):
    r"""Calculate FM result over the embeddings

    Args:
        reduce_sum: bool, whether to sum the result, default is True.

    Input:
        input_x: tensor, A 3D tensor with shape:``(batch_size,field_size,embed_dim)``.

    Output
        output: tensor, A 3D tensor with shape: ``(batch_size,1)`` or ``(batch_size, embed_dim)``.
    """

    def __init__(self, reduce_sum=True):
        super(BaseFactorizationMachine, self).__init__()
        self.reduce_sum = reduce_sum

    def forward(self, input_x):
        square_of_sum = torch.sum(input_x, dim=1) ** 2
        sum_of_square = torch.sum(input_x**2, dim=1)
        output = square_of_sum - sum_of_square
        if self.reduce_sum:
            output = torch.sum(output, dim=1, keepdim=True)
        output = 0.5 * output
        return output


class BiGNNLayer(nn.Module):
    r"""Propagate a layer of Bi-interaction GNN

    .. math::
        output = (L+I)EW_1 + LE \otimes EW_2
    """

    def __init__(self, in_dim, out_dim):
        super(BiGNNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = torch.nn.Linear(in_features=in_dim, out_features=out_dim)
        self.interActTransform = torch.nn.Linear(
            in_features=in_dim, out_features=out_dim
        )

    def forward(self, lap_matrix, eye_matrix, features):
        # for GCF ajdMat is a (N+M) by (N+M) mat
        # lap_matrix L = D^-1(A)D^-1 # 拉普拉斯矩阵
        x = torch.sparse.mm(lap_matrix, features)

        inter_part1 = self.linear(features + x)
        inter_feature = torch.mul(x, features)
        inter_part2 = self.interActTransform(inter_feature)

        return inter_part1 + inter_part2


class AttLayer(nn.Module):
    """Calculate the attention signal(weight) according the input tensor.

    Args:
        infeatures (torch.FloatTensor): A 3D input tensor with shape of[batch_size, M, embed_dim].

    Returns:
        torch.FloatTensor: Attention weight of input. shape of [batch_size, M].
    """

    def __init__(self, in_dim, att_dim):
        super(AttLayer, self).__init__()
        self.in_dim = in_dim
        self.att_dim = att_dim
        self.w = torch.nn.Linear(in_features=in_dim, out_features=att_dim, bias=False)
        self.h = nn.Parameter(torch.randn(att_dim), requires_grad=True)

    def forward(self, infeatures):
        att_signal = self.w(infeatures)  # [batch_size, M, att_dim]
        att_signal = fn.relu(att_signal)  # [batch_size, M, att_dim]

        att_signal = torch.mul(att_signal, self.h)  # [batch_size, M, att_dim]
        att_signal = torch.sum(att_signal, dim=2)  # [batch_size, M]
        att_signal = fn.softmax(att_signal, dim=1)  # [batch_size, M]

        return att_signal


class Dice(nn.Module):
    r"""Dice activation function

    .. math::
        f(s)=p(s) \cdot s+(1-p(s)) \cdot \alpha s

    .. math::
        p(s)=\frac{1} {1 + e^{-\frac{s-E[s]} {\sqrt {Var[s] + \epsilon}}}}
    """

    def __init__(self, emb_size):
        super(Dice, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.alpha = torch.zeros((emb_size,))

    def forward(self, score):
        self.alpha = self.alpha.to(score.device)
        score_p = self.sigmoid(score)

        return self.alpha * (1 - score_p) * score + score_p * score


class SequenceAttLayer(nn.Module):
    """Attention Layer. Get the representation of each user in the batch.

    Args:
        queries (torch.Tensor): candidate ads, [B, H], H means embedding_size * feat_num
        keys (torch.Tensor): user_hist, [B, T, H]
        keys_length (torch.Tensor): mask, [B]

    Returns:
        torch.Tensor: result
    """

    def __init__(
        self,
        mask_mat,
        att_hidden_size=(80, 40),
        activation="sigmoid",
        softmax_stag=False,
        return_seq_weight=True,
    ):
        super(SequenceAttLayer, self).__init__()
        self.att_hidden_size = att_hidden_size
        self.activation = activation
        self.softmax_stag = softmax_stag
        self.return_seq_weight = return_seq_weight
        self.mask_mat = mask_mat
        self.att_mlp_layers = MLPLayers(
            self.att_hidden_size, activation=self.activation, bn=False
        )
        self.dense = nn.Linear(self.att_hidden_size[-1], 1)

    def forward(self, queries, keys, keys_length):
        embedding_size = queries.shape[-1]  # H
        hist_len = keys.shape[1]  # T
        queries = queries.repeat(1, hist_len)

        queries = queries.view(-1, hist_len, embedding_size)

        # MLP Layer
        input_tensor = torch.cat(
            [queries, keys, queries - keys, queries * keys], dim=-1
        )
        output = self.att_mlp_layers(input_tensor)
        output = torch.transpose(self.dense(output), -1, -2)

        # get mask
        output = output.squeeze(1)
        mask = self.mask_mat.repeat(output.size(0), 1)
        mask = mask >= keys_length.unsqueeze(1)

        # mask
        if self.softmax_stag:
            mask_value = -np.inf
        else:
            mask_value = 0.0

        output = output.masked_fill(mask=mask, value=torch.tensor(mask_value))
        output = output.unsqueeze(1)
        output = output / (embedding_size**0.5)

        # get the weight of each user's history list about the target item
        if self.softmax_stag:
            output = fn.softmax(output, dim=2)  # [B, 1, T]

        if not self.return_seq_weight:
            output = torch.matmul(output, keys)  # [B, 1, H]

        return output


class VanillaAttention(nn.Module):
    """
    Vanilla attention layer is implemented by linear layer.

    Args:
        input_tensor (torch.Tensor): the input of the attention layer

    Returns:
        hidden_states (torch.Tensor): the outputs of the attention layer
        weights (torch.Tensor): the attention weights

    """

    def __init__(self, hidden_dim, attn_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, attn_dim), nn.ReLU(True), nn.Linear(attn_dim, 1)
        )

    def forward(self, input_tensor):
        # (B, Len, num, H) -> (B, Len, num, 1)
        energy = self.projection(input_tensor)
        weights = torch.softmax(energy.squeeze(-1), dim=-1)
        # (B, Len, num, H) * (B, Len, num, 1) -> (B, len, H)
        hidden_states = (input_tensor * weights.unsqueeze(-1)).sum(dim=-2)
        return hidden_states, weights


class MultiHeadAttention(nn.Module):
    """
    Multi-head Self-attention layers, a attention score dropout layer is introduced.

    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor

    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer

    """

    def __init__(
        self,
        n_heads,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
    ):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer)

        attention_scores = attention_scores / self.sqrt_attention_head_size
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class FeedForward(nn.Module):
    """
    Point-wise feed-forward layer is implemented by two dense layers.

    Args:
        input_tensor (torch.Tensor): the input of the point-wise feed-forward layer

    Returns:
        hidden_states (torch.Tensor): the output of the point-wise feed-forward layer

    """

    def __init__(
        self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps
    ):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)

        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": fn.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class TransformerLayer(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.

    """

    def __init__(
        self,
        n_heads,
        hidden_size,
        intermediate_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
    ):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.feed_forward = FeedForward(
            hidden_size,
            intermediate_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )

    def forward(self, hidden_states, attention_mask):
        attention_output = self.multi_head_attention(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class TransformerEncoder(nn.Module):
    r"""One TransformerEncoder consists of several TransformerLayers.

    Args:
        n_layers(num): num of transformer layers in transformer encoder. Default: 2
        n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        hidden_size(num): the input and output hidden size. Default: 64
        inner_size(num): the dimensionality in feed-forward layer. Default: 256
        hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12

    """

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
    ):
        super(TransformerEncoder, self).__init__()
        layer = TransformerLayer(
            n_heads,
            hidden_size,
            inner_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.

        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class ItemToInterestAggregation(nn.Module):
    def __init__(self, seq_len, hidden_size, k_interests=5):
        super().__init__()
        self.k_interests = k_interests  # k latent interests
        self.theta = nn.Parameter(torch.randn([hidden_size, k_interests]))

    def forward(self, input_tensor):  # [B, L, d] -> [B, k, d]
        D_matrix = torch.matmul(input_tensor, self.theta)  # [B, L, k]
        D_matrix = nn.Softmax(dim=-2)(D_matrix)
        result = torch.einsum("nij, nik -> nkj", input_tensor, D_matrix)  # #[B, k, d]

        return result


class LightMultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_heads,
        k_interests,
        hidden_size,
        seq_len,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
    ):
        super(LightMultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # initialization for low-rank decomposed self-attention
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.attpooling_key = ItemToInterestAggregation(
            seq_len, hidden_size, k_interests
        )
        self.attpooling_value = ItemToInterestAggregation(
            seq_len, hidden_size, k_interests
        )

        # initialization for decoupled position encoding
        self.attn_scale_factor = 2
        self.pos_q_linear = nn.Linear(hidden_size, self.all_head_size)
        self.pos_k_linear = nn.Linear(hidden_size, self.all_head_size)
        self.pos_scaling = (
            float(self.attention_head_size * self.attn_scale_factor) ** -0.5
        )
        self.pos_ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):  # transfor to multihead
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, pos_emb):
        # linear map
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        # low-rank decomposed self-attention: relation of items
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(self.attpooling_key(mixed_key_layer))
        value_layer = self.transpose_for_scores(
            self.attpooling_value(mixed_value_layer)
        )

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-2)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer_item = torch.matmul(attention_probs, value_layer)

        # decoupled position encoding: relation of positions
        value_layer_pos = self.transpose_for_scores(mixed_value_layer)
        pos_emb = self.pos_ln(pos_emb).unsqueeze(0)
        pos_query_layer = (
            self.transpose_for_scores(self.pos_q_linear(pos_emb)) * self.pos_scaling
        )
        pos_key_layer = self.transpose_for_scores(self.pos_k_linear(pos_emb))

        abs_pos_bias = torch.matmul(pos_query_layer, pos_key_layer.transpose(-1, -2))
        abs_pos_bias = abs_pos_bias / math.sqrt(self.attention_head_size)
        abs_pos_bias = nn.Softmax(dim=-2)(abs_pos_bias)

        context_layer_pos = torch.matmul(abs_pos_bias, value_layer_pos)

        context_layer = context_layer_item + context_layer_pos

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class LightTransformerLayer(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): the output of the point-wise feed-forward sublayer, is the output of the transformer layer
    """

    def __init__(
        self,
        n_heads,
        k_interests,
        hidden_size,
        seq_len,
        intermediate_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
    ):
        super(LightTransformerLayer, self).__init__()
        self.multi_head_attention = LightMultiHeadAttention(
            n_heads,
            k_interests,
            hidden_size,
            seq_len,
            hidden_dropout_prob,
            attn_dropout_prob,
            layer_norm_eps,
        )
        self.feed_forward = FeedForward(
            hidden_size,
            intermediate_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )

    def forward(self, hidden_states, pos_emb):
        attention_output = self.multi_head_attention(hidden_states, pos_emb)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class LightTransformerEncoder(nn.Module):
    r"""One LightTransformerEncoder consists of several LightTransformerLayers.

    Args:
        n_layers(num): num of transformer layers in transformer encoder. Default: 2
        n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        hidden_size(num): the input and output hidden size. Default: 64
        inner_size(num): the dimensionality in feed-forward layer. Default: 256
        hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        hidden_act(str): activation function in feed-forward layer. Default: 'gelu'.
            candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12
    """

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        k_interests=5,
        hidden_size=64,
        seq_len=50,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
    ):
        super(LightTransformerEncoder, self).__init__()
        layer = LightTransformerLayer(
            n_heads,
            k_interests,
            hidden_size,
            seq_len,
            inner_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, pos_emb, output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TrandformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer layers' output,
            otherwise return a list only consists of the output of last transformer layer.
        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, pos_emb)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class ContextSeqEmbAbstractLayer(nn.Module):
    """For Deep Interest Network and feature-rich sequential recommender systems, return features embedding matrices."""

    def __init__(self):
        super(ContextSeqEmbAbstractLayer, self).__init__()
        self.token_field_offsets = {}
        self.float_field_offsets = {}
        self.token_embedding_table = nn.ModuleDict()
        self.float_embedding_table = nn.ModuleDict()
        self.token_seq_embedding_table = nn.ModuleDict()
        self.float_seq_embedding_table = nn.ModuleDict()

        self.token_field_names = None
        self.token_field_dims = None
        self.float_field_names = None
        self.float_field_dims = None
        self.token_seq_field_names = None
        self.token_seq_field_dims = None
        self.float_seq_field_names = None
        self.float_seq_field_dims = None
        self.num_feature_field = None

    def get_fields_name_dim(self):
        """get user feature field and item feature field."""
        self.token_field_names = {type: [] for type in self.types}
        self.token_field_dims = {type: [] for type in self.types}
        self.float_field_names = {type: [] for type in self.types}
        self.float_field_dims = {type: [] for type in self.types}
        self.token_seq_field_names = {type: [] for type in self.types}
        self.token_seq_field_dims = {type: [] for type in self.types}
        self.num_feature_field = {type: 0 for type in self.types}
        self.float_seq_field_names = {type: [] for type in self.types}
        self.float_seq_field_dims = {type: [] for type in self.types}

        for type in self.types:
            for field_name in self.field_names[type]:
                if self.dataset.field2type[field_name] == FeatureType.TOKEN:
                    self.token_field_names[type].append(field_name)
                    self.token_field_dims[type].append(self.dataset.num(field_name))
                elif self.dataset.field2type[field_name] == FeatureType.TOKEN_SEQ:
                    self.token_seq_field_names[type].append(field_name)
                    self.token_seq_field_dims[type].append(self.dataset.num(field_name))
                elif (
                    self.dataset.field2type[field_name] == FeatureType.FLOAT
                    and field_name in self.dataset.config["numerical_features"]
                ):
                    self.float_field_names[type].append(field_name)
                    self.float_field_dims[type].append(self.dataset.num(field_name))
                elif (
                    self.dataset.field2type[field_name] == FeatureType.FLOAT_SEQ
                    and field_name in self.dataset.config["numerical_features"]
                ):
                    self.float_seq_field_names[type].append(field_name)
                    self.float_seq_field_dims[type].append(self.dataset.num(field_name))
                else:
                    continue
                self.num_feature_field[type] += 1

    def get_embedding(self):
        """get embedding of all features."""
        for type in self.types:
            if len(self.token_field_dims[type]) > 0:
                self.token_field_offsets[type] = np.array(
                    (0, *np.cumsum(self.token_field_dims[type])[:-1]), dtype=np.long
                )
                self.token_embedding_table[type] = FMEmbedding(
                    self.token_field_dims[type],
                    self.token_field_offsets[type],
                    self.embedding_size,
                ).to(self.device)
            if len(self.float_field_dims[type]) > 0:
                self.float_field_offsets[type] = np.array(
                    (0, *np.cumsum(self.float_field_dims[type])[:-1]), dtype=np.long
                )
                self.float_embedding_table[type] = FLEmbedding(
                    self.float_field_dims[type],
                    self.float_field_offsets[type],
                    self.embedding_size,
                ).to(self.device)
            if len(self.token_seq_field_dims) > 0:
                self.token_seq_embedding_table[type] = nn.ModuleList()
                for token_seq_field_dim in self.token_seq_field_dims[type]:
                    self.token_seq_embedding_table[type].append(
                        nn.Embedding(token_seq_field_dim, self.embedding_size).to(
                            self.device
                        )
                    )
            if len(self.float_seq_field_dims) > 0:
                self.float_seq_embedding_table[type] = nn.ModuleList()
                for float_seq_field_dim in self.float_seq_field_dims[type]:
                    self.float_seq_embedding_table[type].append(
                        nn.Embedding(float_seq_field_dim, self.embedding_size).to(
                            self.device
                        )
                    )

    def embed_float_fields(self, float_fields, type, embed=True):
        """Get the embedding of float fields.
        In the following three functions("embed_float_fields" "embed_token_fields" "embed_token_seq_fields")
        when the type is user, [batch_size, max_item_length] should be recognised as [batch_size]

        Args:
            float_fields(torch.Tensor): [batch_size, max_item_length, num_float_field]
            type(str): user or item
            embed(bool): embed or not

        Returns:
            torch.Tensor: float fields embedding. [batch_size, max_item_length, num_float_field, embed_dim]

        """
        if float_fields is None:
            return None

        if type == "item":
            embedding_shape = float_fields.shape[:-1] + (-1,)
            float_fields = float_fields.reshape(
                -1, float_fields.shape[-2], float_fields.shape[-1]
            )
            float_embedding = self.float_embedding_table[type](float_fields)
            float_embedding = float_embedding.view(embedding_shape)
        else:
            float_embedding = self.float_embedding_table[type](float_fields)

        return float_embedding

    def embed_token_fields(self, token_fields, type):
        """Get the embedding of token fields

        Args:
            token_fields(torch.Tensor): input, [batch_size, max_item_length, num_token_field]
            type(str): user or item

        Returns:
            torch.Tensor: token fields embedding, [batch_size, max_item_length, num_token_field, embed_dim]

        """
        if token_fields is None:
            return None
        # [batch_size, max_item_length, num_token_field, embed_dim]
        if type == "item":
            embedding_shape = token_fields.shape + (-1,)
            token_fields = token_fields.reshape(-1, token_fields.shape[-1])
            token_embedding = self.token_embedding_table[type](token_fields)
            token_embedding = token_embedding.view(embedding_shape)
        else:
            token_embedding = self.token_embedding_table[type](token_fields)
        return token_embedding

    def embed_float_seq_fields(self, float_seq_fields, type):
        """Embed the float sequence feature columns

        Args:
            float_seq_fields (torch.FloatTensor): The input tensor. shape of [batch_size, seq_len, 2]
            mode (str): How to aggregate the embedding of feature in this field. default=mean

        Returns:
            torch.FloatTensor: The result embedding tensor of float sequence columns.
        """
        fields_result = []
        for i, float_seq_field in enumerate(float_seq_fields):
            embedding_table = self.float_seq_embedding_table[type][i]
            base, index = torch.split(float_seq_field, [1, 1], dim=-1)
            index = index.squeeze(-1)
            mask = index != 0
            mask = mask.float()
            value_cnt = torch.sum(mask, dim=-1, keepdim=True)
            float_seq_embedding = base * embedding_table(index.long())
            mask = mask.unsqueeze(-1).expand_as(float_seq_embedding)
            if self.pooling_mode == "max":
                masked_float_seq_embedding = float_seq_embedding - (1 - mask) * 1e9
                result = torch.max(masked_float_seq_embedding, dim=-2, keepdim=True)
                result = result.values
            elif self.pooling_mode == "sum":
                masked_float_seq_embedding = float_seq_embedding * mask.float()
                result = torch.sum(masked_float_seq_embedding, dim=-2, keepdim=True)
            else:
                masked_float_seq_embedding = float_seq_embedding * mask.float()
                result = torch.sum(masked_float_seq_embedding, dim=-2)
                eps = torch.FloatTensor([1e-8]).to(self.device)
                result = torch.div(result, value_cnt + eps)
                result = result.unsqueeze(-2)

            fields_result.append(result)
        if len(fields_result) == 0:
            return None
        else:
            return torch.cat(fields_result, dim=-2)

    def embed_token_seq_fields(self, token_seq_fields, type):
        """Get the embedding of token_seq fields.

        Args:
            token_seq_fields(torch.Tensor): input, [batch_size, max_item_length, seq_len]`
            type(str): user or item
            mode(str): mean/max/sum

        Returns:
            torch.Tensor: result [batch_size, max_item_length, num_token_seq_field, embed_dim]

        """
        fields_result = []
        for i, token_seq_field in enumerate(token_seq_fields):
            embedding_table = self.token_seq_embedding_table[type][i]
            mask = token_seq_field != 0  # [batch_size, max_item_length, seq_len]
            mask = mask.float()
            value_cnt = torch.sum(
                mask, dim=-1, keepdim=True
            )  # [batch_size, max_item_length, 1]
            token_seq_embedding = embedding_table(
                token_seq_field
            )  # [batch_size, max_item_length, seq_len, embed_dim]
            mask = mask.unsqueeze(-1).expand_as(token_seq_embedding)
            if self.pooling_mode == "max":
                masked_token_seq_embedding = token_seq_embedding - (1 - mask) * 1e9
                result = torch.max(
                    masked_token_seq_embedding, dim=-2, keepdim=True
                )  # [batch_size, max_item_length, 1, embed_dim]
                result = result.values
            elif self.pooling_mode == "sum":
                masked_token_seq_embedding = token_seq_embedding * mask.float()
                result = torch.sum(
                    masked_token_seq_embedding, dim=-2, keepdim=True
                )  # [batch_size, max_item_length, 1, embed_dim]
            else:
                masked_token_seq_embedding = token_seq_embedding * mask.float()
                result = torch.sum(
                    masked_token_seq_embedding, dim=-2
                )  # [batch_size, max_item_length, embed_dim]
                eps = torch.FloatTensor([1e-8]).to(self.device)
                result = torch.div(
                    result, value_cnt + eps
                )  # [batch_size, max_item_length, embed_dim]
                result = result.unsqueeze(
                    -2
                )  # [batch_size, max_item_length, 1, embed_dim]

            fields_result.append(result)
        if len(fields_result) == 0:
            return None
        else:
            return torch.cat(
                fields_result, dim=-2
            )  # [batch_size, max_item_length, num_token_seq_field, embed_dim]

    def embed_input_fields(self, user_idx, item_idx):
        """Get the embedding of user_idx and item_idx

        Args:
            user_idx(torch.Tensor): interaction['user_id']
            item_idx(torch.Tensor): interaction['item_id_list']

        Returns:
            dict: embedding of user feature and item feature

        """
        user_item_feat = {"user": self.user_feat, "item": self.item_feat}
        user_item_idx = {"user": user_idx, "item": item_idx}
        float_fields_embedding = {}
        float_seq_fields_embedding = {}
        token_fields_embedding = {}
        token_seq_fields_embedding = {}
        sparse_embedding = {}
        dense_embedding = {}

        for type in self.types:
            float_fields = []
            for field_name in self.float_field_names[type]:
                feature = user_item_feat[type][field_name][user_item_idx[type]]
                float_fields.append(
                    feature
                    if len(feature.shape) == (3 + (type == "item"))
                    else feature.unsqueeze(-2)
                )
            if len(float_fields) > 0:
                float_fields = torch.cat(
                    float_fields, dim=-1
                )  # [batch_size, max_item_length, num_float_field]
            else:
                float_fields = None
            float_fields_embedding[type] = self.embed_float_fields(float_fields, type)

            float_seq_fields = []
            for field_name in self.float_seq_field_names[type]:
                feature = user_item_feat[type][field_name][user_item_idx[type]]
                float_seq_fields.append(feature)
            # [batch_size, max_item_length, num_token_seq_field, embed_dim] or None
            float_seq_fields_embedding[type] = self.embed_float_seq_fields(
                float_seq_fields, type
            )

            if float_fields_embedding[type] is None:
                dense_embedding[type] = float_seq_fields_embedding[type]
            else:
                if float_seq_fields_embedding[type] is None:
                    dense_embedding[type] = float_fields_embedding[type]
                else:
                    dense_embedding[type] = torch.cat(
                        [
                            float_fields_embedding[type],
                            float_seq_fields_embedding[type],
                        ],
                        dim=-2,
                    )

            token_fields = []
            for field_name in self.token_field_names[type]:
                feature = user_item_feat[type][field_name][user_item_idx[type]]
                token_fields.append(feature.unsqueeze(-1))
            if len(token_fields) > 0:
                token_fields = torch.cat(
                    token_fields, dim=-1
                )  # [batch_size, max_item_length, num_token_field]
            else:
                token_fields = None
            # [batch_size, max_item_length, num_token_field, embed_dim] or None
            token_fields_embedding[type] = self.embed_token_fields(token_fields, type)

            token_seq_fields = []
            for field_name in self.token_seq_field_names[type]:
                feature = user_item_feat[type][field_name][user_item_idx[type]]
                token_seq_fields.append(feature)
            # [batch_size, max_item_length, num_token_seq_field, embed_dim] or None
            token_seq_fields_embedding[type] = self.embed_token_seq_fields(
                token_seq_fields, type
            )

            if token_fields_embedding[type] is None:
                sparse_embedding[type] = token_seq_fields_embedding[type]
            else:
                if token_seq_fields_embedding[type] is None:
                    sparse_embedding[type] = token_fields_embedding[type]
                else:
                    sparse_embedding[type] = torch.cat(
                        [
                            token_fields_embedding[type],
                            token_seq_fields_embedding[type],
                        ],
                        dim=-2,
                    )

        # sparse_embedding[type]
        # shape: [batch_size, max_item_length, num_token_seq_field+num_token_field, embed_dim] or None
        # dense_embedding[type]
        # shape: [batch_size, max_item_length, num_float_field]
        #     or [batch_size, max_item_length, num_float_field, embed_dim] or None
        return sparse_embedding, dense_embedding

    def forward(self, user_idx, item_idx):
        return self.embed_input_fields(user_idx, item_idx)


class ContextSeqEmbLayer(ContextSeqEmbAbstractLayer):
    """For Deep Interest Network, return all features (including user features and item features) embedding matrices."""

    def __init__(self, dataset, embedding_size, pooling_mode, device):
        super(ContextSeqEmbLayer, self).__init__()
        self.device = device
        self.embedding_size = embedding_size
        self.dataset = dataset
        self.user_feat = self.dataset.get_user_feature().to(self.device)
        self.item_feat = self.dataset.get_item_feature().to(self.device)

        self.field_names = {
            "user": list(self.user_feat.interaction.keys()),
            "item": list(self.item_feat.interaction.keys()),
        }

        self.types = ["user", "item"]
        self.pooling_mode = pooling_mode
        try:
            assert self.pooling_mode in ["mean", "max", "sum"]
        except AssertionError:
            raise AssertionError("Make sure 'pooling_mode' in ['mean', 'max', 'sum']!")
        self.get_fields_name_dim()
        self.get_embedding()


class FeatureSeqEmbLayer(ContextSeqEmbAbstractLayer):
    """For feature-rich sequential recommenders, return item features embedding matrices according to
    selected features."""

    def __init__(
        self, dataset, embedding_size, selected_features, pooling_mode, device
    ):
        super(FeatureSeqEmbLayer, self).__init__()

        self.device = device
        self.embedding_size = embedding_size
        self.dataset = dataset
        self.user_feat = None
        self.item_feat = self.dataset.get_item_feature().to(self.device)

        self.field_names = {"item": selected_features}

        self.types = ["item"]
        self.pooling_mode = pooling_mode
        try:
            assert self.pooling_mode in ["mean", "max", "sum"]
        except AssertionError:
            raise AssertionError("Make sure 'pooling_mode' in ['mean', 'max', 'sum']!")
        self.get_fields_name_dim()
        self.get_embedding()


class CNNLayers(nn.Module):
    r"""CNNLayers

    Args:
        - channels(list): a list contains the channels of each layer in cnn layers
        - kernel(list): a list contains the kernels of each layer in cnn layers
        - strides(list): a list contains the channels of each layer in cnn layers
        - activation(str): activation function after each layer in mlp layers. Default: 'relu'
                      candidates: 'sigmoid', 'tanh', 'relu', 'leekyrelu', 'none'

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where

        .. math::
            H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                      \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

        .. math::
            W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                      \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    Examples::

        >>> m = CNNLayers([1, 32, 32], [2,2], [2,2], 'relu')
        >>> input = torch.randn(128, 1, 64, 64)
        >>> output = m(input)
        >>> print(output.size())
        >>> torch.Size([128, 32, 16, 16])
    """

    def __init__(self, channels, kernels, strides, activation="relu", init_method=None):
        super(CNNLayers, self).__init__()
        self.channels = channels
        self.kernels = kernels
        self.strides = strides
        self.activation = activation
        self.init_method = init_method
        self.num_of_nets = len(self.channels) - 1

        if len(kernels) != len(strides) or self.num_of_nets != (len(kernels)):
            raise RuntimeError("channels, kernels and strides don't match\n")

        cnn_modules = []

        for i in range(self.num_of_nets):
            cnn_modules.append(
                nn.Conv2d(
                    self.channels[i],
                    self.channels[i + 1],
                    self.kernels[i],
                    stride=self.strides[i],
                )
            )
            if self.activation.lower() == "sigmoid":
                cnn_modules.append(nn.Sigmoid())
            elif self.activation.lower() == "tanh":
                cnn_modules.append(nn.Tanh())
            elif self.activation.lower() == "relu":
                cnn_modules.append(nn.ReLU())
            elif self.activation.lower() == "leakyrelu":
                cnn_modules.append(nn.LeakyReLU())
            elif self.activation.lower() == "none":
                pass

        self.cnn_layers = nn.Sequential(*cnn_modules)

        if self.init_method is not None:
            self.apply(self.init_weights)

    def init_weights(self, module):
        # We just initialize the module with normal distribution as the paper said
        if isinstance(module, nn.Conv2d):
            if self.init_method == "norm":
                normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        return self.cnn_layers(input_feature)


class FMFirstOrderLinear(nn.Module):
    """Calculate the first order score of the input features.
    This class is a member of ContextRecommender, you can call it easily when inherit ContextRecommender.

    """

    def __init__(self, config, dataset, output_dim=1):
        super(FMFirstOrderLinear, self).__init__()
        self.field_names = dataset.fields(
            source=[
                FeatureSource.INTERACTION,
                FeatureSource.USER,
                FeatureSource.USER_ID,
                FeatureSource.ITEM,
                FeatureSource.ITEM_ID,
            ]
        )
        self.LABEL = config["LABEL_FIELD"]
        self.device = config["device"]
        self.numerical_features = config["numerical_features"]
        self.token_field_names = []
        self.token_field_dims = []
        self.float_field_names = []
        self.float_field_dims = []
        self.token_seq_field_names = []
        self.token_seq_field_dims = []
        self.float_seq_field_names = []
        self.float_seq_field_dims = []

        for field_name in self.field_names:
            if field_name == self.LABEL:
                continue
            if dataset.field2type[field_name] == FeatureType.TOKEN:
                self.token_field_names.append(field_name)
                self.token_field_dims.append(dataset.num(field_name))
            elif dataset.field2type[field_name] == FeatureType.TOKEN_SEQ:
                self.token_seq_field_names.append(field_name)
                self.token_seq_field_dims.append(dataset.num(field_name))
            elif (
                dataset.field2type[field_name] == FeatureType.FLOAT
                and field_name in self.numerical_features
            ):
                self.float_field_names.append(field_name)
                self.float_field_dims.append(dataset.num(field_name))
            elif (
                dataset.field2type[field_name] == FeatureType.FLOAT_SEQ
                and field_name in self.numerical_features
            ):
                self.float_seq_field_names.append(field_name)
                self.float_seq_field_dims.append(dataset.num(field_name))

        if len(self.token_field_dims) > 0:
            self.token_field_offsets = np.array(
                (0, *np.cumsum(self.token_field_dims)[:-1]), dtype=np.long
            )
            self.token_embedding_table = FMEmbedding(
                self.token_field_dims, self.token_field_offsets, output_dim
            )
        if len(self.float_field_dims) > 0:
            self.float_field_offsets = np.array(
                (0, *np.cumsum(self.float_field_dims)[:-1]), dtype=np.long
            )
            self.float_embedding_table = FLEmbedding(
                self.float_field_dims, self.float_field_offsets, output_dim
            )
        if len(self.token_seq_field_dims) > 0:
            self.token_seq_embedding_table = nn.ModuleList()
            for token_seq_field_dim in self.token_seq_field_dims:
                self.token_seq_embedding_table.append(
                    nn.Embedding(token_seq_field_dim, output_dim)
                )
        if len(self.float_seq_field_dims) > 0:
            self.float_seq_embedding_table = nn.ModuleList()
            for float_seq_field_dim in self.float_seq_field_dims:
                self.float_seq_embedding_table.append(
                    nn.Embedding(float_seq_field_dim, output_dim)
                )

        self.bias = nn.Parameter(torch.zeros((output_dim,)), requires_grad=True)

    def embed_float_fields(self, float_fields):
        """Embed the float feature columns

        Args:
            float_fields (torch.FloatTensor): The input dense tensor. shape of [batch_size, num_float_field, 2]
            embed (bool): Return the embedding of columns or just the columns itself. Defaults to ``True``.

        Returns:
            torch.FloatTensor: The result embedding tensor of float columns.
        """
        # input Tensor shape : [batch_size, num_float_field]
        if float_fields is None:
            return None
        # [batch_size, num_float_field, embed_dim]
        float_embedding = self.float_embedding_table(float_fields)

        # [batch_size, 1, output_dim]
        float_embedding = torch.sum(float_embedding, dim=1, keepdim=True)
        return float_embedding

    def embed_float_seq_fields(self, float_seq_fields, mode="mean"):
        """Embed the float sequence feature columns

        Args:
            float_seq_fields (torch.LongTensor): The input tensor. shape of [batch_size, seq_len, 2]
            mode (str): How to aggregate the embedding of feature in this field. default=mean

        Returns:
            torch.FloatTensor: The result embedding tensor of float sequence columns.
        """
        # input is a list of Tensor shape of [batch_size, seq_len]
        fields_result = []
        for i, float_seq_field in enumerate(float_seq_fields):
            embedding_table = self.float_seq_embedding_table[i]
            base, index = torch.split(float_seq_field, [1, 1], dim=-1)
            index = index.squeeze(-1)
            mask = index != 0  # [batch_size, seq_len]
            mask = mask.float()
            value_cnt = torch.sum(mask, dim=1, keepdim=True)  # [batch_size, 1]

            float_seq_embedding = base * embedding_table(
                index.long()
            )  # [batch_size, seq_len, embed_dim]

            mask = mask.unsqueeze(2).expand_as(
                float_seq_embedding
            )  # [batch_size, seq_len, embed_dim]
            if mode == "max":
                masked_float_seq_embedding = (
                    float_seq_embedding - (1 - mask) * 1e9
                )  # [batch_size, seq_len, embed_dim]
                result = torch.max(
                    masked_float_seq_embedding, dim=1, keepdim=True
                )  # [batch_size, 1, embed_dim]
            elif mode == "sum":
                masked_float_seq_embedding = float_seq_embedding * mask.float()
                result = torch.sum(
                    masked_float_seq_embedding, dim=1, keepdim=True
                )  # [batch_size, 1, embed_dim]
            else:
                masked_float_seq_embedding = float_seq_embedding * mask.float()
                result = torch.sum(
                    masked_float_seq_embedding, dim=1
                )  # [batch_size, embed_dim]
                eps = torch.FloatTensor([1e-8]).to(self.device)
                result = torch.div(result, value_cnt + eps)  # [batch_size, embed_dim]
                result = result.unsqueeze(1)  # [batch_size, 1, embed_dim]
            fields_result.append(result)
        if len(fields_result) == 0:
            return None
        else:
            return torch.sum(
                torch.cat(fields_result, dim=1), dim=1, keepdim=True
            )  # [batch_size, num_token_seq_field, embed_dim]

    def embed_token_fields(self, token_fields):
        """Calculate the first order score of token feature columns

        Args:
            token_fields (torch.LongTensor): The input tensor. shape of [batch_size, num_token_field]

        Returns:
            torch.FloatTensor: The first order score of token feature columns
        """
        # input Tensor shape : [batch_size, num_token_field]
        if token_fields is None:
            return None
        # [batch_size, num_token_field, embed_dim]
        token_embedding = self.token_embedding_table(token_fields)
        # [batch_size, 1, output_dim]
        token_embedding = torch.sum(token_embedding, dim=1, keepdim=True)

        return token_embedding

    def embed_token_seq_fields(self, token_seq_fields):
        """Calculate the first order score of token sequence feature columns

        Args:
            token_seq_fields (torch.LongTensor): The input tensor. shape of [batch_size, seq_len]

        Returns:
            torch.FloatTensor: The first order score of token sequence feature columns
        """
        # input is a list of Tensor shape of [batch_size, seq_len]
        fields_result = []
        for i, token_seq_field in enumerate(token_seq_fields):
            embedding_table = self.token_seq_embedding_table[i]
            mask = token_seq_field != 0  # [batch_size, seq_len]
            mask = mask.float()
            value_cnt = torch.sum(mask, dim=1, keepdim=True)  # [batch_size, 1]

            token_seq_embedding = embedding_table(
                token_seq_field
            )  # [batch_size, seq_len, output_dim]

            mask = mask.unsqueeze(2).expand_as(
                token_seq_embedding
            )  # [batch_size, seq_len, output_dim]
            masked_token_seq_embedding = token_seq_embedding * mask.float()
            result = torch.sum(
                masked_token_seq_embedding, dim=1, keepdim=True
            )  # [batch_size, 1, output_dim]

            fields_result.append(result)
        if len(fields_result) == 0:
            return None
        else:
            return torch.sum(
                torch.cat(fields_result, dim=1), dim=1, keepdim=True
            )  # [batch_size, 1, output_dim]

    def forward(self, interaction):
        total_fields_embedding = []
        float_fields = []
        for field_name in self.float_field_names:
            if len(interaction[field_name].shape) == 3:
                float_fields.append(interaction[field_name])
            else:
                float_fields.append(interaction[field_name].unsqueeze(1))

        if len(float_fields) > 0:
            float_fields = torch.cat(float_fields, dim=1)
        else:
            float_fields = None

        float_fields_embedding = self.embed_float_fields(float_fields)

        if float_fields_embedding is not None:
            total_fields_embedding.append(float_fields_embedding)

        float_seq_fields = []
        for field_name in self.float_seq_field_names:
            float_seq_fields.append(interaction[field_name])

        float_seq_fields_embedding = self.embed_float_seq_fields(float_seq_fields)

        if float_seq_fields_embedding is not None:
            total_fields_embedding.append(float_seq_fields_embedding)

        token_fields = []
        for field_name in self.token_field_names:
            token_fields.append(interaction[field_name].unsqueeze(1))
        if len(token_fields) > 0:
            token_fields = torch.cat(
                token_fields, dim=1
            )  # [batch_size, num_token_field]
        else:
            token_fields = None
        # [batch_size, 1, output_dim] or None
        token_fields_embedding = self.embed_token_fields(token_fields)
        if token_fields_embedding is not None:
            total_fields_embedding.append(token_fields_embedding)

        token_seq_fields = []
        for field_name in self.token_seq_field_names:
            token_seq_fields.append(interaction[field_name])
        # [batch_size, 1, output_dim] or None
        token_seq_fields_embedding = self.embed_token_seq_fields(token_seq_fields)
        if token_seq_fields_embedding is not None:
            total_fields_embedding.append(token_seq_fields_embedding)

        return (
            torch.sum(torch.cat(total_fields_embedding, dim=1), dim=1) + self.bias
        )  # [batch_size, output_dim]


class SparseDropout(nn.Module):
    """
    This is a Module that execute Dropout on Pytorch sparse tensor.
    """

    def __init__(self, p=0.5):
        super(SparseDropout, self).__init__()
        # p is ratio of dropout
        # convert to keep probability
        self.kprob = 1 - p

    def forward(self, x):
        if not self.training:
            return x

        mask = ((torch.rand(x._values().size()) + self.kprob).floor()).type(torch.bool)
        rc = x._indices()[:, mask]
        val = x._values()[mask] * (1.0 / self.kprob)
        return torch.sparse.FloatTensor(rc, val, x.shape)


# ============================================================
# Hyper-Connection Layers
# Reference: "Hyper-Connections" (Zhu et al., 2024, arXiv:2409.19606)
# Algorithm 2 PyTorch implementation
# ============================================================


class _PureMultiHeadAttention(nn.Module):
    """Multi-head self-attention WITHOUT internal residual connection.

    This is a "pure" version intended to be used inside HyperConnection,
    where all residual management is delegated to the HC depth-connection.
    Uses Pre-Norm (LayerNorm applied to input before projection).

    Args:
        n_heads (int): Number of attention heads.
        hidden_size (int): Model hidden dimension.
        hidden_dropout_prob (float): Dropout on attention output.
        attn_dropout_prob (float): Dropout on attention weights.
        layer_norm_eps (float): Epsilon for LayerNorm.
        expansion_rate (int): HC expansion rate n; output projection is
            scaled by 1/sqrt(n) to keep output variance consistent.
    """

    def __init__(
        self,
        n_heads: int,
        hidden_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        layer_norm_eps: float,
        expansion_rate: int = 1,
    ) -> None:
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by n_heads ({n_heads})"
            )
        self.num_attention_heads = n_heads
        self.attention_head_size = hidden_size // n_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dense = nn.Linear(hidden_size, hidden_size)
        # Scale output projection by 1/sqrt(n) as per HC paper (Sec. 4 implementation note)
        with torch.no_grad():
            self.dense.weight.data.div_(math.sqrt(expansion_rate))

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

        # Pre-Norm: applied to the input before QKV projection
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, L, D] -> [B, heads, L, head_dim]"""
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, h0: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h0: [B, L, D]  - width-connected input from HyperConnection
            attention_mask: [B, 1, 1, L] or [B, 1, L, L]

        Returns:
            out: [B, L, D]  - pure attention output (NO residual added)
        """
        # Pre-Norm
        x = self.layer_norm(h0)

        q = self._split_heads(self.query(x))   # [B, heads, L, head_dim]
        k = self._split_heads(self.key(x))     # [B, heads, L, head_dim]
        v = self._split_heads(self.value(x))   # [B, heads, L, head_dim]

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-1, -2)) / self.sqrt_head_size  # [B, heads, L, L]
        scores = scores + attention_mask
        probs = self.attn_dropout(self.softmax(scores))

        ctx = torch.matmul(probs, v)                       # [B, heads, L, head_dim]
        ctx = ctx.permute(0, 2, 1, 3).contiguous()
        ctx = ctx.view(ctx.size(0), ctx.size(1), self.all_head_size)

        out = self.out_dropout(self.dense(ctx))            # [B, L, D]
        return out                                         # No residual, no outer LayerNorm


class _PureFeedForward(nn.Module):
    """Point-wise FFN WITHOUT internal residual connection.

    Uses Pre-Norm (LayerNorm applied to input before dense layers).
    Output linear is scaled by 1/sqrt(n) as per HC paper.

    Args:
        hidden_size (int): Input/output dimension.
        inner_size (int): Intermediate dimension.
        hidden_dropout_prob (float): Dropout probability.
        hidden_act (str): Activation function name.
        layer_norm_eps (float): Epsilon for LayerNorm.
        expansion_rate (int): HC expansion rate n.
    """

    def __init__(
        self,
        hidden_size: int,
        inner_size: int,
        hidden_dropout_prob: float,
        hidden_act: str,
        layer_norm_eps: float,
        expansion_rate: int = 1,
    ) -> None:
        super().__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.dense_2 = nn.Linear(inner_size, hidden_size)
        # Scale output linear by 1/sqrt(n)
        with torch.no_grad():
            self.dense_2.weight.data.div_(math.sqrt(expansion_rate))

        self.intermediate_act_fn = self._get_act(hidden_act)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        # Pre-Norm
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    @staticmethod
    def _get_act(act: str):
        """Return activation function by name."""

        def _gelu(x):
            return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

        act_map = {
            "gelu": _gelu,
            "relu": fn.relu,
            "swish": lambda x: x * torch.sigmoid(x),
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        if act not in act_map:
            raise ValueError(f"Unsupported activation: {act}. Choose from {list(act_map)}")
        return act_map[act]

    def forward(self, h0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h0: [B, L, D]

        Returns:
            out: [B, L, D]  - pure FFN output (NO residual added)
        """
        # Pre-Norm
        x = self.layer_norm(h0)
        x = self.dense_1(x)
        x = self.intermediate_act_fn(x)
        x = self.dropout(self.dense_2(x))
        return x                            # No residual, no outer LayerNorm


class HyperConnection(nn.Module):
    r"""Hyper-Connection module replacing residual connections.

    Implements the full "width-connection + layer computation + depth-connection"
    cycle described in Algorithm 1 & 2 of the HC paper (Zhu et al., 2024).

    Each HyperConnection manages ONE transformer sub-block T composed of
    a Multi-Head Attention layer followed by a Feed-Forward Network.

    Data flow (Algorithm 1, adapted for Attn+FFN as one T):
    ::

        ┌─ Width-Connection ──────────────────────────────────────┐
        │  H [B,L,N,D]  →  alpha^T @ H  →  mix_h [B,L,N+1,D]    │
        │    mix_h[:,0,:] = h0  (layer input)                      │
        │    mix_h[:,1:,:]= H'  (residual stream)                  │
        └──────────────────────────────────────────────────────────┘
                │
                v h0 [B,L,D]
        ┌─ Layer T ────────────────────────────────────────────────┐
        │  h_attn = PureAttn(h0) + h0     (attn pass)             │
        │  h_o    = PureFFN(h_attn)       (ffn pass, pure output)  │
        └──────────────────────────────────────────────────────────┘
                │
                v h_o [B,L,D]
        ┌─ Depth-Connection ───────────────────────────────────────┐
        │  H_next = einsum(beta, h_o) + H'                         │
        │         = [B,L,N,D]                                       │
        └──────────────────────────────────────────────────────────┘

    Note on internal attn residual:
        Inside T, Attention uses h0 as its own residual (h_attn = Pure_Attn(h0) + h0)
        to maintain stable gradients within T. The FFN sub-block is purely
        applied on h_attn with no further residual -- all outer residual management
        is done by the depth-connection.

    Args:
        hidden_size (int): Model hidden dimension D.
        expansion_rate (int): Number of hyper-hidden vectors N (n in paper).
        layer_id (int): Zero-based layer index k, used for static init Eq.14.
        n_heads (int): Attention heads.
        inner_size (int): FFN intermediate size.
        hidden_dropout_prob (float): Dropout on attn/ffn output.
        attn_dropout_prob (float): Dropout on attention weights.
        hidden_act (str): FFN activation.
        layer_norm_eps (float): LayerNorm epsilon.
        dynamic (bool): Enable dynamic hyper-connections (DHC). Default: True.
        use_tanh (bool): Apply tanh on dynamic delta. Default: True.
    """

    def __init__(
        self,
        hidden_size: int,
        expansion_rate: int,
        layer_id: int,
        n_heads: int,
        inner_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        hidden_act: str,
        layer_norm_eps: float,
        dynamic: bool = True,
        use_tanh: bool = True,
    ) -> None:
        super().__init__()
        self.expansion_rate = expansion_rate
        self.dynamic = dynamic
        self.use_tanh = use_tanh
        n = expansion_rate

        # ── Static parameters (Eq. 14) ────────────────────────────────────
        # static_beta β: [N]  – depth-connection output weights, init to 1
        self.static_beta = nn.Parameter(torch.ones(n))

        # static_alpha [N, N+1] = concat(am [N,1], ar [N,N])
        #   am: one-hot at (layer_id % n), row → selects which hidden feeds T
        #   ar: identity → residual stream passes unchanged
        init_am = torch.zeros(n, 1)
        init_am[layer_id % n, 0] = 1.0
        init_ar = torch.eye(n)
        self.static_alpha = nn.Parameter(torch.cat([init_am, init_ar], dim=1))  # [N, N+1]

        # ── Dynamic parameters (Eqs. 11-13), init to 0 ────────────────────
        if self.dynamic:
            # alpha_fn projects each hidden vector: [D] -> [N+1] weights
            self.dynamic_alpha_fn = nn.Parameter(torch.zeros(hidden_size, n + 1))
            self.dynamic_alpha_scale = nn.Parameter(torch.ones(1) * 0.01)
            # beta_fn: single vector [D], dot-product with each hidden → scalar per hidden
            self.dynamic_beta_fn = nn.Parameter(torch.zeros(hidden_size))
            self.dynamic_beta_scale = nn.Parameter(torch.ones(1) * 0.01)
            # LayerNorm for computing dynamic weights (separate from Pre-Norm in T)
            self.dynamic_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        # ── Sub-layers (T = PureAttn + PureFFN) ───────────────────────────
        self.pure_attn = _PureMultiHeadAttention(
            n_heads=n_heads,
            hidden_size=hidden_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attn_dropout_prob=attn_dropout_prob,
            layer_norm_eps=layer_norm_eps,
            expansion_rate=expansion_rate,
        )
        self.pure_ffn = _PureFeedForward(
            hidden_size=hidden_size,
            inner_size=inner_size,
            hidden_dropout_prob=hidden_dropout_prob,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            expansion_rate=expansion_rate,
        )

    # ------------------------------------------------------------------
    # Width-Connection  (Algorithm 2: width_connection)
    # ------------------------------------------------------------------
    def _width_connection(self, h: torch.Tensor):
        """Compute alpha/beta and perform width connection.

        Args:
            h: [B, L, N, D]

        Returns:
            mix_h: [B, L, N+1, D]  – stacked [h0, H'], where
                   mix_h[:,:,0,:] = h0 (input to T)
                   mix_h[:,:,1:,:] = H' (residual stream)
            beta:  [B, L, N]       – depth-connection output weights
        """
        # Static baseline
        alpha = self.static_alpha.unsqueeze(0).unsqueeze(0)  # [1,1,N,N+1]
        beta = self.static_beta.unsqueeze(0).unsqueeze(0)    # [1,1,N]

        if self.dynamic:
            # norm_h: [B, L, N, D]
            norm_h = self.dynamic_norm(h)

            # Dynamic alpha: [B,L,N,D] @ [D,N+1] -> [B,L,N,N+1]
            d_alpha = norm_h @ self.dynamic_alpha_fn
            if self.use_tanh:
                d_alpha = torch.tanh(d_alpha)
            alpha = alpha + d_alpha * self.dynamic_alpha_scale  # broadcast [B,L,N,N+1]

            # Dynamic beta: [B,L,N,D] @ [D] -> [B,L,N]
            d_beta = norm_h @ self.dynamic_beta_fn
            if self.use_tanh:
                d_beta = torch.tanh(d_beta)
            beta = beta + d_beta * self.dynamic_beta_scale      # broadcast [B,L,N]

        # mix_h = alpha^T @ h: [B,L,N+1,N] @ [B,L,N,D] = [B,L,N+1,D]
        mix_h = alpha.transpose(-1, -2) @ h
        return mix_h, beta

    # ------------------------------------------------------------------
    # Depth-Connection  (Algorithm 2: depth_connection)
    # ------------------------------------------------------------------
    @staticmethod
    def _depth_connection(
        mix_h: torch.Tensor, h_o: torch.Tensor, beta: torch.Tensor
    ) -> torch.Tensor:
        """Combine layer output with residual stream.

        Args:
            mix_h: [B, L, N+1, D]
            h_o:   [B, L, D]   – output of sub-layer T
            beta:  [B, L, N]   – depth-connection weights

        Returns:
            H_next: [B, L, N, D]
        """
        # H' = mix_h[..., 1:, :] is the residual stream [B,L,N,D]
        # depth term: beta_i * h_o  for each of the N hidden vectors
        depth_term = torch.einsum("bld,bln->blnd", h_o, beta)  # [B,L,N,D]
        return depth_term + mix_h[..., 1:, :]                  # [B,L,N,D]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, H: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """One hyper-connection forward pass (= one transformer block).

        Args:
            H:              [B, L, N, D]  hyper-hidden matrix
            attention_mask: [B, 1, 1, L]  causal mask (additive, -inf style)

        Returns:
            H_next: [B, L, N, D]  updated hyper-hidden matrix
        """
        # Step 1 – Width Connection
        mix_h, beta = self._width_connection(H)   # mix_h:[B,L,N+1,D], beta:[B,L,N]

        # Step 2 – Extract h0, compute T(h0)
        h0 = mix_h[:, :, 0, :]                    # [B, L, D]

        # Attention sub-block: Pure attention + internal attention residual
        # NOTE: we keep h0 → h_attn residual inside T for stable gradient flow.
        h_attn = self.pure_attn(h0, attention_mask) + h0   # [B, L, D]

        # FFN sub-block: pure, no residual (depth-conn manages this)
        h_o = self.pure_ffn(h_attn)                        # [B, L, D]

        # Step 3 – Depth Connection
        H_next = self._depth_connection(mix_h, h_o, beta)  # [B, L, N, D]
        return H_next


class HyperTransformerEncoder(nn.Module):
    r"""Stack of HyperConnection transformer blocks.

    Args:
        n_layers (int): Number of transformer blocks (each = Attn + FFN).
        n_heads (int): Attention heads.
        hidden_size (int): Model dimension D.
        inner_size (int): FFN intermediate dimension.
        hidden_dropout_prob (float): Dropout probability.
        attn_dropout_prob (float): Attention dropout probability.
        hidden_act (str): FFN activation.
        layer_norm_eps (float): LayerNorm epsilon.
        expansion_rate (int): HC expansion rate N.
        dynamic_hc (bool): Use dynamic hyper-connections.
        hc_use_tanh (bool): Use tanh in dynamic delta.
    """

    def __init__(
        self,
        n_layers: int = 2,
        n_heads: int = 2,
        hidden_size: int = 64,
        inner_size: int = 256,
        hidden_dropout_prob: float = 0.5,
        attn_dropout_prob: float = 0.5,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-12,
        expansion_rate: int = 4,
        dynamic_hc: bool = True,
        hc_use_tanh: bool = True,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                HyperConnection(
                    hidden_size=hidden_size,
                    expansion_rate=expansion_rate,
                    layer_id=i,
                    n_heads=n_heads,
                    inner_size=inner_size,
                    hidden_dropout_prob=hidden_dropout_prob,
                    attn_dropout_prob=attn_dropout_prob,
                    hidden_act=hidden_act,
                    layer_norm_eps=layer_norm_eps,
                    dynamic=dynamic_hc,
                    use_tanh=hc_use_tanh,
                )
                for i in range(n_layers)
            ]
        )

    def forward(
        self,
        H: torch.Tensor,
        attention_mask: torch.Tensor,
        output_all_encoded_layers: bool = True,
    ):
        """
        Args:
            H: [B, L, N, D]  initial hyper-hidden matrix
            attention_mask: [B, 1, 1, L]
            output_all_encoded_layers: if True return list of all H, else list of last H only.

        Returns:
            List[torch.Tensor]:  each element [B, L, N, D]
        """
        all_layers = []
        for layer in self.layers:
            H = layer(H, attention_mask)
            if output_all_encoded_layers:
                all_layers.append(H)
        if not output_all_encoded_layers:
            all_layers.append(H)
        return all_layers


# ============================================================
# Exact Special Orthogonal Hyper-Connections (SOHC)
# Based on Cayley Transform & Newton-Schulz Iteration
# ============================================================

class CayleyOrthogonalEngine(nn.Module):
    r"""Cayley Orthogonal Transform Engine via Newton-Schulz Iteration.
    
    This module enforces strict special orthogonal manifold constraints (SO(n)) 
    on the residual mapping matrices, preventing representation collapse 
    and multiplicative gradient decay in extremely deep networks.
    
    Data Flow:
    ::
    
        +------------------------------------------------------------------+
        | SOHC: Cayley Transform + Newton-Schulz Iteration Engine          |
        +------------------------------------------------------------------+
        | 1. A = 0.5 * (W - W.T), where W ~ N(0, σ)                        |
        | 2. Z = I + A,  X_0 = I - A                                       |
        | 3. For k in 1..7: X_{k+1} = X_k @ (2I - Z @ X_k)                 |
        | 4. H_res = (I - A) @ X_7       ===> Strict SO(n) Isometric Space |
        +------------------------------------------------------------------+
    """
    def __init__(self, hidden_size: int, ns_iterations: int = 7, init_std: float = 0.02):
        super().__init__()
        self.hidden_size = hidden_size
        self.ns_iterations = ns_iterations
        # Unconstrained learnable weight W ~ N(0, init_std)
        self.weight = nn.Parameter(torch.empty(hidden_size, hidden_size))
        nn.init.normal_(self.weight, mean=0.0, std=init_std)

    def forward(self) -> torch.Tensor:
        """
        Returns:
            H_res (torch.Tensor): [D, D] exactly special orthogonal matrix.
        """
        device = self.weight.device
        dtype = self.weight.dtype
        I = torch.eye(self.hidden_size, device=device, dtype=dtype)
        
        # 1. Skew-Symmetric Construction: A^T = -A
        A = 0.5 * (self.weight - self.weight.t())
        
        # 2. Setup for Inv approximation: Z = I + A, X_0 = I - A
        Z = I + A
        X = I - A 
        
        # 3. Newton-Schulz Iterations
        for _ in range(self.ns_iterations):
            # X_{k+1} = X_k(2I - Z X_k)
            inner = 2.0 * I - torch.mm(Z, X)
            X = torch.mm(X, inner)
            
        # 4. Cayley Transform: H_res = (I - A)\((I + A)^-1\)
        H_res = torch.mm(I - A, X)
        return H_res


class DynamicSigmoidRouting(nn.Module):
    r"""Dynamic Sigmoid Routing for feature extraction and fusion.
    
    Provides independent, element-wise dynamic scaling based on context.
    Unlike softmax which creates a zero-sum game, sigmoid allows the model
    to independently suppress or amplify each feature channel based on
    the structural requirements of the deep network.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        # Linear layer for evaluating mapping importance
        self.router = nn.Linear(hidden_size, hidden_size)
        # Initialize slightly negatively or around 0 for stable start
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.router.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [B, L, D]
        Returns:
            gate (torch.Tensor): [B, L, D] semantic gates (0.0 ~ 1.0)
        """
        return torch.sigmoid(self.router(x))


class _SOHCPureMultiHeadAttention(nn.Module):
    """Multi-head self-attention WITHOUT internal residual connection.
    Uses Pre-Norm (LayerNorm applied to input before projection).
    """
    def __init__(
        self,
        n_heads: int,
        hidden_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by n_heads ({n_heads})"
            )
        self.num_attention_heads = n_heads
        self.attention_head_size = hidden_size // n_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dense = nn.Linear(hidden_size, hidden_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

        # Pre-Norm
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, L, D] -> [B, heads, L, head_dim]"""
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, h0: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Pre-Norm
        x = self.layer_norm(h0)

        q = self._split_heads(self.query(x))   # [B, heads, L, head_dim]
        k = self._split_heads(self.key(x))     # [B, heads, L, head_dim]
        v = self._split_heads(self.value(x))   # [B, heads, L, head_dim]

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-1, -2)) / self.sqrt_head_size  # [B, heads, L, L]
        scores = scores + attention_mask
        probs = self.attn_dropout(self.softmax(scores))

        ctx = torch.matmul(probs, v)                       # [B, heads, L, head_dim]
        ctx = ctx.permute(0, 2, 1, 3).contiguous()
        ctx = ctx.view(ctx.size(0), ctx.size(1), self.all_head_size)

        out = self.out_dropout(self.dense(ctx))            # [B, L, D]
        return out                                         # No residual


class _SOHCPureFeedForward(nn.Module):
    """Point-wise FFN WITHOUT internal residual connection.
    Uses Pre-Norm (LayerNorm applied to input before dense layers).
    """

    def __init__(
        self,
        hidden_size: int,
        inner_size: int,
        hidden_dropout_prob: float,
        hidden_act: str,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.dense_2 = nn.Linear(inner_size, hidden_size)

        self.intermediate_act_fn = self._get_act(hidden_act)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        # Pre-Norm
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    @staticmethod
    def _get_act(act: str):
        """Return activation function by name."""

        def _gelu(x):
            return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

        act_map = {
            "gelu": _gelu,
            "relu": fn.relu if 'fn' in globals() else torch.nn.functional.relu,
            "swish": lambda x: x * torch.sigmoid(x),
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        if act not in act_map:
            raise ValueError(f"Unsupported activation: {act}. Choose from {list(act_map)}")
        return act_map[act]

    def forward(self, h0: torch.Tensor) -> torch.Tensor:
        # Pre-Norm
        x = self.layer_norm(h0)
        x = self.dense_1(x)
        x = self.intermediate_act_fn(x)
        x = self.dropout(self.dense_2(x))
        return x                            # No residual



class SOHCEncoderLayer(nn.Module):
    r"""A single Wide-SOHC Transformer Block (N hyper-hidden channels).

    Data flow (matches the development spec document):
    ::

        Input H: [B, L, N, D]  -- N hyper-hidden vectors

        +--------------------------------------------------------------+
        | 1. Main Stream: Special Orthogonal (SO(N)) Channel Mixing    |
        |    H_res = CayleyNS(W)         # [N, N] SO(N) matrix        |
        |    iso = einsum(H_res, H)      # [B, L, N, D]               |
        +--------------------------------------------------------------+
        |                                                              |
        | 2. Dynamic Semantic Routing Stream:                          |
        |    Gate_pre  = Sigmoid(Linear(H_mean))   # [B, L, 1, N]    |
        |    h0 = (Gate_pre @ H).squeeze(-2)       # [B, L, D]       |
        |    h_attn = PureAttn(h0) + h0            # [B, L, D]       |
        |    h_ffn  = PureFFN(h_attn)              # [B, L, D]       |
        |    Gate_post = Sigmoid(Linear(H_mean))   # [B, L, N, 1]   |
        |    sem = Gate_post * h_ffn.unsqueeze(-2) # [B, L, N, D]   |
        +--------------------------------------------------------------+
        |                                                              |
        | 3. Synthesis (Parallel Stream Addition):                     |
        |    H_next = iso + sem          # [B, L, N, D]              |
        +--------------------------------------------------------------+
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
        ns_iterations: int = 7,
        init_std: float = 0.02,
        no_ortho: bool = False,
        static_routing: bool = False,
        use_standard_block: bool = False,
        use_softmax_routing: bool = False,
    ):
        super().__init__()
        self.expansion_rate = expansion_rate  # N
        self.no_ortho = no_ortho
        self.static_routing = static_routing
        self.use_standard_block = use_standard_block
        self.use_softmax_routing = use_softmax_routing
        n = expansion_rate

        if self.use_standard_block:
            self.ln1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
            self.ln2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        # ── 1. Special Orthogonal Channel Engine ──────────────────────────
        # Generates an N×N SO(N) matrix via Cayley transform.
        # This mixes information ACROSS the N hyper-hidden channels
        # while keeping the channel-space isometric (no signal loss).
        self.cayley_w = nn.Parameter(
            torch.zeros(n, n).normal_(0.0, init_std)
        )
        self.ns_iterations = ns_iterations

        # ── 2. Dynamic Width-Connection Routing ───────────────────────────
        # gate_pre  : [D] → [N] scalar weights (selects which channels
        #             contribute to sub-layer input h0)
        # gate_post : [D] → [N] scalar weights (controls how much of the
        #             sub-layer output is written back to each channel)
        self.gate_pre  = nn.Linear(hidden_size, n, bias=True)
        self.gate_post = nn.Linear(hidden_size, n, bias=True)
        # Initialise so that each gate starts uniform (1/N)
        nn.init.zeros_(self.gate_pre.weight)
        nn.init.constant_(self.gate_pre.bias, -math.log(n - 1) if n > 1 else 0.0)
        nn.init.zeros_(self.gate_post.weight)
        nn.init.zeros_(self.gate_post.bias)

        # ── 3. Pure Sub-Layers (no internal residuals) ────────────────────
        self.pure_attn = _SOHCPureMultiHeadAttention(
            n_heads=n_heads,
            hidden_size=hidden_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attn_dropout_prob=attn_dropout_prob,
            layer_norm_eps=layer_norm_eps,
        )
        self.pure_ffn = _SOHCPureFeedForward(
            hidden_size=hidden_size,
            inner_size=inner_size,
            hidden_dropout_prob=hidden_dropout_prob,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
        )

    def _cayley_ortho(self) -> torch.Tensor:
        """Return an N×N SO(N) matrix via Cayley + Newton-Schulz.

        Returns:
            H_res (torch.Tensor): [N, N]  strictly orthogonal.
        """
        W = self.cayley_w
        A = 0.5 * (W - W.t())                          # skew-symmetric
        I = torch.eye(self.expansion_rate, device=W.device, dtype=W.dtype)
        Z = I + A
        X = I - A                                       # X_0 ≈ (I+A)^{-1}
        # Newton-Schulz iterations: X_{k+1} = X_k(2I - Z X_k)
        for _ in range(self.ns_iterations):
            X = X @ (2 * I - Z @ X)
        H_res = (I - A) @ X                              # (I-A)(I+A)^{-1}
        return H_res

    def forward(
        self,
        H: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            H (torch.Tensor):              [B, L, N, D]
            attention_mask (torch.Tensor): [B, 1, 1, L]

        Returns:
            H_next (torch.Tensor): [B, L, N, D]

        SOHC Synthesis (Fixed):
        ::
            Step 1 — SO(N) Isometric Channel Mixing:
              H_res  = CayleyNS(W)                # [N, N] in SO(N)

            Step 2 — Semantic Stream via Dynamic Routing:
              g_pre  = Sigmoid(Linear(H_mean))    # [B, L, N]
              h0     = einsum(g_pre, H)           # [B, L, D]  (soft channel select)
              h_attn = PureAttn(h0) + h0          # [B, L, D]  (inner residual)
              h_ffn  = PureFFN(h_attn)            # [B, L, D]  (semantic update)
              g_post = Sigmoid(Linear(H_mean))    # [B, L, N]
              sem    = einsum(g_post, h_ffn)      # [B, L, N, D]

            Step 3 — Synthesis with depth-residual (BUG FIX):
              H_pre  = H + sem                   # [B, L, N, D]  depth-residual
              H_next = einsum(H_pre, H_res)      # [B, L, N, D]  iso channel mix

            Rationale: Previously `iso_stream + sem_stream` had NO skip-connection
            for H itself. When H_res drifts from I during training, all historical
            information in H is lost → gradient vanishing / representation collapse
            on sparse datasets. The fix mirrors HC's depth-connection: H passes
            through as a residual base, sem_stream adds the incremental update,
            then SO(N) rotation provides isometric channel shuffling on the result.
        """
        B, L, N, D = H.shape

        # ── Step 1: Compute SO(N) channel-mixing matrix ───────────────────
        # H_res: [N, N]  -- isometric rotation across N hyper-hidden channels
        if self.no_ortho:
            H_res = torch.eye(self.expansion_rate, device=H.device, dtype=H.dtype)
        else:
            H_res = self._cayley_ortho()                    # [N, N]

        # ── Step 2: Dynamic Semantic Routing Stream ───────────────────────
        # Use the mean over N channels as a context vector for the gates
        H_mean = H.mean(dim=2)                          # [B, L, D]

        # Gate_pre: soft selection of which channels supply the sub-layer
        if self.static_routing:
            g_pre = torch.ones(B, L, N, device=H.device) / N
        elif self.use_softmax_routing:
            g_pre = torch.softmax(self.gate_pre(H_mean), dim=-1) # [B, L, N]
        else:
            g_pre = torch.sigmoid(self.gate_pre(H_mean))   # [B, L, N]
        # Weighted mix of N channel vectors → one input for Attn/FFN
        h0 = torch.einsum("bln,blnd->bld", g_pre, H)   # [B, L, D]

        # Attention sub-block (keep inner residual for gradient flow)
        h_attn = self.pure_attn(h0, attention_mask) + h0  # [B, L, D]
        if self.use_standard_block:
            h_attn = self.ln1(h_attn)

        # FFN sub-block (pure output, no inner residual — managed by depth-conn)
        h_ffn = self.pure_ffn(h_attn)                  # [B, L, D]
        if self.use_standard_block:
            h_ffn = self.ln2(h_ffn + h_attn)

        # Gate_post: controls how strongly the output is written back per channel
        if self.static_routing:
            g_post = torch.ones(B, L, N, device=H.device) / N
        elif self.use_softmax_routing:
            g_post = torch.softmax(self.gate_post(H_mean), dim=-1) # [B, L, N]
        else:
            g_post = torch.sigmoid(self.gate_post(H_mean)) # [B, L, N]
        # Broadcast h_ffn back to all N channels with individual gates
        sem_stream = torch.einsum(
            "bln,bld->blnd", g_post, h_ffn
        )                                               # [B, L, N, D]

        # ── Step 3: Synthesis (Parallel Isometric Rotation + Semantic Stream) ──
        # Final flow: x_{l+1} = (x_l @ H_res) + x_sem
        H_iso = torch.einsum("blnd,mn->blmd", H, H_res)  # [B, L, N, D]  iso mix
        H_next = H_iso + sem_stream                      # [B, L, N, D]  parallel add
        return H_next


class SOHCEncoder(nn.Module):
    r"""Stack of Wide-SOHC Encoder Layers.

    Args:
        n_layers (int): Number of transformer blocks.
        expansion_rate (int): Number of hyper-hidden channels N. Default: 4.
        ns_iterations (int): Newton-Schulz iterations for SO(N) matrix. Default: 7.
        init_std (float): Init std for Cayley kernel W. Default: 0.02.
    """

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        hidden_size: int,
        inner_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        hidden_act: str,
        layer_norm_eps: float,
        expansion_rate: int = 4,
        ns_iterations: int = 7,
        init_std: float = 0.02,
        no_ortho: bool = False,
        static_routing: bool = False,
        use_standard_block: bool = False,
        use_softmax_routing: bool = False,
    ):
        super().__init__()
        self.expansion_rate = expansion_rate
        self.layers = nn.ModuleList([
            SOHCEncoderLayer(
                hidden_size=hidden_size,
                expansion_rate=expansion_rate,
                n_heads=n_heads,
                inner_size=inner_size,
                hidden_dropout_prob=hidden_dropout_prob,
                attn_dropout_prob=attn_dropout_prob,
                hidden_act=hidden_act,
                layer_norm_eps=layer_norm_eps,
                ns_iterations=ns_iterations,
                init_std=init_std,
                no_ortho=no_ortho,
                static_routing=static_routing,
                use_standard_block=use_standard_block,
                use_softmax_routing=use_softmax_routing,
            )
            for _ in range(n_layers)
        ])

    def forward(
        self,
        H: torch.Tensor,
        attention_mask: torch.Tensor,
        output_all_encoded_layers: bool = True,
    ):
        """
        Args:
            H: [B, L, N, D]  initial hyper-hidden matrix.
            attention_mask: [B, 1, 1, L]
        Returns:
            List[Tensor]: each element [B, L, N, D]
        """
        all_layers = []
        for layer in self.layers:
            H = layer(H, attention_mask)
            if output_all_encoded_layers:
                all_layers.append(H)
        if not output_all_encoded_layers:
            all_layers.append(H)
        return all_layers

