import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention calculation"""

    def __init__(self, dropout_rate=0.0, **kwargs):
        """Initialize ScaledDotProductAttention
        Args:
            dropout_rate (float): attention dropout_rate rate
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, q, k, v, attn_mask=None):
        """Forward
        Args:
            q (torch.Tensor): Query matrix, (B, T_q, D_q)
            k (torch.Tensor): Key matrix, (B, T_k, D_k)
            v (torch.Tensor): Value matrix, (B, T_v, D_v) T_v = T_k, D_v = D_k
            attn_mask (torch.BoolTensor | None): Mask tensor. True element will be masked.
        Returns:
            output (B, T_q, D_v); attention (B, T_q, T_k)
        """
        attention = torch.bmm(q, k.permute(0, 2, 1))  # (B, T_q, T_k)

        # Scale
        attention *= k.size(-1) ** -0.5

        if attn_mask is not None:
            attention.masked_fill_(attn_mask, -np.inf)  # positions that require masking are now -np.inf

        attention = F.softmax(attention, dim=-1)

        attention = self.dropout(attention)

        output = attention.bmm(v)  # (B, T_q, D_v)

        return output, attention



class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=512, num_heads=8, dropout_rate=0.0,attention_type='scaled_dot'):
        super().__init__()

        self.h_size = model_dim
        self.num_heads = num_heads
        self.head_h_size = model_dim // num_heads

        self.linear_q = nn.Linear(self.h_size, self.h_size)
        self.linear_k = nn.Linear(self.h_size, self.h_size)
        self.linear_v = nn.Linear(self.h_size, self.h_size)

        self.attention = ScaledDotProductAttention( q_dim=self.head_h_size, k_dim=self.head_h_size)
        # self.attention = self.build_attention( q_dim=self.head_h_size, k_dim=self.head_h_size)
        #self.fc_o = nn.Linear(self.head_h_size, self.head_h_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.lnorm = nn.LayerNorm(model_dim)

    def forward(self, q, k, v, attn_mask=None):
        batch_size = q.size(0)

        # Residual
        residual = q

        # Linear projection
        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)

        # Form multi heads
        q = q.view(self.num_heads * batch_size, -1, self.head_h_size)  # (h * B, T_q, D / h)
        k = k.view(self.num_heads * batch_size, -1, self.head_h_size)  # (h * B, T_k, D / h)
        v = v.view(self.num_heads * batch_size, -1, self.head_h_size)  # (h * B, T_v, D / h)

        if attn_mask is not None:
            attn_mask = attn_mask.repeat(self.num_heads, 1, 1)  # (h * B, T_q, T_k)
    
        context, attention = self.attention(q, k, v, attn_mask=attn_mask)
        #context, attention = ScaledDotProductAttention()(q, k, v, attn_mask=attn_mask)
        # context: (h * B, T_q, D_v) attention: (h * B, T_q, T_k)

        # Concatenate heads
        context = context.view(batch_size, -1, self.h_size)  # (B, T_q, D)

        # Dropout
        output = self.dropout(context)  # (B, T_q, D)

        # Residual connection and Layer Normalization
        output = self.lnorm(residual + output)  # (B, T_q, D)

        return output, attention

    



class FeedForward(nn.Module):

    def __init__(self, model_dim, hidden_dim, dropout_rate=0.5):
        super().__init__()

        self.model_dim = model_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.linear1 = nn.Linear(self.model_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.model_dim)
        self.norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        output = self.linear2(F.relu(self.linear1(x)))

        output = self.dropout(output)

        output = self.norm(output + x)
        return output