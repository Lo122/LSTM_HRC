# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.utils.data as data
# import math
# import copy

# # Pose windows
# X_cur:   [B, T, pose_dim]        # 当前window pose序列
# X_hist:  [B, M, T, pose_dim]     # 当前step内的历史M个windows（不跨step）

# # Conditioning inputs
# step_id:       [B]               # hard-coded / GT
# element_feat:  [B, element_dim]  # 从design/BIM映射来的element信息
# env_feat:      [B, env_dim]      # 从environment model来的摘要信息（可选）
# pos_feat:      [B, pos_dim]      # human/robot位置、距离、zone等


# class TransformerEncoderBlock(nn.Module):
#     def __init__(self, d_model, num_heads, d_ff,dropout=0.1):
#         super(TransformerEncoderBlock, self).__init__()
#         self.slef_atten = MultiHeadSelfAttention(d_model, num_heads)
#         self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, X,mask=None):
#         # X: [B, L, d_model]

#         # --- Multi-head self attention ---
#         atten_output = self.slef_atten(query=X, key=X, value=X, mask=mask)     # [B, L, d_model]
#         X = self.norm1(X + self.dropout(atten_output))       # Add & Norm
#         ff_output = self.feed_forward(X)                        # [B, L, d_model]
#         X = self.norm2(X + self.dropout(ff_output))           # Add & Norm

#         return X


# # --------------------------------------------------
# # Key blocks of the Transformer Encoder:
# # - Multi-Head Self Attention
# # - Position-wise Feed Forward Network
# # - Positional encoding

# # multi-head self attention block

# class MultiHeadSelfAttention(nn.Module):
#     def __init__(self, d_model, num_heads):
#         super(MultiHeadSelfAttention, self).__init__()
#         assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.d_k = d_model // num_heads
        
#         self.W_q = nn.Linear(d_model, d_model)
#         self.W_k = nn.Linear(d_model, d_model)
#         self.W_v = nn.Linear(d_model, d_model)
#         self.W_o = nn.Linear(d_model, d_model)
        
#     def scaled_dot_product_attention(self, Q, K, V, mask=None):
#         attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
#         if mask is not None:
#             attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
#         attn_probs = torch.softmax(attn_scores, dim=-1)
#         output = torch.matmul(attn_probs, V)
#         return output
        
#     def split_heads(self, x):
#         batch_size, seq_length, d_model = x.size()
#         return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
#     def combine_heads(self, x):
#         batch_size, _, seq_length, d_k = x.size()
#         return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
#     def forward(self, Q, K, V, mask=None):
#         Q = self.split_heads(self.W_q(Q))
#         K = self.split_heads(self.W_k(K))
#         V = self.split_heads(self.W_v(V))
        
#         attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
#         output = self.W_o(self.combine_heads(attn_output))
#         return output

# # position-wise feed forward network 
# class PositionWiseFeedForward(nn.Module):
#     def __init__(self, d_model, d_ff):
#         super(PositionWiseFeedForward, self).__init__()
#         self.fc1 = nn.Linear(d_model, d_ff)
#         self.fc2 = nn.Linear(d_ff, d_model)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         return self.fc2(self.relu(self.fc1(x)))

# # positional encoding
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_seq_length):
#         super(PositionalEncoding, self).__init__()
        
#         pe = torch.zeros(max_seq_length, d_model)
#         position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
        
#         self.register_buffer('pe', pe.unsqueeze(0))
        
#     def forward(self, x):
#         return x + self.pe[:, :x.size(1)]
    
# # --------------------------------------------------
