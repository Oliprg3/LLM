import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B,T,E = x.size()
        q = self.query(x).view(B,T,self.heads,self.head_dim).transpose(1,2)
        k = self.key(x).view(B,T,self.heads,self.head_dim).transpose(1,2)
        v = self.value(x).view(B,T,self.heads,self.head_dim).transpose(1,2)
        attn_scores = torch.matmul(q, k.transpose(-2,-1))/math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(T,T,device=x.device)).unsqueeze(0).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(mask==0, float('-inf'))
        attn = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1,2).contiguous().view(B,T,E)
        return self.fc(out)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, heads, ff_hidden, dropout):
        super().__init__()
        self.attn = CausalSelfAttention(embed_dim, heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden),
            nn.GELU(),
            nn.Linear(ff_hidden, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, heads, ff_hidden, layers, seq_len, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(seq_len, embed_dim)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, heads, ff_hidden, dropout) for _ in range(layers)])
        self.ln = nn.LayerNorm(embed_dim)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.seq_len = seq_len

    def forward(self, x):
        B,T = x.size()
        pos = torch.arange(0,T,device=x.device).unsqueeze(0)
        x = self.token_emb(x)+self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        return self.fc_out(x)

    def generate(self, idx, max_len, temperature=1.0):
        for _ in range(max_len):
            idx_cond = idx[:,-self.seq_len:]
            logits = self.forward(idx_cond)
            logits = logits[:,-1,:]/temperature
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            idx = torch.cat((idx,next_idx), dim=1)
        return idx
