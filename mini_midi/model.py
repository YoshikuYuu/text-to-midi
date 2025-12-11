import torch
from torch import nn
from torch.nn import functional as F
from transformers import T5EncoderModel

class MiniText2MIDI(nn.Module):
    """
    This code was adapted from https://github.com/AMAAI-Lab/Text2midi/tree/main.
    """
    def __init__(self, 
                 remi_vocab_size, 
                 max_remi_tokens, 
                 d_model=256,
                 n_heads=4,
                 n_layers=4,
                 dim_ff=512):
        super().__init__()
        
        self.encoder = T5EncoderModel.from_pretrained("google/flan-t5-base")
        for p in self.encoder.parameters():
            p.requires_grad = False
        enc_hidden = self.encoder.config.d_model

        self.decoder = TransformerDecoder(
            vocab_size=remi_vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dim_ff=dim_ff,
            max_len=max_remi_tokens
        )

        self.enc_proj = nn.Linear(enc_hidden, d_model)

    def forward(self, text_input_ids, attention_mask, decoder_input_ids, labels=None):
        with torch.no_grad():
            enc_out = self.encoder(input_ids=text_input_ids, attention_mask=attention_mask).last_hidden_state
        enc_out = self.enc_proj(enc_out)
        logits = self.decoder(decoder_input_ids, enc_out)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100
            )
        return {"logits": logits, "loss": loss}

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=4, n_heads=4, dim_ff=512, max_len=1024):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model))

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, dim_ff)
        for _ in range(n_layers)])

        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, enc_outputs):
        B, T = input_ids.shape
        x = self.embedding(input_ids) + self.pos_emb[:, :T]

        # causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, device=input_ids.device), diagonal=1
        ).bool()

        for layer in self.layers:
            x = layer(
                x,
                enc_outputs,
                tgt_mask=causal_mask
            )

        logits = self.lm_head(x)
        return logits


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, n_heads=4, dim_ff=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_outputs, tgt_mask=None, tgt_padding_mask=None):
        # Self-attention (decoder attends to itself)
        _x = x
        x, _ = self.self_attn(
            x, x, x,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_padding_mask
        )
        x = _x + self.dropout(x)
        x = self.norm1(x)

        # Cross-attention (decoder attends to encoder)
        _x = x
        x, _ = self.cross_attn(
            x, enc_outputs, enc_outputs
        )
        x = _x + self.dropout(x)
        x = self.norm2(x)

        # Feedforward
        _x = x
        x = self.linear2(F.relu(self.linear1(x)))
        x = _x + self.dropout(x)
        x = self.norm3(x)

        return x


