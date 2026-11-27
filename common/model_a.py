# Modelo A: encoder BiLSTM + decoder LSTM con atención de producto punto
# Atención como en las slides de RNNs / encoder-decoder:
#   score(h^d_{i-1}, h^e_j) = h^d_{i-1} . h^e_j
#   alpha_ij = softmax(score)
#   c_i = sum_j alpha_ij h^e_j
#   h^d_i = LSTM([emb(y_{i-1}); c_i], h^d_{i-1})
#   y_i = softmax(V h^d_i)
# El encoder es bidireccional (tamaño 2H), así que el decoder usa tamaño 2H para que el producto punto cuadre.
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, dropout, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0)

    def forward(self, src, lengths):
        # src: [B, L]
        output = self.dropout(self.embedding(src))
        packed = pack_padded_sequence(output, lengths, batch_first=True, enforce_sorted=False)
        packed, (h, c) = self.lstm(packed)
        output, _ = pad_packed_sequence(packed, batch_first=True, total_length=src.size(1))
        # output: [B, L, 2H]; h, c: [capas*2, B, H] -> juntamos adelante y atrás: [capas, B, 2H]
        h = h.view(self.num_layers, 2, h.size(1), h.size(2))
        c = c.view(self.num_layers, 2, c.size(1), c.size(2))
        h = torch.cat([h[:, 0], h[:, 1]], dim=-1)
        c = torch.cat([c[:, 0], c[:, 1]], dim=-1)
        return output, (h, c)


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, dropout, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim + hidden_size, hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def attention(self, h_prev, enc_output, src_mask):
        # h_prev: [B, 2H], enc_output: [B, L, 2H], src_mask: [B, L]
        scores = torch.matmul(enc_output, h_prev.unsqueeze(2)).squeeze(2)   # [B, L]
        scores = scores.masked_fill(src_mask == 0, -1e9)                    # no mirar el padding
        alpha = F.softmax(scores, dim=-1)                                   # [B, L]
        context = torch.matmul(alpha.unsqueeze(1), enc_output)              # [B, 1, 2H]
        return context, alpha

    def forward_step(self, y_prev, state, enc_output, src_mask):
        # un paso: y_prev [B] -> logits [B, V]
        h, c = state
        emb = self.dropout(self.embedding(y_prev)).unsqueeze(1)             # [B, 1, E]
        context, alpha = self.attention(h[-1], enc_output, src_mask)       # h[-1]: última capa
        output, (h, c) = self.lstm(torch.cat([emb, context], dim=-1), (h, c))
        logits = self.output_layer(self.dropout(output.squeeze(1)))
        return logits, (h, c), alpha


class Seq2SeqAttention(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, emb_dim=256, hidden_size=256, dropout=0.3, num_layers=1):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, emb_dim, hidden_size, dropout, num_layers)
        self.decoder = Decoder(tgt_vocab_size, emb_dim, 2 * hidden_size, dropout, num_layers)

    def forward(self, src, lengths, target_input):
        # teacher forcing: en cada paso el decoder recibe la palabra correcta anterior
        enc_output, state = self.encoder(src, lengths)
        src_mask = (src != 0)
        logits, alphas = [], []
        for t in range(target_input.size(1)):
            step_logits, state, alpha = self.decoder.forward_step(target_input[:, t], state, enc_output, src_mask)
            logits.append(step_logits)
            alphas.append(alpha)
        return torch.stack(logits, dim=1), torch.stack(alphas, dim=1)       # [B, T, V], [B, T, L]

    def translate(self, src, lengths, sos_idx, eos_idx, max_len=80):
        # decodificación greedy: en cada paso se toma la palabra más probable
        with torch.no_grad():
            enc_output, state = self.encoder(src, lengths)
            src_mask = (src != 0)
            y_prev = torch.LongTensor([sos_idx] * src.size(0)).to(src.device)
            outputs, alphas = [], []
            finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
            for _ in range(max_len):
                logits, state, alpha = self.decoder.forward_step(y_prev, state, enc_output, src_mask)
                y_prev = torch.argmax(logits, dim=-1)
                outputs.append(y_prev)
                alphas.append(alpha)
                finished = finished | (y_prev == eos_idx)
                if finished.all():
                    break
        return torch.stack(outputs, dim=1), torch.stack(alphas, dim=1)      # [B, T], [B, T, L]
