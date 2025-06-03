import torch
import random
from torch import nn
import torch.nn.functional as F
import math

class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, n_layers=1, dropout=0.5):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

    def forward(self, src, hidden=None):
        embedded = self.embed(src)
        outputs, hidden = self.gru(embedded, hidden)
        # Sum bidirectional outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs, hidden


class Attention(nn.Module):  # Fixed capitalization from nn.module -> nn.Module
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1.0 / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)  # (batch, timestep, hidden)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # (batch, timestep, hidden)
        attn_energies = self.score(h, encoder_outputs)  # (batch, timestep)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)  # (batch, 1, timestep)

    def score(self, hidden, encoder_outputs):
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], 2)))  # (batch, timestep, hidden)
        energy = energy.transpose(1, 2)  # (batch, hidden, timestep)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # (batch, 1, hidden)
        energy = torch.bmm(v, energy)  # (batch, 1, timestep)
        return energy.squeeze(1)  # (batch, timestep)


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size, n_layers=1, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embed = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size, n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, last_hidden, encoder_outputs):
        embedded = self.embed(input).unsqueeze(0)  # (1, batch, embed_size)
        embedded = self.dropout(embedded)

        attn_weights = self.attention(last_hidden[-1], encoder_outputs)  # (batch, 1, seq_len)
        context = torch.bmm(attn_weights, encoder_outputs.transpose(0, 1))  # (batch, 1, hidden)
        context = context.transpose(0, 1)  # (1, batch, hidden)

        rnn_input = torch.cat([embedded, context], 2)  # (1, batch, embed+hidden)
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (batch, hidden)
        context = context.squeeze(0)  # (batch, hidden)
        output = self.out(torch.cat([output, context], 1))  # (batch, output_size)
        output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(1)
        max_len = trg.size(0)
        vocab_size = self.decoder.output_size

        outputs = torch.zeros(max_len, batch_size, vocab_size).to(self.device)

        encoder_output, hidden = self.encoder(src)
        hidden = hidden[:self.decoder.n_layers]

        # First input to the decoder is the <sos> token (usually trg[0])
        output = trg[0].to(self.device)

        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(output, hidden, encoder_output)
            outputs[t] = output
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)  # best guess
            output = trg[t] if is_teacher else top1
            output = output.to(self.device)

        return outputs
