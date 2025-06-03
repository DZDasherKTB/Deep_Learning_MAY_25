import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import pandas as pd
import re
import random
from torch import nn
import torch.nn.functional as F
import math
from torch import optim
from torch.nn.utils import clip_grad_norm_
# Special tokens
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'

# -------------------------------
# Vocabulary
# -------------------------------
class Vocab:
    def __init__(self, freq_threshold=1):
        self.freq_threshold = freq_threshold
        self.itos = {0: PAD_TOKEN, 1: SOS_TOKEN, 2: EOS_TOKEN, 3: UNK_TOKEN}
        self.stoi = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2, UNK_TOKEN: 3}

    def __len__(self):
        return len(self.itos)

    def build_vocab(self, sentences):
        freq = Counter()
        for sentence in sentences:
            tokens = self.tokenize(sentence)
            # Debug print
            freq.update(tokens)

        for word, count in freq.items():
            if count >= self.freq_threshold and word not in self.stoi:
                idx = len(self.stoi)
                self.stoi[word] = idx
                self.itos[idx] = word

    def tokenize(self, sentence):
        tokens = re.findall(r"\w+|\S", sentence.lower())
        return tokens

    def numericalize(self, sentence):
        tokens = self.tokenize(sentence)
        # print(f"Numericalizing sentence: {sentence}")
        # print(f"Tokens: {tokens}")
        numericalized = [self.stoi.get(token, self.stoi[UNK_TOKEN]) for token in tokens]
        # print(f"Numericalized: {numericalized}")
        return numericalized


class TranslationDataset(Dataset):
    def __init__(self, src_sentences, trg_sentences, src_vocab, trg_vocab, src_len=20, trg_len=20):
        self.src_sentences = src_sentences
        self.trg_sentences = trg_sentences
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.src_len = src_len
        self.trg_len = trg_len

    def __len__(self):
        return len(self.src_sentences)

    def pad_sequence(self, seq, max_len, pad_idx):
        seq = seq[:max_len]
        seq += [pad_idx] * (max_len - len(seq))
        return seq

    def __getitem__(self, idx):
        # print(f"\nGetting item at index {idx}")
        src = self.src_vocab.numericalize(self.src_sentences[idx])
        trg = self.trg_vocab.numericalize(self.trg_sentences[idx])

        src = [self.src_vocab.stoi[SOS_TOKEN]] + src + [self.src_vocab.stoi[EOS_TOKEN]]
        trg = [self.trg_vocab.stoi[SOS_TOKEN]] + trg + [self.trg_vocab.stoi[EOS_TOKEN]]

        # print(f"Src with SOS/EOS tokens: {src}")
        # print(f"Trg with SOS/EOS tokens: {trg}")

        src = self.pad_sequence(src, self.src_len, self.src_vocab.stoi[PAD_TOKEN])
        trg = self.pad_sequence(trg, self.trg_len, self.trg_vocab.stoi[PAD_TOKEN])

        # print(f"Padded Src: {src}")
        # print(f"Padded Trg: {trg}")

        return torch.tensor(src), torch.tensor(trg)


# -------------------------------
# Sentence Cleaning
# -------------------------------
def clean_sentence(sentence):
    sentence = sentence.lower().strip()
    sentence = re.sub(r"[^a-zA-Z0-9äöüß?.!,¿]+", " ", sentence)
    sentence = re.sub(r"\s+", " ", sentence).strip()
    return sentence

# -------------------------------
# Load & Preprocess Data
# -------------------------------

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
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        device = src.device  # Infer from input
        batch_size = src.size(1)
        max_len = trg.size(0)
        vocab_size = self.decoder.output_size

        outputs = torch.zeros(max_len, batch_size, vocab_size).to(device)

        encoder_output, hidden = self.encoder(src)
        n_layers = self.decoder.n_layers
        hidden = hidden.view(n_layers, 2, batch_size, self.decoder.hidden_size)
        hidden = hidden.sum(dim=1)  # Combine forward & backward hidden states

        output = trg[0].to(device)

        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(output, hidden, encoder_output)
            outputs[t] = output
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            output = trg[t] if is_teacher else top1
            output = output.to(device)

        return outputs

def train(model, dataloader, optimizer, criterion, device, max_grad_norm=1.0):
    model.train()
    epoch_loss = 0

    for src, trg in dataloader:
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()
        output = model(src, trg)
        
        # output shape: (trg_len, batch_size, vocab_size)
        # trg shape: (trg_len, batch_size)
        output_dim = output.shape[-1]
        
        output = output[1:].reshape(-1, output_dim)  # ignore SOS token
        trg = trg[1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()

        # Gradient clipping: clips gradients norm to max_grad_norm
        clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for src, trg in dataloader:
            src, trg = src.to(device), trg.to(device)

            output = model(src, trg, teacher_forcing_ratio=0)  # no teacher forcing

            output_dim = output.shape[-1]
            output = output[1:].reshape(-1, output_dim)
            trg = trg[1:].reshape(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

def translate_sentence(sentence, src_vocab, trg_vocab, model, device, max_len=50):
    model.eval()

    # Convert sentence to indices
    tokens = sentence.lower().split()
    tokens = ["<sos>"] + tokens + ["<eos>"]
    src_indices = [src_vocab.stoi.get(token, src_vocab.stoi["<unk>"]) for token in tokens]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(1).to(device)  # (seq_len, 1)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)

        # Match decoder layers (only take forward hidden states from encoder)
        hidden = hidden[:model.decoder.n_layers]

    # Start with <sos>
    trg_indices = [trg_vocab.stoi["<sos>"]]

    for _ in range(max_len):
        trg_tensor = torch.LongTensor([trg_indices[-1]]).to(device)  # (1,)
        with torch.no_grad():
            output, hidden, _ = model.decoder(trg_tensor, hidden, encoder_outputs)
        pred_token = output.argmax(1).item()
        trg_indices.append(pred_token)

        if pred_token == trg_vocab.stoi["<eos>"]:
            break

    # Convert indices back to tokens
    translated_tokens = [trg_vocab.itos[i] for i in trg_indices[1:]]  # skip <sos>
    return translated_tokens

def show_random_translations(dataset, src_vocab, trg_vocab, model, device, num_samples=5):
    model.eval()

    for _ in range(num_samples):
        # Pick a random index
        idx = random.randint(0, len(dataset) - 1)
        
        # Get source and target tensors
        src_tensor, trg_tensor = dataset[idx]
        
        # Convert tensor to original sentence strings (without padding)
        def tensor_to_sentence(tensor, vocab):
            tokens = [vocab.itos[token.item()] for token in tensor if token.item() != vocab.stoi[PAD_TOKEN]]
            tokens = [t for t in tokens if t not in (SOS_TOKEN, EOS_TOKEN)]
            return ' '.join(tokens)

        input_sentence = tensor_to_sentence(src_tensor, src_vocab)
        target_sentence = tensor_to_sentence(trg_tensor, trg_vocab)

        # Translate
        predicted_tokens = translate_sentence(input_sentence, src_vocab, trg_vocab, model, device)
        predicted_sentence = ' '.join(predicted_tokens)

        print("🔹 Input   :", input_sentence)
        print("🎯 Target  :", target_sentence)
        print("🤖 Output  :", predicted_sentence)
        print("-" * 60)

if __name__ == "__main__":
  import torch.multiprocessing
  torch.multiprocessing.freeze_support()
  
  data_df = pd.read_csv('./archive (3)/deu.txt', sep='\t', usecols=[0, 1], names=['en', 'de'], nrows=10000)
  data_df['en'] = data_df['en'].apply(clean_sentence)
  data_df['de'] = data_df['de'].apply(clean_sentence)

  MAX_LEN = 20
  data_df = data_df[
      data_df['en'].apply(lambda x: len(x.split()) <= MAX_LEN) &
      data_df['de'].apply(lambda x: len(x.split()) <= MAX_LEN)
  ]

  # Shuffle and Split
  pairs = list(zip(data_df['en'], data_df['de']))
  random.seed(42)
  random.shuffle(pairs)

  split_idx = int(len(pairs) * 0.9)
  train_pairs = pairs[:split_idx]
  val_pairs   = pairs[split_idx:]

  train_en, train_de = zip(*train_pairs)
  val_en, val_de     = zip(*val_pairs)

  # Build vocab from training set
  src_vocab = Vocab()
  trg_vocab = Vocab()
  src_vocab.build_vocab(train_en)
  trg_vocab.build_vocab(train_de)

  # Dataset and Dataloader
  train_dataset = TranslationDataset(train_en, train_de, src_vocab, trg_vocab)
  val_dataset   = TranslationDataset(val_en, val_de, src_vocab, trg_vocab)

  train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
  val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Dynamically set vocab sizes
  INPUT_SIZE = len(src_vocab)    # Vocabulary size for source language (e.g., English)
  OUTPUT_SIZE = len(trg_vocab)   # Vocabulary size for target language (e.g., German)

  # Model hyperparameters
  EMBED_SIZE = 128        # Embedding dimension
  HIDDEN_SIZE = 256       # GRU hidden size
  N_LAYERS = 1            # Number of GRU layers
  DROPOUT = 0           # Dropout rate

  # Training hyperparameters
  BATCH_SIZE = 64         # Batch size
  SRC_SEQ_LEN = 20        # Source sequence length
  TRG_SEQ_LEN = 20        # Target sequence length

  # Padding token index (same for both since <pad> is index 0)
  pad_token_idx = src_vocab.stoi['<pad>']

  encoder = Encoder(INPUT_SIZE, EMBED_SIZE, HIDDEN_SIZE, N_LAYERS, DROPOUT).to(device)
  decoder = Decoder(EMBED_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, N_LAYERS, DROPOUT).to(device)
  model = Seq2Seq(encoder, decoder).to(device)  # No device arg in your Seq2Seq class

  optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
  criterion = torch.nn.NLLLoss(ignore_index=pad_token_idx)

  num_epochs = 25
  for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion, device)
    val_loss = evaluate(model, val_loader, criterion, device)

    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

show_random_translations(val_dataset, src_vocab, trg_vocab, model, device, num_samples=10)
