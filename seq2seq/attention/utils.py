import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import pandas as pd
import re
import random

# Special tokens
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'

# -------------------------------
# Vocabulary
# -------------------------------
class Vocab:
    def __init__(self, freq_threshold=2):
        self.freq_threshold = freq_threshold
        self.itos = {0: PAD_TOKEN, 1: SOS_TOKEN, 2: EOS_TOKEN, 3: UNK_TOKEN}
        self.stoi = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2, UNK_TOKEN: 3}

    def __len__(self):
        return len(self.itos)

    def build_vocab(self, sentences):
        freq = Counter()
        for sentence in sentences:
            tokens = self.tokenize(sentence)
            freq.update(tokens)

        for word, count in freq.items():
            if count >= self.freq_threshold and word not in self.stoi:
                idx = len(self.stoi)
                self.stoi[word] = idx
                self.itos[idx] = word

    def tokenize(self, sentence):
        return re.findall(r"\w+|\S", sentence.lower())

    def numericalize(self, sentence):
        tokens = self.tokenize(sentence)
        return [self.stoi.get(token, self.stoi[UNK_TOKEN]) for token in tokens]

# -------------------------------
# Dataset
# -------------------------------
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
        src = self.src_vocab.numericalize(self.src_sentences[idx])
        trg = self.trg_vocab.numericalize(self.trg_sentences[idx])

        src = [self.src_vocab.stoi[SOS_TOKEN]] + src + [self.src_vocab.stoi[EOS_TOKEN]]
        trg = [self.trg_vocab.stoi[SOS_TOKEN]] + trg + [self.trg_vocab.stoi[EOS_TOKEN]]

        src = self.pad_sequence(src, self.src_len, self.src_vocab.stoi[PAD_TOKEN])
        trg = self.pad_sequence(trg, self.trg_len, self.trg_vocab.stoi[PAD_TOKEN])

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

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=128, shuffle=False)
