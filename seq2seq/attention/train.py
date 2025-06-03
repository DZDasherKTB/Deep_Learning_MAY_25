import torch
import torch.nn.functional as F
from torch import optim
from torch.nn.utils import clip_grad_norm_
from model import Encoder,Decoder,Seq2Seq
from utils import train_loader,val_loader,src_vocab,trg_vocab

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dynamically set vocab sizes
INPUT_SIZE = len(src_vocab)    # Vocabulary size for source language (e.g., English)
OUTPUT_SIZE = len(trg_vocab)   # Vocabulary size for target language (e.g., German)

# Model hyperparameters
EMBED_SIZE = 256        # Embedding dimension
HIDDEN_SIZE = 512       # GRU hidden size
N_LAYERS = 2            # Number of GRU layers
DROPOUT = 0.3           # Dropout rate

# Training hyperparameters
BATCH_SIZE = 64         # Batch size
SRC_SEQ_LEN = 20        # Source sequence length
TRG_SEQ_LEN = 20        # Target sequence length

# Padding token index (same for both since <pad> is index 0)
pad_token_idx = src_vocab.stoi['<pad>']

encoder = Encoder(INPUT_SIZE, EMBED_SIZE, HIDDEN_SIZE, N_LAYERS, DROPOUT).to(device)
decoder = Decoder(EMBED_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, N_LAYERS, DROPOUT).to(device)
model = Seq2Seq(encoder, decoder).to(device)  # No device arg in your Seq2Seq class

optimizer = optim.Adam(model.parameters())
criterion = torch.nn.NLLLoss(ignore_index=pad_token_idx)

num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion, device)
    val_loss = evaluate(model, val_loader, criterion, device)

    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
