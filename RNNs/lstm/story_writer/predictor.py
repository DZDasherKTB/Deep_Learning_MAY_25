import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import re

# ---- 1. Define the model class ----
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# ---- 2. Load vocab and model weights ----
with open("./word2idx.pkl", "rb") as f:
    word2idx = pickle.load(f)
with open("./idx2word.pkl", "rb") as f:
    idx2word = pickle.load(f)

vocab_size = len(word2idx)
seq_length = 30 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = LSTMModel(vocab_size, embed_size=64, hidden_size=128, num_layers=2)
model.load_state_dict(torch.load("next_word.pth", map_location=device))
model.to(device)
model.eval()

# ---- 3. Prediction function ----
def predict_next_word(prompt, k=3):
    model.eval()
    prompt_tokens = re.findall(r'[\u0900-\u097F]+|[^\s\w]', prompt, re.UNICODE)
    input_seq = prompt_tokens[-seq_length:]
    input_ids = [word2idx.get(w, 0) for w in input_seq]
    input_tensor = torch.tensor([input_ids]).to(next(model.parameters()).device) 

    with torch.no_grad():
        output = model(input_tensor)                
        probs = torch.softmax(output, dim=1)         # probabilities
        top_probs, top_indices = torch.topk(probs, k)

        predictions = [idx2word[idx.item()] for idx in top_indices[0]]
        return predictions

def tokenize_hindi(text):
    return re.findall(r'[\u0900-\u097F]+|[^\s\w]', text, re.UNICODE)

def encode(text, word2idx):
    tokens = tokenize_hindi(text)
    return [word2idx.get(token, word2idx.get("<unk>", 1)) for token in tokens]

def top_k_sampling(logits, top_k=10, temperature=1.0):
    logits = logits / temperature
    top_k = min(top_k, logits.size(-1))  # Ensure k is not larger than vocab size

    top_k_logits, top_k_indices = torch.topk(logits, top_k)
    probs = torch.softmax(top_k_logits, dim=-1)
    next_idx = torch.multinomial(probs, 1).item()
    return top_k_indices[next_idx].item()


def generate_story(prompt, seq_length=10, max_words=50, temperature=1.0, top_k=10, device='cpu'):
    model.eval()
    model.to(device)

    words = tokenize_hindi(prompt)  # use Hindi tokenizer

    for _ in range(max_words):
        input_seq = encode(''.join(words[-seq_length:]), word2idx)
        input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            logits = output[0]  # logits for the next token
            next_idx = top_k_sampling(logits, top_k=top_k, temperature=temperature)
            next_word = idx2word.get(next_idx, "<unk>")

        if next_word == "<eos>":
            break

        words.append(next_word)

    return ' '.join(words)

# ---- 4. Interactive loop ----
if __name__ == "__main__":
    prompt = "जीकर भी जी न सके"
    print("📝 Random Story Generator\n")
    story = generate_story(prompt, max_words=100, temperature=0.8, top_k=2)
    print("🧠 Generated Story:\n", story)

