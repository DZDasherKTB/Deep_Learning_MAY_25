import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import re

# ---- 1. Define the model class ----
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # last timestep output
        out = self.fc(out)
        return out

# ---- 2. Load vocab and model weights ----
with open("word2idx.pkl", "rb") as f:
    word2idx = pickle.load(f)
with open("idx2word.pkl", "rb") as f:
    idx2word = pickle.load(f)

vocab_size = len(word2idx)
seq_length = 5  # this must match what you used in training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RNNModel(vocab_size, embed_size=64, hidden_size=128, num_layers=1)
model.load_state_dict(torch.load("next_word.pth", map_location=device))
model.to(device)
model.eval()

# ---- 3. Prediction function ----
def predict_next_word(prompt, k=3):
    prompt_tokens = re.findall(r"\b\w+\b|[^\w\s]", prompt.lower())
    input_seq = prompt_tokens[-seq_length:]
    input_ids = [word2idx.get(w, 0) for w in input_seq]
    input_tensor = torch.tensor([input_ids]).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        top_probs, top_indices = torch.topk(probs, k)

        predictions = [idx2word[idx.item()] for idx in top_indices[0]]
        return predictions

# ---- 4. Interactive loop (optional) ----
if __name__ == "__main__":
    print("Welcome to the word predictor!")
    prompt = input("Start with a prompt: ")

    while True:
        predictions = predict_next_word(prompt)
        print(f"Prompt: '{prompt}'")
        print("Predicted next words:", predictions)

        next_word = input("Choose next word (or type 'exit'): ")
        if next_word.lower() == "exit":
            break
        prompt += " " + next_word
