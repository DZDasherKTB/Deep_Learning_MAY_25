# 🧠 Deep Learning Projects – From Scratch to Transformers

Welcome to my Deep Learning playground.  
This repo is a culmination of my journey — starting with basic CNNs to building full Transformer and GAN models from scratch.

---

## 🚀 Highlights of the Journey

### 🟦 CNNs
- ✅ Achieved 90%+ accuracy on CIFAR100  
  [📄 Code](./EMNIST_Letters/Cifar100.ipynb)
- 🔗 Learned residual connections (ResNet-style)  
  [📄 Code](./EMNIST_Letters/Cifar100_ResidualBlockCNN%20copy.ipynb)

### 🔁 RNNs, LSTMs, GRUs
- 📈 Sin wave predictors
- 🧠 Next-word predictors  
  [Play](./RNNs/rnn/next_word/play.py) | [LSTM Version](./RNNs/lstm/next_word/play.py)
- ✍️ Sentence generators  
  [RNN](./RNNs/rnn/next_word/sentence.py) | [LSTM](./RNNs/lstm/next_word/sentence.py)
- 📚 Story generator (a bit chaotic!)  
  [Script](./RNNs/lstm/story_writer/new_predictor.py)

---

## 🌍 Language Translation Models

### 🔃 Sequence-to-Sequence (Encoder-Decoder)
- 🇫🇷 French → English (without attention)  
  [📄 Code](./seq2seq/f2e/encoder_decoder.ipynb)
- 🇫🇷 French → English (with attention) — _great results_ ✅  
  [📄 Code](./seq2seq/f2e_attention/encoder_decoder.ipynb)
- 🇩🇪 English → German (first trial)  
  [📄 Code](./seq2seq/attention/tinker.ipynb)

### 🧠 Transformer (From Scratch)
- 📚 Implemented from scratch (Multi-head Attention, Positional Encoding, etc.)  
  [📄 Model Notebook](./Transformers/scratch/model.ipynb)
- 🇳🇱 English → Dutch  
  [🔗 Code Folder](./Transformers/english2german/)

---

## 🎨 GANs (Image Generation)

### 🔳 Basic GAN
- MNIST digit generation  
  [🔗 Code Folder](./GAN/GAN_MNIST)

### 🧑‍🦲 DCGAN
- Realistic human face generation  
  [🔗 Code Folder](./GAN/DCGAN)
- [▶️ Run this](./GAN/DCGAN/run.py) to generate your own faces

---

## 💡 Try It Yourself

| Task | Try This |
|------|----------|
| 🧠 Sentence generation | [`sentence.py`](./RNNs/lstm/next_word/sentence.py) |
| 📚 Story writing | [`new_predictor.py`](./RNNs/lstm/story_writer/new_predictor.py) |
| 🌍 Translate English → German | [`Transformer Notebook`](./Transformers/english2german/model.ipynb) |
| 🎨 Generate faces (DCGAN) | [`run.py`](./GAN/DCGAN/run.py) |

---

## 🤝 Why I’m Sharing This

I’m not doing this to show off models — I’m hoping this reaches someone just starting out, feeling lost like I once was.

This repo is **for you** — clone it, run it, tweak it, break it, fix it, and learn from it.

If you ever need help, I promise to give it my best — to the maximum of what I know and can guide you through.

Let’s push the boundaries of what we can build together 💻🚀  
Reach out if you’re on a similar path — I’d love to connect!

---

## 🏷️ Tags
`Deep Learning`, `Transformer`, `GAN`, `RNN`, `LSTM`, `CNN`, `Machine Learning`, `Open Source`, `Sequence-to-Sequence`, `Language Models`, `Face Generator`, `Neural Networks`
