# Machine Translation

## Overview

This project implements an end-to-end **Neural Machine Translation (NMT)** system for **English → Traditional Chinese**, built entirely with **fairseq**.  
The model translates English sentences into Chinese by first **encoding** the English sequence into contextual representations using a Transformer **encoder**, and then **generating** the Chinese output with a **decoder** that performs **self-attention followed by cross-attention over the encoder outputs**.

![seq2seq](https://i.imgur.com/0zeDyuI.png)

The system follows a **two-stage back-translation framework**:

1. **Backward Model (zh → en)** — trained on TED2020 to generate synthetic English translations for monolingual Chinese data.
2. **Forward Model (en → zh)** — trained on a mixture of original parallel data and synthetic data produced through back-translation.

### Dataset Summary

- **Parallel Data**: approximately **390,000** English–Chinese sentence pairs from **TED2020**.
- **Monolingual Data**: approximately **780,000** Chinese sentences used for **back-translation**.
- 📦 [DATA.zip](https://drive.google.com/file/d/1we1EXwFnUSaPSBK7Guljr-I0lSkmCkkn/view?usp=share_link)

---

## Experiment Setup

### Data & Preprocessing

- **Cleaning (`clean_corpus`)**

  - **English**: parenthetical content was removed, hyphens were stripped, and spacing around punctuation was normalized.
  - **Chinese**: full-width characters were converted to half-width, parenthetical spans and underscores were removed, quotes were normalized, and punctuation was separated from words.
  - Length and ratio filters were applied to remove noisy or mismatched sentence pairs.

- **Split**

  - The parallel data was randomly split into **99% / 1%** portions for **train** and **validation** sets at the sentence-pair level.

- **Subword Units**

  - **SentencePiece** with a **unigram** model and a **joint vocabulary of 8,000 tokens** was applied to both English and Chinese.

- **Binarization**

  - All preprocessed data were processed using `fairseq_cli.preprocess` to produce multiple binary datasets under `./DATA/data-bin/XXX`,  
    where `XXX` referred to different stages such as `ted2020`, `mono`, `synthetic`, or `ted2020_with_mono`.  
    Each dataset included `dict.en.txt`, `dict.zh.txt`, and corresponding `.bin` and `.idx` files.

### Model Architecture

We used a standard Transformer encoder–decoder architecture: the English source was encoded by a `TransformerEncoder` into contextual representations, and a `TransformerDecoder` autoregressively generated the Chinese target. **Each decoder layer first applied self-attention over previously generated tokens, then performed cross-attention over the encoder outputs** to align with source semantics, followed by a feed-forward block with residual connections and normalization. Separate `nn.Embedding` tables were used for the source and target vocabularies, with shared weights between the decoder’s input and output embeddings (`share_decoder_input_output_embed=True`).

#### Backward Model (zh → en)

This model was used to translate Chinese monolingual sentences into synthetic English.

```python
arch_args = Namespace(
    encoder_embed_dim=512,
    encoder_ffn_embed_dim=2048,
    encoder_layers=6,
    decoder_embed_dim=512,
    decoder_ffn_embed_dim=2048,
    decoder_layers=6,
    share_decoder_input_output_embed=True,
    dropout=0.1,
)

def add_transformer_args(args):
    args.encoder_attention_heads = 8
    args.encoder_normalize_before = True
    args.decoder_attention_heads = 8
    args.decoder_normalize_before = True
    args.activation_fn = "relu"
    args.max_source_positions = 1024
    args.max_target_positions = 1024
    from fairseq.models.transformer import base_architecture
    base_architecture(arch_args)

add_transformer_args(arch_args)
```

#### Forward Model (en → zh)

The main model was trained on the combination of real and synthetic parallel data.

```python
arch_args = Namespace(
    encoder_embed_dim=768,
    encoder_ffn_embed_dim=3072,
    encoder_layers=6,
    decoder_embed_dim=768,
    decoder_ffn_embed_dim=3072,
    decoder_layers=6,
    share_decoder_input_output_embed=True,
    dropout=0.3,
)

def add_transformer_args(args):
    args.encoder_attention_heads = 12
    args.encoder_normalize_before = True
    args.decoder_attention_heads = 12
    args.decoder_normalize_before = True
    args.activation_fn = "relu"
    args.max_source_positions = 1024
    args.max_target_positions = 1024
    from fairseq.models.transformer import base_architecture
    base_architecture(arch_args)

add_transformer_args(arch_args)
```

### Hyperparameters

- **Epochs**: backward model — 30 epochs; forward model — 48 epochs
- **Batch Size**: token-based batching (`max_tokens=8192` for backward, `4096` for forward)
- **Gradient Accumulation**: 2 mini-batches per update
- **Optimizer**: `AdamW`
- **Scheduler**: **Noam** (warm-up → inverse square-root decay)
- **Loss Function**: `LabelSmoothedCrossEntropy` with **ε = 0.1**

---

## Techniques Used

### Data Cleaning

Thorough text normalization and length filtering reduced corpus noise and improved sentence alignment quality, resulting in cleaner parallel data for training.

### Subword Units

**SentencePiece (unigram)** was applied to prevent out-of-vocabulary issues and ensure consistent subword representations across English and Chinese.

👉 [Hugging Face LLM Course – Subword Tokenization (Unigram)](https://huggingface.co/learn/llm-course/zh-TW/chapter6/7)

### Stronger Transformer

The model replaced RNN/GRU encoders with a **Transformer** architecture that leveraged **self-attention and cross-attention** to capture long-range dependencies in both languages.

### Teacher Forcing

During training, the decoder received the ground-truth token from the previous step, which stabilized convergence and accelerated learning.

### Beam Search

During inference, **beam search with a beam size of 5** was used to balance translation quality and computational efficiency.

### Label-Smoothed Cross-Entropy Criterion

This loss encouraged smoother output distributions by reducing overconfidence, mitigating overfitting, and improving translation robustness.

### Learning Rate Scheduling

The **Noam** schedule was adopted to ensure stable Transformer training:

$$
\text{lr} = d_\text{model}^{-0.5} \cdot
\min(\text{step}^{-0.5},\; \text{step} \cdot \text{warmup}^{-1.5})
$$

> LR curve visualization:

![This is an image](./lr_curve.png)

### Mixed Precision & Gradient Control

- **Automatic Mixed Precision (AMP)** via `GradScaler` ensured numerical stability in FP16.
- **Gradient Accumulation** aggregated multiple mini-batches before each update.
- **Gradient Clipping (`clip_norm=1.0`)** prevented exploding gradients.
- **Token-level averaging** normalized gradients by sample size for consistent optimization.

### Checkpoint Averaging

Averaging the **last 5 checkpoints** formed a lightweight ensemble that improved BLEU stability and reduced evaluation variance.

### Back-Translation

1. The **zh→en** model was trained first.
2. It was used to translate monolingual Chinese sentences into English.
3. The resulting synthetic `(en, zh)` pairs were merged into the parallel corpus.
4. The **en→zh** model was retrained, yielding substantial BLEU improvements.

---

## Results

| Metric               | Score         |
| :------------------- | :------------ |
| **Public BLEU**      | **31.6**      |
| **Private BLEU**     | **30.7**      |
| **Leaderboard Rank** | **31 / 1110** |

![This is an image](./result.png)

---

## How to Reproduce

1. **Clone the repository**

   ```bash
   git clone https://github.com/b06608062/NTU-2021-Machine-Learning-HW5-Machine-Translation.git
   cd NTU-2021-Machine-Learning-HW5-Machine-Translation
   ```

2. **Download and extract the `DATA.zip` to the project root**

3. **Create and activate the virtual environment**

   ```bash
   conda create -n mlhw5py39 python=3.9 -y
   conda activate mlhw5py39
   python -m pip install --upgrade pip
   ```

4. **Install dependencies**

   ```bash
   # https://pytorch.org
   pip install torch --index-url https://download.pytorch.org/whl/cu126
   conda install -y numpy matplotlib tqdm sentencepiece
   git clone https://github.com/pytorch/fairseq.git
   cd fairseq
   git checkout 9a1c497
   pip install .
   pip install --upgrade "omegaconf==2.1.1" "hydra-core==1.1.2"
   pip install "numpy==1.23.5"
   cd ..
   ```

5. **Train models**

   ```bash
   python HW5_backward.py
   python HW5_forward.py
   ```
