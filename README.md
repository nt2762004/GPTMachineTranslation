# GPT Machine Translate

This project performs English-Vietnamese Machine Translation using GPT (Generative Pre-trained Transformer) architecture. The project includes both training from scratch and fine-tuning a pretrained model.

Link Deployed: https://gptmachinetranslation.streamlit.app/

## Folder Structure

```
├── gpt-mt-pre.ipynb              # Notebook for fine-tuning GPT-2
├── gpt-mt.ipynb                  # Notebook for training GPT from scratch
├── README.md                     # Project description file
├── en-vi.txt/                    # Original bilingual data (TED2020)
└── Train_spm/                    # Folder for training SentencePiece Tokenizer
    ├── Train_spm.ipynb           # Notebook for training tokenizer
    ├── ted2020_spm.model         # Tokenizer model
    └── ...
```

### `Train_spm/Train_spm.ipynb` (Data Preparation & Tokenizer)
This notebook performs preprocessing steps and prepares the tokenizer for the model.

*   **Goal:** Clean text data and train SentencePiece Tokenizer model.
*   **Main steps:**
    *   **Preprocessing:**
        *   Convert text to lowercase.
        *   Remove unnecessary special characters, normalize spaces.
    *   **Tokenization:**
        *   Train a custom tokenizer on the TED2020 dataset using the `sentencepiece` library.
        *   Create vocabulary (vocab) and tokenizer model (`.model`).

### `gpt-mt.ipynb` (Training from Scratch)
This notebook builds and trains a GPT model from the beginning (Non-pretrained).

*   **Goal:** Build and train a Transformer Decoder model for machine translation tasks.
*   **Main steps:**
    *   **Load Tokenizer:** Use the tokenizer created in the previous step.
    *   **Data Formatting:** Prepare sentence pairs `Source: [English] Target: [Vietnamese]`.
    *   **Build Model:**
        *   Transformer Decoder architecture (Embedding, Positional Encoding, Multi-head Self-Attention, Feed Forward Network).
    *   **Training:**
        *   Use **CrossEntropyLoss** and **AdamW** optimizer.
        *   Apply **Teacher Forcing**.
    *   **Evaluation:**
        *   Use **BLEU Score** to evaluate accuracy.
        *   Inference using Greedy Search method.

### `gpt-mt-pre.ipynb` (Fine-tuning GPT-2)
This notebook uses a pretrained GPT-2 model to fine-tune for translation tasks.

*   **Goal:** Use knowledge from the Pretrained model to improve translation quality.
*   **Main steps:**
    *   **Load Pretrained Model:** Use `GPT2LMHeadModel` and `GPT2Tokenizer` from the `transformers` library.
    *   **Fine-tuning:** Adjust model weights on the English-Vietnamese dataset.
    *   **Evaluation:** Use **ROUGE Score** to evaluate coverage.

## Installation Requirements

To run these notebooks, you need to install the following Python libraries:

```bash
pip install torch transformers sentencepiece nltk numpy
```

## Streamlit Preference App

The repository also includes a Streamlit app that loads the trained checkpoint in `output/best_model.pt` and the SentencePiece tokenizer in `output/ted2020_spm.model`.

Run it with:

```bash
streamlit run app.py
```

The app generates greedy, beam, and sampling candidates for a source sentence, then lets you record which output you prefer. Preferences are saved to `output/preferences.csv`.

---

## Details in Kaggle Links

*   [Non-pretrain Version](https://www.kaggle.com/code/tneduvn/gpt-mt)
*   [Pretrain Version](https://www.kaggle.com/code/tneduvn/gpt-mt-pre)
