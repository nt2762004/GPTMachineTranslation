from __future__ import annotations

import csv
import html
import math
import random
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import sentencepiece as spm

try:
    import gdown
except ImportError:
    gdown = None



BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
MODEL_PATH = OUTPUT_DIR / "best_model.pt"
TOKENIZER_PATH = OUTPUT_DIR / "ted2020_spm.model"
SOURCE_PATH = OUTPUT_DIR / "cleaned.en"
REFERENCE_PATH = OUTPUT_DIR / "cleaned.vi"
PREFERENCES_PATH = OUTPUT_DIR / "preferences.csv"
VOCAB_SIZE = 52000
MAX_SEQ_LEN = 512


st.set_page_config(page_title="GPT MT Preference Lab", layout="wide", initial_sidebar_state="expanded")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background-color: #0a0e27;
            color: #e4e4e7;
        }

        #MainMenu {
            visibility: hidden;
            height: 0;
        }

        [data-testid="stHeader"] {
            background: transparent;
            padding: 0;
        }

        .stApp {
            background: linear-gradient(135deg, #0a0e27 0%, #0f1437 50%, #1a1f3a 100%);
            color: #e4e4e7;
        }

        section.main > div.block-container {
            padding-top: 2rem;
            padding-bottom: 2.5rem;
            max-width: 1400px;
        }

        /* === SIDEBAR === */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f1437 0%, #141829 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.12);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.5rem;
            padding-bottom: 1rem;
        }

        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div {
            color: #e4e4e7;
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #fff;
            letter-spacing: -0.02em;
        }

        [data-testid="stSidebar"] .stCaption {
            color: #94a3b8;
            font-size: 0.8rem;
        }

        /* === HERO SECTION === */
        .hero {
            padding: 2rem;
            border-radius: 1.5rem;
            background: linear-gradient(135deg, #1a1f3a 0%, #151d2a 100%);
            border: 1px solid rgba(148, 163, 184, 0.15);
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
            position: relative;
            overflow: hidden;
            margin-bottom: 2rem;
        }

        .hero::before {
            content: '';
            position: absolute;
            inset: 0;
            height: 4px;
            background: linear-gradient(90deg, #ff6b6b 0%, #ffa94d 25%, #4ecdc4 50%, #45b7d1 75%, #96ceb4 100%);
            border-radius: 1.5rem 1.5rem 0 0;
        }

        .hero::after {
            content: '';
            position: absolute;
            top: 0;
            right: -200px;
            width: 400px;
            height: 400px;
            background: radial-gradient(circle, rgba(79, 172, 254, 0.08), transparent);
            border-radius: 50%;
            pointer-events: none;
        }

        .hero h1 {
            margin: 0;
            font-size: 2.2rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            color: #fff;
            position: relative;
            z-index: 1;
        }

        .hero p {
            margin: 0.6rem 0 0;
            color: #cbd5e1;
            line-height: 1.6;
            font-size: 1rem;
            position: relative;
            z-index: 1;
        }

        .hero-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem;
            margin-top: 1.2rem;
            position: relative;
            z-index: 1;
        }

        .chip {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            border-radius: 999px;
            background: rgba(148, 163, 184, 0.08);
            border: 1px solid rgba(148, 163, 184, 0.2);
            color: #cbd5e1;
            font-size: 0.85rem;
            font-weight: 500;
            transition: all 0.3s ease;
        }

        .chip:hover {
            border-color: rgba(79, 172, 254, 0.4);
            background: rgba(79, 172, 254, 0.1);
        }

        .chip-dot {
            width: 0.6rem;
            height: 0.6rem;
            border-radius: 50%;
            background: linear-gradient(135deg, #ff6b6b 0%, #4ecdc4 100%);
            animation: pulse 2s ease-in-out infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }

        /* === PANELS === */
        .panel {
            background: linear-gradient(135deg, #111629 0%, #151d2a 100%);
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 1.25rem;
            padding: 1.5rem;
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(10px);
        }

        .panel:hover {
            border-color: rgba(79, 172, 254, 0.2);
            box-shadow: 0 12px 32px rgba(79, 172, 254, 0.1);
        }

        .panel h2, .panel h3 {
            color: #fff;
            margin-bottom: 1rem;
            font-weight: 600;
        }

        .panel p, .panel span, .panel div {
            color: #cbd5e1;
        }

        /* === CANDIDATE CARDS === */
        .candidate-card {
            background: linear-gradient(135deg, #0f1437 0%, #1a1f3a 100%);
            border: 1px solid rgba(79, 172, 254, 0.15);
            border-radius: 1.2rem;
            padding: 1.3rem;
            min-height: 200px;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.4);
            transition: all 0.3s ease;
            display: flex;
            flex-direction: column;
        }

        .candidate-card:hover {
            border-color: rgba(79, 172, 254, 0.4);
            box-shadow: 0 16px 40px rgba(79, 172, 254, 0.15);
            transform: translateY(-2px);
        }

        .candidate-card.selected {
            border-color: rgba(52, 211, 153, 0.6);
            box-shadow: 0 16px 40px rgba(52, 211, 153, 0.2);
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.05) 0%, rgba(52, 211, 153, 0.05) 100%);
        }

        .candidate-title {
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #4f9cfe;
            margin-bottom: 0.8rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .candidate-title::before {
            content: '';
            width: 3px;
            height: 3px;
            border-radius: 50%;
            background: #4f9cfe;
        }

        .candidate-text {
            font-size: 1rem;
            line-height: 1.75;
            color: #e4e4e7;
            white-space: pre-wrap;
            word-break: break-word;
            flex: 1;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.95rem;
            letter-spacing: 0.005em;
        }

        .small-note {
            color: #94a3b8;
            font-size: 0.85rem;
            margin-top: 0.5rem;
        }

        /* === FORMS === */
        [data-testid="stForm"] {
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 1.25rem;
            padding: 1.5rem;
            background: linear-gradient(135deg, #111629 0%, #151d2a 100%);
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.3);
        }

        [data-testid="stForm"] label {
            color: #fff;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }

        /* === INPUTS === */
        textarea, input, [data-baseweb="select"] > div {
            background: rgba(30, 41, 59, 0.8) !important;
            border: 1px solid rgba(148, 163, 184, 0.15) !important;
            color: #e4e4e7 !important;
            border-radius: 0.9rem !important;
            padding: 0.75rem 1rem !important;
            font-family: 'JetBrains Mono', monospace !important;
        }

        textarea:focus, input:focus, [data-baseweb="select"]:focus {
            border-color: rgba(79, 172, 254, 0.5) !important;
            box-shadow: 0 0 0 3px rgba(79, 172, 254, 0.1) !important;
        }

        textarea::placeholder, input::placeholder {
            color: #475569 !important;
        }

        /* === BUTTONS === */
        button,
        .stButton > button,
        .stDownloadButton > button,
        [data-testid="stFormSubmitButton"] button {
            border-radius: 0.9rem !important;
            border: 1px solid rgba(148, 163, 184, 0.15) !important;
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
            color: #cbd5e1 !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3) !important;
            padding: 0.65rem 1.3rem !important;
        }

        button:hover,
        .stButton > button:hover,
        .stDownloadButton > button:hover,
        [data-testid="stFormSubmitButton"] button:hover {
            border-color: rgba(79, 172, 254, 0.4) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 12px 28px rgba(79, 172, 254, 0.2) !important;
        }

        button[kind="primary"],
        .stButton > button[kind="primary"],
        .stForm [data-testid="stFormSubmitButton"] button {
            background: linear-gradient(135deg, #4f9cfe 0%, #0ea5e9 100%) !important;
            border: 0 !important;
            color: #fff !important;
            box-shadow: 0 12px 28px rgba(79, 172, 254, 0.25) !important;
        }

        button[kind="primary"]:hover,
        .stButton > button[kind="primary"]:hover,
        .stForm [data-testid="stFormSubmitButton"] button:hover {
            box-shadow: 0 16px 40px rgba(79, 172, 254, 0.35) !important;
        }

        /* === TABS === */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            border-bottom: 1px solid rgba(148, 163, 184, 0.12);
            padding-bottom: 1rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 0.9rem;
            color: #94a3b8;
            background: transparent;
            border: 1px solid transparent;
            font-weight: 600;
            transition: all 0.3s ease;
        }

        .stTabs [data-baseweb="tab"]:hover {
            color: #cbd5e1;
            border-color: rgba(148, 163, 184, 0.2);
        }

        .stTabs [aria-selected="true"] {
            color: #fff !important;
            background: linear-gradient(135deg, rgba(79, 172, 254, 0.15) 0%, rgba(14, 165, 233, 0.1) 100%) !important;
            border-color: rgba(79, 172, 254, 0.3) !important;
        }

        /* === METRICS === */
        .stMetric {
            background: linear-gradient(135deg, #0f1437 0%, #1a1f3a 100%);
            border: 1px solid rgba(79, 172, 254, 0.15);
            border-radius: 1.1rem;
            padding: 1.2rem;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
        }

        .stMetric > label {
            color: #94a3b8;
            font-size: 0.85rem;
            font-weight: 500;
        }

        .stMetric > div {
            color: #4f9cfe;
            font-size: 1.8rem;
            font-weight: 700;
            margin-top: 0.5rem;
        }

        /* === DATAFRAME === */
        .stDataFrame, [data-testid="stDataFrame"] {
            background: linear-gradient(135deg, #0f1437 0%, #1a1f3a 100%);
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 1.1rem;
            overflow: hidden;
        }

        .stDataFrame table {
            background: transparent;
        }

        /* === RADIO & CHECKBOX === */
        .stRadio > label,
        .stCheckbox > label {
            color: #e4e4e7 !important;
            font-weight: 500;
        }

        .stRadio > label > span,
        .stCheckbox > label > span {
            color: #cbd5e1 !important;
        }

        /* === SLIDERS === */
        .stSlider > label {
            color: #fff;
            font-weight: 600;
        }

        .stSlider [data-baseweb="slider"] {
            color: #4f9cfe;
        }

        /* === ALERTS & INFO === */
        .stAlert {
            border-radius: 1rem;
            border-left: 4px solid;
        }

        .stInfo {
            background: rgba(14, 165, 233, 0.1) !important;
            border-left-color: #0ea5e9 !important;
            color: #7dd3fc !important;
        }

        .stSuccess {
            background: rgba(52, 211, 153, 0.1) !important;
            border-left-color: #34d399 !important;
            color: #86efac !important;
        }

        .stWarning {
            background: rgba(250, 117, 56, 0.1) !important;
            border-left-color: #ff9f43 !important;
            color: #ffb366 !important;
        }

        .stError {
            background: rgba(255, 75, 75, 0.1) !important;
            border-left-color: #ff5252 !important;
            color: #ff9999 !important;
        }

        /* === SPINNER === */
        .stSpinner > div > div {
            border-top-color: #4f9cfe !important;
        }

        /* === ANIMATIONS === */
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .panel, .candidate-card, .hero {
            animation: slideIn 0.4s ease-out;
        }

        /* === SCROLLBAR === */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: rgba(30, 41, 59, 0.5);
        }

        ::-webkit-scrollbar-thumb {
            background: rgba(79, 172, 254, 0.4);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: rgba(79, 172, 254, 0.6);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def clean_text(text: str) -> str:
    normalized = text.lower().strip()
    normalized = "".join(
        character for character in normalized
        if character.isalnum() or character.isspace() or character == "."
    )
    return " ".join(normalized.split())


def extract_translation(decoded_text: str) -> str:
    if "Target:" in decoded_text:
        decoded_text = decoded_text.split("Target:", 1)[1]
    if "[EOS]" in decoded_text:
        decoded_text = decoded_text.split("[EOS]", 1)[0]
    return decoded_text.strip()


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_output, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x


class GPTModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        max_seq_len: int = MAX_SEQ_LEN,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self.create_positional_encoding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout)

    def create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def generate_mask(self, size: int) -> torch.Tensor:
        return torch.triu(torch.ones(size, size), diagonal=1).bool()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, sequence_length = x.size()
        device = x.device
        embeddings = self.embedding(x) * math.sqrt(self.d_model)
        positional_encoding = self.positional_encoding[:sequence_length].to(device)
        x = self.dropout(embeddings + positional_encoding)

        mask = self.generate_mask(sequence_length).to(device)
        for block in self.blocks:
            x = block(x, attn_mask=mask)

        x = self.norm(x)
        return self.fc_out(x)


def download_and_extract_data() -> None:
    """Download and extract model data from Google Drive if not present."""
    # Check if data already exists
    if MODEL_PATH.exists() and TOKENIZER_PATH.exists():
        return
    
    if gdown is None:
        raise ImportError(
            "Missing dependency: gdown. Install it with `pip install gdown` to auto-download model data."
        )
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = OUTPUT_DIR / "data.zip"
    
    # Google Drive file ID from the shared link
    gdrive_id = "1Kg1mBvcl_FZXCWiomTeVd9dKWkaEH5Dm"
    
    try:
        st.info("📥 Downloading model data from Google Drive... This may take a few minutes.")
        gdown.download(
            f"https://drive.google.com/uc?id={gdrive_id}&confirm=t",
            str(zip_path),
            quiet=False
        )
        
        st.info("📦 Extracting files...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(OUTPUT_DIR)
        
        zip_path.unlink()  # Delete zip after extraction
        st.success("✅ Model data ready!")
    except Exception as e:
        st.error(f"❌ Failed to download data: {str(e)}")
        raise


@st.cache_resource(show_spinner=False)
def load_runtime_assets():
    # Download data if needed
    download_and_extract_data()
    
    if spm is None:
        raise ImportError(
            "Missing dependency: sentencepiece. Install it with `pip install sentencepiece` and restart the app."
        )
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {MODEL_PATH}")
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"Missing tokenizer model: {TOKENIZER_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = spm.SentencePieceProcessor(model_file=str(TOKENIZER_PATH))
    model = GPTModel(vocab_size=VOCAB_SIZE, max_seq_len=MAX_SEQ_LEN)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, tokenizer, device


@st.cache_data(show_spinner=False)
def load_examples(limit: int = 2000) -> pd.DataFrame:
    if not SOURCE_PATH.exists() or not REFERENCE_PATH.exists():
        return pd.DataFrame(columns=["source", "reference"])

    records: list[dict[str, str]] = []
    with SOURCE_PATH.open("r", encoding="utf-8") as source_file, REFERENCE_PATH.open("r", encoding="utf-8") as reference_file:
        for index, (source_line, reference_line) in enumerate(zip(source_file, reference_file)):
            if index >= limit:
                break
            source_text = source_line.strip()
            reference_text = reference_line.strip()
            if source_text and reference_text:
                records.append({"source": source_text, "reference": reference_text})

    return pd.DataFrame(records)


def build_prompt(source_text: str) -> str:
    return f"Source: {clean_text(source_text)} Target:"


def encode_prompt(tokenizer: spm.SentencePieceProcessor, source_text: str) -> list[int]:
    return tokenizer.encode(build_prompt(source_text), out_type=int)


def decode_output(tokenizer: spm.SentencePieceProcessor, token_ids: list[int]) -> str:
    return extract_translation(tokenizer.decode(token_ids))


def translate_greedy(
    source_text: str,
    model: GPTModel,
    tokenizer: spm.SentencePieceProcessor,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    generated_tokens = encode_prompt(tokenizer, source_text)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if len(generated_tokens) >= MAX_SEQ_LEN:
                break
            input_tensor = torch.tensor([generated_tokens], dtype=torch.long, device=device)
            logits = model(input_tensor)[0, -1]
            next_token = int(torch.argmax(logits, dim=-1).item())
            generated_tokens.append(next_token)
            if next_token == tokenizer.eos_id():
                break

    return decode_output(tokenizer, generated_tokens)


def translate_sampling(
    source_text: str,
    model: GPTModel,
    tokenizer: spm.SentencePieceProcessor,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    generated_tokens = encode_prompt(tokenizer, source_text)
    temperature = max(temperature, 1e-5)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if len(generated_tokens) >= MAX_SEQ_LEN:
                break
            input_tensor = torch.tensor([generated_tokens], dtype=torch.long, device=device)
            logits = model(input_tensor)[0, -1] / temperature

            if top_k > 0:
                top_values, top_indices = torch.topk(logits, k=min(top_k, logits.shape[-1]))
                probabilities = torch.softmax(top_values, dim=-1)
                sampled_index = torch.multinomial(probabilities, num_samples=1).item()
                next_token = int(top_indices[sampled_index].item())
            else:
                probabilities = torch.softmax(logits, dim=-1)
                next_token = int(torch.multinomial(probabilities, num_samples=1).item())

            generated_tokens.append(next_token)
            if next_token == tokenizer.eos_id():
                break

    return decode_output(tokenizer, generated_tokens)


def translate_beam_search(
    source_text: str,
    model: GPTModel,
    tokenizer: spm.SentencePieceProcessor,
    device: torch.device,
    max_new_tokens: int,
    beam_width: int,
) -> str:
    prompt_tokens = encode_prompt(tokenizer, source_text)
    beams: list[tuple[list[int], float]] = [(prompt_tokens.copy(), 0.0)]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            expanded_beams: list[tuple[list[int], float]] = []
            all_finished = True

            for tokens, score in beams:
                if tokens and tokens[-1] == tokenizer.eos_id():
                    expanded_beams.append((tokens, score))
                    continue

                all_finished = False
                input_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
                logits = model(input_tensor)[0, -1]
                log_probabilities = torch.log_softmax(logits, dim=-1)
                top_log_probs, top_indices = torch.topk(log_probabilities, k=min(beam_width, log_probabilities.shape[-1]))

                for log_prob, token_index in zip(top_log_probs.tolist(), top_indices.tolist()):
                    expanded_beams.append((tokens + [int(token_index)], score + float(log_prob)))

            if all_finished:
                break

            def normalized_score(candidate: tuple[list[int], float]) -> float:
                tokens, score = candidate
                generated_length = max(1, len(tokens) - len(prompt_tokens))
                return score / generated_length

            beams = sorted(expanded_beams, key=normalized_score, reverse=True)[:beam_width]

    return decode_output(tokenizer, beams[0][0])


def append_preference(record: dict[str, str]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = PREFERENCES_PATH.exists()
    with PREFERENCES_PATH.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(record.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


def candidate_card(title: str, text: str) -> str:
    if not text:
        body = "<span class='small-note'>⏳ Pending generation...</span>"
    else:
        body = html.escape(text).replace("\n", "<br>")
    
    return (
        f"<div class='candidate-card'>"
        f"<div class='candidate-title'>{html.escape(title)}</div>"
        f"<div class='candidate-text'>{body}</div>"
        f"</div>"
    )


inject_styles()

st.markdown(
    """
    <div class="hero">
        <h1>🌐 GPT Machine Translation Lab</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model, tokenizer, device = load_runtime_assets()
except Exception as error:
    st.error(str(error))
    st.stop()


if "example_index" not in st.session_state:
    st.session_state.example_index = 0
if "generated_outputs" not in st.session_state:
    st.session_state.generated_outputs = {}
if "selected_candidate" not in st.session_state:
    st.session_state.selected_candidate = None

examples = load_examples(limit=2000)

with st.sidebar:
    st.markdown("### ⚙️ Generation Config")
    st.divider()
    
    st.markdown("**Model Info**")
    st.caption(f"🖥️ Device: `{device.type.upper()}`")
    st.caption(f"🎯 Checkpoint: `{MODEL_PATH.name}`")
    st.caption(f"🔤 Tokenizer: `{TOKENIZER_PATH.name}`")
    
    st.divider()
    st.markdown("**Decoding Parameters**")
    
    max_new_tokens = st.slider(
        "Max tokens", 
        min_value=16, max_value=200, value=80, step=4,
        help="Độ dài tối đa của bản dịch"
    )
    
    beam_width = st.slider(
        "Beam width", 
        min_value=1, max_value=6, value=3, step=1,
        help="Số lượng giả thuyết song song (beam search)"
    )
    
    temperature = st.slider(
        "Temperature", 
        min_value=0.2, max_value=2.0, value=0.9, step=0.1,
        help="Độ đa dạng: cao = đa dạng, thấp = ổn định"
    )
    
    top_k = st.slider(
        "Top-k sampling", 
        min_value=0, max_value=100, value=40, step=5,
        help="Chỉ lấy top-k token có xác suất cao nhất"
    )
    
    st.divider()
    st.markdown("**Data Source**")
    use_example_dataset = st.checkbox(
        "📚 Use example dataset",
        value=True,
        help="Duyệt qua cleaned.en/cleaned.vi"
    )
    
    if use_example_dataset:
        st.caption(f"📊 Dataset size: **{len(examples):,}** pairs")
    
    st.caption("💡 **Tip:** Dùng ví dụ sẵn để duyệt nhanh hoặc nhập câu của riêng bạn.")


[tab_translate] = st.tabs(["🔄 Translate & Prefer"])

with tab_translate:
    left_column, right_column = st.columns([1.2, 0.8], gap="large")

    with left_column:
        st.markdown("### 📥 Input Source")
        
        reference_text = ""
        if use_example_dataset and not examples.empty:
            st.markdown("**Browse Examples**")
            nav_left, nav_mid, nav_right = st.columns(3, gap="small")
            with nav_left:
                if st.button("← Previous", use_container_width=True):
                    st.session_state.example_index = max(0, st.session_state.example_index - 1)
                    st.rerun()
            with nav_mid:
                if st.button("🎲 Random", use_container_width=True):
                    st.session_state.example_index = random.randint(0, len(examples) - 1)
                    st.rerun()
            with nav_right:
                if st.button("Next →", use_container_width=True):
                    st.session_state.example_index = min(len(examples) - 1, st.session_state.example_index + 1)
                    st.rerun()

            current_row = examples.iloc[st.session_state.example_index % len(examples)]
            source_text = current_row["source"]
            reference_text = current_row["reference"]
            
            st.text_area(
                "English source",
                value=source_text,
                height=100,
                disabled=True,
                label_visibility="collapsed"
            )
            st.text_area(
                "Vietnamese reference",
                value=reference_text,
                height=100,
                disabled=True,
                label_visibility="collapsed"
            )
            st.caption(f"Example **{st.session_state.example_index + 1}** / **{len(examples)}**")
        else:
            source_text = st.text_area(
                "English source",
                value="Artificial intelligence is transforming the way we work and live.",
                height=120,
                placeholder="Nhập câu tiếng Anh...",
            )
            st.caption("📌 Pattern: `Source: ... Target:...`")

    with right_column:
        st.markdown("### 🎯 Decoding Modes")
        
        st.markdown("""
        **🟢 Greedy**  
        Nhanh nhất, ổn định, chọn token có xác suất cao nhất
        
        **🔵 Beam Search**  
        Cân bằng: tìm kiếm rộng hơn, chất lượng tốt hơn
        
        **🟣 Sampling**  
        Đa dạng nhất, phù hợp so sánh preference
        """)
        
        st.divider()
        
        generate_button = st.button(
            "✨ Generate All Candidates",
            type="primary",
            use_container_width=True,
            key="gen_btn"
        )

    if generate_button:
        if not source_text.strip():
            st.warning("⚠️ Hãy nhập một câu tiếng Anh trước!")
        else:
            with st.spinner("Generating translations..."):
                st.session_state.generated_outputs = {
                    "🟢 Greedy": translate_greedy(source_text, model, tokenizer, device, max_new_tokens),
                    "🔵 Beam": translate_beam_search(source_text, model, tokenizer, device, max_new_tokens, beam_width),
                    "🟣 Sampling": translate_sampling(
                        source_text,
                        model,
                        tokenizer,
                        device,
                        max_new_tokens,
                        temperature,
                        top_k,
                    ),
                }
                st.session_state.selected_candidate = None
                st.rerun()

    generated_outputs = st.session_state.generated_outputs

    if generated_outputs:
        st.markdown("---")
        
        st.markdown("### 🎪 Candidate Outputs")
        candidate_columns = st.columns(3, gap="medium")
        
        for column, (candidate_name, candidate_text) in zip(candidate_columns, generated_outputs.items()):
            with column:
                st.markdown(candidate_card(candidate_name, candidate_text), unsafe_allow_html=True)



