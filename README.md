# GPT Machine Translate

Dự án này thực hiện dịch máy (Machine Translation) Anh-Việt sử dụng kiến trúc GPT (Generative Pre-trained Transformer). Dự án bao gồm cả phương pháp huấn luyện từ đầu (training from scratch) và tinh chỉnh mô hình đã huấn luyện trước (fine-tuning pretrained model).

## Cấu trúc Thư mục

```
├── gpt-mt-pre.ipynb              # Notebook fine-tuning GPT-2
├── gpt-mt.ipynb                  # Notebook training GPT from scratch
├── Link_GPT_MachineTranslate.txt # Link Kaggle
├── README.md                     # File mô tả dự án
├── en-vi.txt/                    # Dữ liệu song ngữ gốc (TED2020)
└── Train_spm/                    # Thư mục huấn luyện SentencePiece Tokenizer
    ├── Train_spm.ipynb           # Notebook huấn luyện tokenizer
    ├── ted2020_spm.model         # Model tokenizer
    └── ...
```

### `Train_spm/Train_spm.ipynb` (Chuẩn bị dữ liệu & Tokenizer)
Notebook này thực hiện các bước tiền xử lý và chuẩn bị tokenizer cho mô hình.

*   **Mục tiêu:** Làm sạch dữ liệu văn bản và huấn luyện mô hình SentencePiece Tokenizer.
*   **Các bước chính:**
    *   **Tiền xử lý (Preprocessing):**
        *   Chuyển văn bản về chữ thường.
        *   Loại bỏ các ký tự đặc biệt không cần thiết, chuẩn hóa khoảng trắng.
    *   **Tokenization:**
        *   Huấn luyện tokenizer riêng trên tập dữ liệu TED2020 sử dụng thư viện `sentencepiece`.
        *   Tạo ra bộ từ vựng (vocab) và mô hình tokenizer (`.model`).

### `gpt-mt.ipynb` (Training from Scratch)
Notebook này triển khai và huấn luyện mô hình GPT từ đầu (Non-pretrained).

*   **Mục tiêu:** Xây dựng và huấn luyện mô hình Transformer Decoder cho tác vụ dịch máy.
*   **Các bước chính:**
    *   **Load Tokenizer:** Sử dụng tokenizer đã tạo từ bước trước.
    *   **Định dạng dữ liệu:** Chuẩn bị cặp câu `Source: [English] Target: [Vietnamese]`.
    *   **Xây dựng Mô hình:**
        *   Kiến trúc Transformer Decoder (Embedding, Positional Encoding, Multi-head Self-Attention, Feed Forward Network).
    *   **Huấn luyện:**
        *   Sử dụng hàm mất mát **CrossEntropyLoss** và tối ưu hóa **AdamW**.
        *   Áp dụng **Teacher Forcing**.
    *   **Đánh giá:**
        *   Sử dụng **BLEU Score** để đánh giá độ chính xác.
        *   Inference bằng phương pháp Greedy Search.

### `gpt-mt-pre.ipynb` (Fine-tuning GPT-2)
Notebook này sử dụng mô hình GPT-2 đã được huấn luyện trước để tinh chỉnh cho tác vụ dịch.

*   **Mục tiêu:** Tận dụng tri thức từ mô hình Pretrained để cải thiện chất lượng dịch.
*   **Các bước chính:**
    *   **Load Pretrained Model:** Sử dụng `GPT2LMHeadModel` và `GPT2Tokenizer` từ thư viện `transformers`.
    *   **Fine-tuning:** Tinh chỉnh trọng số mô hình trên tập dữ liệu Anh-Việt.
    *   **Đánh giá:** Sử dụng **ROUGE Score** để đánh giá độ bao phủ.

## Yêu cầu cài đặt

Để chạy các notebook này, cần cài đặt các thư viện Python sau:

```bash
pip install torch transformers sentencepiece nltk numpy
```

## Link Kaggle

*   [Non-pretrain Version](https://www.kaggle.com/code/tneduvn/gpt-mt)
*   [Pretrain Version](https://www.kaggle.com/code/tneduvn/gpt-mt-pre)
