# GPT Machine Translate Project

Dự án này thực hiện dịch máy (Machine Translation) Anh-Việt sử dụng kiến trúc GPT (Generative Pre-trained Transformer). Dự án bao gồm cả phương pháp huấn luyện từ đầu (training from scratch) và tinh chỉnh mô hình đã huấn luyện trước (fine-tuning pretrained model).

## Cấu trúc thư mục

- **`Train_spm/`**: Thư mục chứa mã nguồn và dữ liệu liên quan đến việc huấn luyện SentencePiece Tokenizer.
  - `Train_spm.ipynb`: Notebook thực hiện tiền xử lý dữ liệu (làm sạch văn bản) và huấn luyện mô hình SentencePiece.
  - `ted2020_spm.model`, `ted2020_spm.vocab`: Mô hình tokenizer và bộ từ vựng đã được huấn luyện.
  - `cleaned.en`, `cleaned.vi`: Dữ liệu văn bản tiếng Anh và tiếng Việt sau khi đã được làm sạch.
  - `gpt_data.txt`, `tokenized_gpt.txt`: Các file dữ liệu trung gian cho quá trình huấn luyện.

- **`gpt-mt.ipynb`**: Notebook triển khai và huấn luyện mô hình GPT cho dịch máy từ đầu (Non-pretrained). Sử dụng tokenizer SentencePiece đã tạo ở bước trước.

- **`gpt-mt-pre.ipynb`**: Notebook sử dụng mô hình GPT-2 đã được huấn luyện trước (`GPT2LMHeadModel`) và tokenizer của nó (`GPT2Tokenizer`) để tinh chỉnh (fine-tune) cho tác vụ dịch máy.

- **`en-vi.txt/`**: Thư mục chứa dữ liệu song ngữ gốc (TED2020).

- **`Link_GPT_MachineTranslate.txt`**: File chứa liên kết đến các notebook gốc trên Kaggle.

## Quy trình Modeling (Modeling Flow)

Dự án tuân theo quy trình modeling chuẩn trong NLP cho bài toán dịch máy:

### 1. Tiền xử lý dữ liệu (Data Preprocessing)
- **Làm sạch (Cleaning)**: Chuyển văn bản về chữ thường, loại bỏ các ký tự đặc biệt không cần thiết, chuẩn hóa khoảng trắng.
- **Định dạng (Formatting)**:
  - Dữ liệu được định dạng theo cặp câu: `Source: [English sentence] Target: [Vietnamese sentence]`.
  - Đối với mô hình GPT (Causal Language Modeling), mô hình sẽ học cách dự đoán token tiếp theo dựa trên chuỗi token trước đó.

### 2. Tokenization
- **Non-pretrained**:
  - Sử dụng thư viện `sentencepiece` để huấn luyện một tokenizer riêng trên tập dữ liệu TED2020.
  - Tạo ra bộ từ vựng (vocab) và mô hình tokenizer (`.model`) phù hợp với đặc thù dữ liệu Anh-Việt.
- **Pretrained**:
  - Sử dụng `GPT2Tokenizer` có sẵn của mô hình GPT-2.
  - Tận dụng khả năng hiểu ngôn ngữ đã học được từ trước.

### 3. Xây dựng Mô hình (Model Architecture)
- **Non-pretrained (Training from Scratch)**:
  - Xây dựng kiến trúc Transformer Decoder từ đầu.
  - Bao gồm các lớp: Embedding, Positional Encoding, Multi-head Self-Attention, Feed Forward Network.
- **Pretrained (Fine-tuning)**:
  - Sử dụng `GPT2LMHeadModel` từ thư viện `transformers`.
  - Tinh chỉnh (Fine-tune) trọng số của mô hình trên tập dữ liệu dịch máy mới.

### 4. Huấn luyện (Training)
- Sử dụng hàm mất mát **CrossEntropyLoss**.
- Tối ưu hóa bằng thuật toán **AdamW**.
- Sử dụng **Teacher Forcing** (trong quá trình training, đầu vào là chuỗi đích thực tế thay vì dự đoán của mô hình).

### 5. Đánh giá (Evaluation)
- **Inference**: Sử dụng phương pháp Greedy Search (chọn token có xác suất cao nhất) để sinh câu dịch.
- **Metrics**:
  - **BLEU Score**: Đánh giá độ chính xác của bản dịch dựa trên n-grams trùng khớp với câu tham chiếu.
  - **ROUGE Score**: Đánh giá độ bao phủ (recall) của bản dịch (sử dụng trong notebook fine-tuning).

## Hướng dẫn sử dụng

1. **Chuẩn bị dữ liệu và Tokenizer**:
   - Mở và chạy notebook `Train_spm/Train_spm.ipynb`.
   - Quá trình này sẽ đọc dữ liệu gốc, làm sạch văn bản và huấn luyện SentencePiece tokenizer. Kết quả sẽ tạo ra các file trong thư mục `Train_spm/`.

2. **Huấn luyện mô hình**:
   - **Cách 1: Huấn luyện từ đầu (Non-pretrained)**
     - Notebook này sẽ load tokenizer từ `Train_spm/` và huấn luyện mô hình GPT trên dữ liệu đã chuẩn bị.
   
   - **Cách 2: Fine-tune Pretrained Model**
     - Notebook này sử dụng thư viện `transformers` để tải mô hình GPT-2 và fine-tune trên tập dữ liệu Anh-Việt.

## Link Kaggle

- Link Kaggle Non-pretrain: [https://www.kaggle.com/code/tneduvn/gpt-mt](https://www.kaggle.com/code/tneduvn/gpt-mt)
- Link Kaggle Pretrain: [https://www.kaggle.com/code/tneduvn/gpt-mt-pre](https://www.kaggle.com/code/tneduvn/gpt-mt-pre)
