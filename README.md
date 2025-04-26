

readme_content = """
# AI Text Summarization using Google PEGASUS transformer model and Hugging Face Samsum Dataset

## 📚 Project Overview
This project develops an **abstractive text summarization pipeline** using the **PEGASUS** (`google/pegasus-cnn_dailymail`) model from Hugging Face.  
It focuses on summarizing dialogues from the **SAMSum** dataset, demonstrating the capabilities of **Generative AI** in producing human-like summaries.

---

## 🛠️ Technologies Used
- **Model**: PEGASUS (fine-tuned on CNN/DailyMail)
- **Dataset**: SAMSum (Test Split – 819 dialogues)
- **Libraries**:
  - Hugging Face `transformers`
  - Hugging Face `datasets`
  - `evaluate` (for ROUGE score computation)
  - `nltk`
  - `tqdm`
  - `PyTorch`

---

## ⚙️ Workflow

### 1. Dataset Loading
- Loaded the SAMSum dataset using Hugging Face `datasets` library.

### 2. Model Loading
- Loaded the pre-trained **PEGASUS** model and tokenizer.

### 3. Summarization
- Efficiently tokenized input dialogues.
- Used **batch generation** for faster inference.
- Configured **beam search** (`num_beams=8`) with a length penalty to generate better summaries.

### 4. Evaluation
- Computed **ROUGE-1**, **ROUGE-2**, **ROUGE-L**, and **ROUGE-Lsum** scores.
- Used the `evaluate` library for metric computation.

### 5. Optimization
- Leveraged **GPU acceleration** using CUDA.
- Implemented **progressive batching** to reduce inference time by ~2×.

---

## 📈 Results
- Achieved **high ROUGE scores** reflecting strong summarization performance.
- **Significantly reduced inference time** with batching and device optimization.
- Generated **fluent**, **coherent**, and **human-like summaries**.

---

## 🔥 Key Highlights
- Worked with a **state-of-the-art transformer model (PEGASUS)**.
- Handled a large model (~2.28 GB) efficiently on limited resources (Google Colab).
- Built an **end-to-end text summarization and evaluation pipeline**.
- Followed **industry best practices** for batching, evaluation, and GPU optimization.

---

## 🧩 Future Work
- Fine-tune PEGASUS specifically on the **SAMSum dataset** for domain adaptation.
- Deploy the model using an interactive web app (e.g., **Streamlit**, **FastAPI**).
- Explore additional evaluation metrics like **BLEU** and **METEOR**.

---

## 📍 How to Run

### Install Required Libraries
```bash
pip install transformers[sentencepiece] datasets evaluate sacrebleu rouge_score py7zr tqdm nltk
