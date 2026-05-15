<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2,12,24&height=200&section=header&text=📝%20Text%20Summarizer%20Using%20T5&fontSize=42&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Abstractive%20Dialogue%20Summarization%20with%20Fine-Tuned%20T5%20%2B%20FastAPI&descAlignY=60&descAlign=50" width="100%"/>

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-T5-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/ronakrajput8882/text-summarizer-t5)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-HF%20Spaces-FFD21E?style=for-the-badge)](https://ronakrajput8882-text-summarizer.hf.space/)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

</div>

---

## 📌 Project Overview

A production-ready **abstractive text summarization** web application powered by a fine-tuned **Google T5 (Text-to-Text Transfer Transformer)** model. The app accepts raw dialogue or paragraph-length text and generates concise, human-like summaries using beam search decoding.

The model is hosted on Hugging Face Hub (`ronakrajput8882/text-summarizer-t5`), served via a **FastAPI** REST backend, and deployed using **Docker on Render**.

> ✅ Supports CPU, CUDA (GPU), and Apple MPS — auto-detected at runtime.

<div align="center">

### 🌐 [Try the Live Demo → ronakrajput8882-text-summarizer.hf.space](https://ronakrajput8882-text-summarizer.hf.space/)

</div>

---

## 📂 Dataset & Model

| Property | Detail |
|:---|:---|
| **Base Model** | `google/t5-base` |
| **Fine-tuned Model** | `ronakrajput8882/text-summarizer-t5` |
| **Task** | Abstractive Dialogue Summarization |
| **Max Input Length** | 512 tokens |
| **Max Summary Length** | 150 tokens |
| **Decoding Strategy** | Beam Search (num_beams=4, early_stopping) |

---

## 🔄 Pipeline Workflow

```
Raw Text Input → Text Cleaning → Tokenization (T5Tokenizer) → T5 Inference → Beam Decode → Summary Output
```

**1️⃣ Text Cleaning**
- Strip HTML tags, normalize whitespace, remove carriage returns
- Convert to lowercase for consistent tokenization

**2️⃣ Tokenization**
- `T5Tokenizer` with `padding="max_length"`, `max_length=512`, `truncation=True`
- Returns PyTorch tensors mapped to the active device

**3️⃣ Model Inference**
- `T5ForConditionalGeneration.generate()` with `num_beams=4` and `early_stopping=True`
- Produces token IDs decoded back to human-readable text via `skip_special_tokens=True`

**4️⃣ API Serving**
- FastAPI POST endpoint `/summarize/` accepts JSON `{ "dialogue": "..." }`
- GET `/` serves the HTML frontend via Jinja2 templates
- GET `/health` for deployment health checks

---

## 🤖 Model — ⭐ Fine-Tuned T5

```python
model = T5ForConditionalGeneration.from_pretrained("ronakrajput8882/text-summarizer-t5")
tokenizer = T5Tokenizer.from_pretrained("ronakrajput8882/text-summarizer-t5")

targets = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=150,
    num_beams=4,
    early_stopping=True
)
```
- Fine-tuned T5-base on dialogue summarization task
- Encoder-decoder transformer architecture (Text-to-Text)
- Beam search ensures higher quality, coherent summaries vs greedy decoding
- Hosted publicly on Hugging Face Hub for easy loading

---

## 📊 Results

| Metric | Value |
|:---|:---:|
| **Max Input Tokens** | 512 |
| **Max Output Tokens** | 150 |
| **Beam Width** | 4 |
| **Decoding** | Beam Search + Early Stopping |
| **Device Support** | CPU / CUDA / Apple MPS |

---

## 🔍 Key Insights

- 📈 **Beam search (num_beams=4)** consistently produces more fluent summaries than greedy decoding, especially for multi-sentence dialogues
- 🧹 **Text cleaning** (HTML stripping + whitespace normalization) significantly reduces tokenizer noise and improves output coherence
- 🚀 **T5's text-to-text format** makes the model architecture uniform — summarization, translation, and QA all use identical `input → output` framing
- 🐳 **Docker + Render deployment** with `render.yaml` enables zero-config CI/CD — push code, get deployment
- ⚡ Auto device detection (MPS → CUDA → CPU) makes the app run efficiently across MacBooks, cloud GPUs, and free-tier servers

---

## 🗂️ Repository Structure

```
Text-Summarizer-Using-T5/
│
├── app.py              # FastAPI application — routes, model loading, inference
├── index.html          # Frontend UI served via Jinja2 templates
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker container configuration
├── render.yaml         # Render deployment config
├── .gitignore
└── LICENSE
```

---

## 🚀 Quick Start

### Local Setup

```bash
# Clone the repository
git clone https://github.com/ronakrajput8882/Text-Summarizer-Using-T5.git
cd Text-Summarizer-Using-T5

# Install dependencies
pip install -r requirements.txt

# Run the FastAPI server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Docker

```bash
docker build -t text-summarizer .
docker run -p 8000:8000 text-summarizer
```

### API Usage

```bash
curl -X POST "http://localhost:8000/summarize/" \
  -H "Content-Type: application/json" \
  -d '{"dialogue": "Person A: Did you finish the report? Person B: Not yet, I need two more hours. Person A: Okay, I will inform the team."}'
```

**Response:**
```json
{
  "summary": "person b needs two more hours to finish the report. person a will inform the team."
}
```

Visit `http://localhost:8000` for the web UI.

---

## 🧠 Key Learnings

- Fine-tuning T5 via Hugging Face `Trainer` API and hosting the checkpoint on HF Hub for direct `from_pretrained()` loading
- Building production FastAPI apps with Pydantic schema validation and Jinja2 HTML rendering
- Dockerizing ML inference services and deploying to **Hugging Face Spaces** for free live demo hosting
- Handling multi-device inference (MPS/CUDA/CPU) with graceful fallback logic
- Text preprocessing pipelines for transformer input normalization

---

## 🛠️ Tech Stack

| Tool | Use |
|:---|:---|
| `transformers` | T5ForConditionalGeneration, T5Tokenizer |
| `torch` | Model inference, device management |
| `fastapi` | REST API backend |
| `uvicorn` | ASGI server |
| `pydantic` | Request schema validation |
| `Jinja2` | HTML template rendering |
| `Docker` | Containerized deployment |
| `Hugging Face Spaces` | Live demo deployment |
| `Hugging Face Hub` | Model hosting |

---

<div align="center">

### Connect with me

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/ronaksinh-rajput8882)
[![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://instagram.com/techwithronak)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ronakrajput8882)

*If you found this useful, please ⭐ the repo!*

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2,12,24&height=100&section=footer" width="100%"/>

</div>
