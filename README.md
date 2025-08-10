# Adobe India Hackathon – Document Intelligence RAG Pipeline

## 📌 Overview
This project is a **Retrieval-Augmented Generation (RAG)** pipeline built for the Adobe India Hackathon.  
It processes PDF documents, retrieves relevant sections based on a specific job description or persona, and generates **structured JSON summaries** ranked by importance.

It uses:
- **Ollama** for running a custom fine-tuned LLM (`hackathon`)
- **LangChain** for prompt handling, document processing, and orchestration
- **HuggingFace Embeddings** (`all-MiniLM-L6-v2`) + **FAISS** for semantic search
- **Docker** for containerized deployment

---

## 🚀 Features
- 📂 **Multi-PDF ingestion** – processes all PDFs from `/app/input`
- 🧠 **Custom local LLM** (`hackathon`) created from a `Modelfile` + `.gguf` weights
- 🔍 **Semantic search** with FAISS vector store
- 📊 **Importance ranking** for each extracted section
- 📝 Outputs **structured JSON** with metadata, source document, page number, and ranked summaries
- 🐳 Fully **Dockerized** for easy deployment

---

## 🛠️ Tech Stack
- **Python 3.11**
- **Ollama** (Custom model creation & inference)
- **LangChain**
- **HuggingFace Embeddings**
- **FAISS** (vector database)
- **PyPDFLoader** for PDF parsing
- **Docker (multi-stage build)**

---

## 📂 Project Structure
```
├── final1.py              # Main RAG processing script
├── requirements.txt       # Python dependencies
├── Dockerfile             # Multi-stage Docker build (Python + Ollama)
├── Modelfile              # Ollama model definition
├── model.gguf             # LLM weights (DeepSeek-R1 Distill Qwen 1.5B)
├── /input                  # Place PDFs and config.json here
├── /output                 # Processed JSON output
```

---

## ⚙️ Installation & Usage

### 1️⃣ Clone the repository
```bash
git clone https://github.com/rajatwhosebrowsing/adobeAdobe_India_Hackathon1.git
cd adobeAdobe_India_Hackathon1
```

### 2️⃣ Prepare input
Inside the `/input` folder, place:
- One or more `.pdf` files
- A `config.json` file in the format:
```json
{
  "persona": "Senior Data Analyst",
  "job_to_be_done": "Identify key financial metrics and risk indicators"
}
```

### 3️⃣ Build Docker image
```bash
docker build -t hackathon-rag .
```

### 4️⃣ Run the container
```bash
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output hackathon-rag
```

The processed `result.json` will be in `/output`.

---

## 🖥️ How It Works
1. **Load PDFs & Config** → Reads persona & job target from `config.json`
2. **Split Text** → Splits documents into overlapping chunks
3. **Embed & Store** → Creates FAISS vector store with HuggingFace embeddings
4. **Retrieve** → Gets top-k chunks relevant to the job
5. **Analyze** → Uses `hackathon` LLM to output structured JSON with:
   - `section_title`
   - `importance_rank`
   - `refined_text` summary
6. **Save** → Writes output to `/output/result.json`

---

## 📦 Dependencies
Install locally (for development without Docker):
```bash
pip install -r requirements.txt
```

---

## 📝 Example Output
```json
{
  "metadata": {
    "input_documents": ["report1.pdf"],
    "persona": "Senior Data Analyst",
    "job_to_be_done": "Identify key financial metrics and risk indicators",
    "processing_timestamp": "2025-08-10T14:00:00"
  },
  "Extracted Section": [
    {
      "section_title": "Q2 Financial Overview",
      "importance_rank": 1,
      "refined_text": "Summarized analysis here...",
      "document": "report1.pdf",
      "page_number": 5
    }
  ]
}
```

---

## 🧪 Testing
Run locally:
```bash
python3 final1.py
```

Make sure you have:
- `/input` folder with PDFs and `config.json`
- Ollama running locally with `hackathon` model loaded

---

## 📜 License
MIT License © 2025 Rajat
