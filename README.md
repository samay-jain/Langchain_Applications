# 🧠 AI-Powered Summarizers

This repository contains two independent Python applications powered by LLMs for intelligent content summarization:

1. 📰 **News Article Summarizer**
2. 📺 **YouTube Video Summarizer + Q&A**

Each tool uses OpenAI or Ollama-based local LLMs through LangChain to efficiently analyze, process, and summarize real-world content.

---

## 📰 News Article Summarizer

### 📖 Description

The **News Article Summarizer** allows users to input a URL of a news article and receive a clean, structured summary. It uses `newspaper3k` for content extraction and LangChain for summarization. Both OpenAI (cloud) and Ollama (local) LLMs are supported.

---

### 🔧 Features

- ✅ Extracts article title, text, authors, publish date using `newspaper3k`
- 🧠 Summarizes content using LangChain with:
  - OpenAI GPT models (`gpt-3.5`, `gpt-4`, `gpt-4o-mini`)
  - Ollama-based local models (e.g., `llama3.2`)
- 📋 Uses `MapReduce` summarization for better coherence with longer articles
- 🛠️ Custom prompt templates for fine-tuned output
- 🔗 Shows source URL and author metadata
- 🗃️ Saves extracted and summarized text to disk (optional)

---
---

## 📺 YouTube Video Summarizer + Q&A

### 📖 Description

The **YouTube Video Summarizer + Q&A** tool allows users to input a YouTube video URL, extract and transcribe audio using Whisper, summarize the transcript using LLMs, and optionally ask follow-up questions. It supports both OpenAI (cloud) and Ollama (local) LLMs for intelligent processing and Q&A.

---

### 🔧 Features

- 🎧 Extracts audio from YouTube videos using `yt_dlp`
- 📝 Transcribes speech using OpenAI Whisper (supports CLI or Python versions)
- 🧠 Summarizes transcripts using LangChain with:
  - OpenAI GPT models (`gpt-3.5`, `gpt-4`, `gpt-4o`)
  - Ollama-based local models (e.g., `llama3.2`, `mistral`)
- 📋 Uses `MapReduce` summarization for improved coherence on long transcripts
- ❓ Q&A mode allows users to ask follow-up questions about the video
- 🧠 Supports vector search via:
  - ChromaDB (local)
  - OpenAI embeddings (cloud)
  - Nomic embeddings (via Ollama)
- 🛠️ Custom prompt templates for summarization and question-answering
- 🗃️ Saves transcriptions, summaries, and answers to disk (optional)

---
### 🚀 How to Use

```bash
# 1. Clone the Repository
git clone https://github.com/samay-jain/Langchain_Applications

# 2. Install Dependencies
pip install -r requirements.txt

# 3. Run the Application

# 📰 For News Article Summarizer
python main.py

# 📺 For YouTube Video Summarizer + Q&A
python main.py
```

---

### ⚙️ Configuration (Optional)

```bash
# If using OpenAI, set your API key as an environment variable
OPENAI_API_KEY=your-openai-api-key

# If using Ollama, ensure the Ollama server is running
# and the required model (e.g., llama3.2) is available
ollama run llama3.2
```

