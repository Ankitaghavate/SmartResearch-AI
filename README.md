# 📘 Advanced Multi-Agent Research System

An AI-powered research platform that simulates a structured academic workflow using **Retrieval-Augmented Generation (RAG)**, **FAISS Vector Memory**, **Multi-Agent Architecture**, and **T5-based Summarization**.

---

## 🚀 Overview

This system goes beyond traditional chatbots by:

* Planning before searching
* Retrieving real-time web information
* Generating professional research reports
* Critiquing and refining outputs
* Verifying factual accuracy
* Creating executive summaries
* Exporting reports as PDFs
* Enabling interactive Q&A on generated reports

👉 Think of it as a **dynamic "Chat with PDF" system** that first creates the research document and then allows users to interact with it intelligently.

---

## 🧠 Key Features

* 🔍 Real-time web search using SerpAPI
* 🧩 Retrieval-Augmented Generation (RAG)
* 🗂️ FAISS Vector Database for semantic retrieval
* 🤖 Multi-Agent System

  * Planner Agent
  * Writer Agent
  * Critic Agent
  * Improver Agent
  * Verifier Agent
  * Summarizer Agent
* 📋 T5-based Executive Summary Generation
* 📥 Professional PDF Export
* 🎓 Quality Assessment & Confidence Scoring
* 💬 Interactive RAG-Powered Q&A Assistant
* ⚡ Parallel Search Processing
* 🎨 Streamlit-Based User Interface
* 📚 Semantic Memory using Sentence Transformers

---

## 🏗️ System Architecture

<p align="center">
 <img width="1024" height="1536" alt="ChatGPT Image Jun 12, 2026, 03_23_46 PM" src="https://github.com/user-attachments/assets/685468a6-4b58-4edc-8a61-7f57f15fe060" />

</p>

---

## 🤖 Agent Responsibilities

### 📌 Planner Agent

* Analyzes the research topic
* Generates optimized search queries
* Identifies important focus areas

### 🔍 Retrieval Layer

* Uses SerpAPI for real-time information retrieval
* Collects relevant data from multiple sources

### 💾 Memory Layer

* Splits retrieved content into chunks
* Creates embeddings using Sentence Transformers
* Stores vectors in FAISS for semantic search

### 📝 Writer Agent

* Generates structured research drafts
* Organizes content into logical sections

### 🔎 Critic Agent

* Reviews generated reports
* Detects missing information
* Suggests improvements

### ✨ Improver Agent

* Refines content quality
* Enhances clarity and completeness

### ✅ Verifier Agent

* Performs fact-checking
* Validates critical information
* Improves report reliability

### 📋 Summarizer Agent (T5)

* Generates concise executive summaries
* Extracts key insights
* Produces reader-friendly content

### 💬 Q&A Assistant

* Retrieves relevant context from FAISS
* Answers user questions using RAG
* Enables interactive report exploration

---

## 🧰 Tech Stack

### Programming & Frameworks

* Python
* Streamlit

### AI & Machine Learning

* Llama 3 8B Instruct (OpenRouter)
* T5 Transformer
* Sentence Transformers
* Retrieval-Augmented Generation (RAG)

### Search & Retrieval

* SerpAPI
* FAISS Vector Database

### Libraries

* NumPy
* Requests
* Python Dotenv

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone <your-repository-link>
cd project-folder
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate Environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / Mac**

```bash
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Configure Environment Variables

Create a `.env` file:

```env
OPENROUTER_API_KEY=your_api_key
SERPAPI_API_KEY=your_api_key
```

---

## ▶️ Run the Application

```bash
streamlit run main.py
```

---

## 🎯 Advantages

✔ Reduces hallucinations using RAG

✔ Uses real-time web information

✔ Semantic retrieval with FAISS

✔ Multi-agent quality improvement workflow

✔ Fact-checking through Verifier Agent

✔ Executive summaries with T5

✔ Professional PDF export

✔ Interactive report chatbot

✔ Modular and scalable architecture

---

## 🔮 Future Scope

* 📚 Integration with ArXiv and PubMed
* 📑 Automatic Citation Generation (APA / IEEE)
* 🔄 Hybrid Search (BM25 + Vector Search)
* 🎯 Re-ranking Models
* 🌐 Multi-Language Research Support
* ☁️ Cloud Deployment (AWS, Azure, GCP)
* 📊 Research Analytics Dashboard
* 🧠 Advanced Multi-Agent Collaboration

---

## ❤️ Author

**Ankita Ghavate**
