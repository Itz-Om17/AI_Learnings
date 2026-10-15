# 📚 AI PDF Quiz Generator (RAG-Powered)

An intelligent, full-stack application that transforms any PDF document (textbooks, research papers, notes, study guides) into interactive multiple-choice quizzes using **Retrieval-Augmented Generation (RAG)** and local LLMs via **Ollama**.

---

## 🌟 What Does This Project Do?

Instead of reading through dozens or hundreds of pages and manually creating questions, you simply:
1. **Upload a PDF document**.
2. **Choose your settings**:
   - Difficulty level (*Easy*, *Medium*, *Hard*)
   - Number of questions (e.g., 5, 10, 15)
   - Scope (*Entire PDF* or a *Custom Page Range*)
3. **Generate Quiz**: The AI analyzes the document, generates questions with 4 options and clear explanations, and presents an interactive test right in your browser.
4. **Take the Test**: Select your answers, submit the quiz, get instant scores, and review detailed explanations for each question.

---

## 🧠 How It Uses RAG (Explained in Simple Words)

### What is RAG?
**RAG** stands for **Retrieval-Augmented Generation**. 

Normally, if you ask an AI model like LLaMA to create a quiz about a 50-page document:
- The entire document might be **too large** to fit into the AI's prompt memory (context window).
- The model might **hallucinate** (make up facts that aren't actually in your notes).

**RAG solves this in two steps:**
1. **Retrieval**: Search the document for the most relevant sections related to the request.
2. **Augmentation & Generation**: Give *only* those relevant sections to the AI and say: *"Using strictly this text, generate high-quality quiz questions."*

---

### The 5-Step RAG Pipeline in This Project

```
  ┌──────────────┐
  │  Uploaded    │
  │     PDF      │
  └──────┬───────┘
         │
         ▼
  [Step 1: Text Extraction]  ──► PyMuPDF extracts text from full PDF or selected pages
         │
         ▼
  [Step 2: Smart Chunking]   ──► Splits text into 800-character overlapping chunks
         │
         ▼
  [Step 3: Vector Store]     ──► Ollama (nomic-embed-text) converts chunks into vectors 
         │                       and saves them in ChromaDB
         ▼
  [Step 4: Semantic Search]  ──► Finds top 6 most relevant text passages for quiz context
         │
         ▼
  [Step 5: Quiz Generation]  ──► Ollama (Llama 3) creates structured MCQs in JSON format
         │
         ▼
  ┌──────────────┐
  │ Interactive  │
  │  React Quiz  │
  └──────────────┘
```

#### 1. Text Extraction (`pdf_utils.py`)
- Uses **PyMuPDF (`fitz`)** to extract raw text from the uploaded PDF.
- Supports extracting either the entire document or a specific page range (e.g., pages 5 to 12).

#### 2. Text Chunking (`chunking.py`)
- Long text cannot be converted into meaningful embeddings as a single block.
- Using LangChain's **`RecursiveCharacterTextSplitter`**, the document is cut into smaller pieces of **800 characters** each, with a **100-character overlap**.
- The overlap ensures concepts that cross boundaries aren't accidentally cut in half.

#### 3. Vector Embeddings & Storage (`vector_store.py`)
- Each text chunk is converted into a numerical vector (embedding) using **`nomic-embed-text`** via Ollama.
- These vectors capture the *meaning* of the text, not just exact words.
- They are stored locally in **ChromaDB**, isolated in a dedicated folder per PDF.

#### 4. Semantic Retrieval (`main.py`)
- When a quiz is requested, the system performs a similarity search in ChromaDB.
- It pulls the top **6 most relevant chunks** (`k=6`) to form the focused context for the AI.

#### 5. Grounded Quiz Generation (`quiz_generator.py`)
- The retrieved context is passed to **`llama3`** (running locally via Ollama) with a strict prompt:
  - Generate exactly the requested number of questions.
  - Rely **only** on the provided academic context (preventing hallucinations).
  - Return clean, structured JSON with questions, options (A, B, C, D), correct answers, and explanations.
- Includes an automatic retry loop if the output needs completion or formatting fixes.

---

## 🏗️ Architecture & Tech Stack

This project is organized into 3 decoupled layers:

```
Quiz_Generator/
├── ai-service/     # Python AI Microservice (RAG & LLM Engine)
├── backend/        # Node.js Express API Gateway (Uploads & Routing)
└── frontend/       # React + Vite + Tailwind CSS User Interface
```

### 1. AI Service (`/ai-service`)
- **FastAPI**: Lightweight, high-speed Python REST API.
- **LangChain**: Orchestrates chunking, embeddings, and LLM prompts.
- **ChromaDB**: Local, embedded vector database.
- **Ollama**:
  - `nomic-embed-text`: High-speed, high-quality text embedding model.
  - `llama3`: Large Language Model for reasoning and MCQ generation.
- **PyMuPDF (`fitz`)**: Fast PDF text parser.

### 2. Backend Gateway (`/backend`)
- **Node.js & Express**: Handles API routing and CORS.
- **Multer**: Manages multipart PDF uploads to the `uploads/` folder.
- **Axios**: Communicates seamlessly with the Python AI service.

### 3. Frontend (`/frontend`)
- **React (Vite)**: Fast, modern UI.
- **Tailwind CSS**: Glassmorphism dark-mode UI design.
- Features: File selector, difficulty & page-range controls, live quiz taking, answer verification, and score reporting.

---

## 📂 Project Directory Structure

```text
Quiz_Generator/
├── ai-service/
│   ├── chunking.py          # Splits extracted text into chunks with overlap
│   ├── main.py              # FastAPI server handling /generate-quiz
│   ├── pdf_utils.py         # PyMuPDF extractor (supports page ranges)
│   ├── quiz_generator.py    # Prompt template and Ollama LLaMA 3 invocation
│   ├── quiz_parser.py       # JSON parsing helpers
│   ├── requirements.txt     # Python dependencies
│   └── vector_store.py      # ChromaDB vector store creation & embeddings
│
├── backend/
│   ├── package.json         # Node.js dependencies (express, multer, axios, cors)
│   ├── server.js            # Express server entry point (Port 5000)
│   ├── routes/
│   │   └── quizRoutes.js    # /api/quiz/generate endpoint handling uploads
│   └── uploads/             # Temp storage for uploaded PDFs
│
└── frontend/
    ├── package.json         # React & Tailwind dependencies
    ├── index.html           # HTML entry point
    ├── vite.config.js       # Vite build config
    └── src/
        ├── App.jsx          # Main quiz generator & quiz taking interface
        ├── main.jsx         # React DOM mount point
        └── services/
            └── api.js       # Axios client for backend API
```

---

## 🚀 Getting Started

### Prerequisites

1. **Node.js** (v18+)
2. **Python** (v3.10+)
3. **Ollama**: [Download & Install Ollama](https://ollama.com/)

Pull the required AI models in your terminal:
```bash
ollama pull llama3
ollama pull nomic-embed-text
```

---

### Step-by-Step Setup

#### 1. Start the AI Service (Python)

Navigate to the `ai-service` directory:
```bash
cd ai-service
```

Create and activate a virtual environment (optional but recommended):
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

Install the dependencies:
```bash
pip install fastapi uvicorn pydantic PyMuPDF langchain langchain-community langchain-text-splitters langchain-ollama chromadb
```

Start the FastAPI server:
```bash
uvicorn main:app --reload --port 8000
```
> The AI service will run at `http://localhost:8000`.

---

#### 2. Start the Backend Gateway (Node.js)

Open a new terminal and navigate to `backend`:
```bash
cd backend
npm install
npm start
```
> The Express server will run at `http://localhost:5000`.

---

#### 3. Start the Frontend (React + Vite)

Open a third terminal and navigate to `frontend`:
```bash
cd frontend
npm install
npm run dev
```
> The web app will be available at `http://localhost:5173`.

---

## 🎯 How to Use the App

1. Open `http://localhost:5173` in your browser.
2. Click **Choose File** and upload any `.pdf` document.
3. Select the **Difficulty**: `Easy`, `Medium`, or `Hard`.
4. Enter the **Number of Questions** you want.
5. Choose **Page Mode**:
   - `Full PDF`: Indexes the entire document.
   - `Custom Pages`: Specify Start and End page numbers (e.g., from page 1 to 5).
6. Click **Generate Quiz** and wait a few seconds.
7. Solve the multiple-choice questions!
8. Click **Submit Quiz** to view your score, check correct answers, and read explanations.

---

## 🔒 Key Benefits of This Setup

- **100% Local & Private**: All embeddings and LLM generations run locally through Ollama. Your documents never leave your computer.
- **Accurate & Grounded**: By using RAG, questions are generated directly from the document's content, significantly reducing hallucinations.
- **Fast & Scalable**: Decoupled architecture allows the AI service, backend, and frontend to scale or be updated independently.
