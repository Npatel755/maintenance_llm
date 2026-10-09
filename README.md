# Local Maintenance Manual Assistant

A local RAG application that answers questions about maintenance
manuals using retrieved PDF passages and source citations.

## Features

- Extracts text from PDFs and splits it into overlapping chunks
- Creates document embeddings using Ollama
- Stores embeddings and source metadata in Chroma
- Retrieves relevant passages for each question
- Generates answers with numbered passage citations
- Displays answers and source passages through Streamlit

## Tech Stack

Python, PyMuPDF, Ollama, Chroma, and Streamlit.

Models:

- Embeddings: nomic-embed-text
- Answer generation: llama3.2:3b

## Setup

Run these commands from the project root.

Create and activate a virtual environment on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

Install and start the Ollama application, then download the models:

```powershell
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

## Index a Manual

Place a text-based PDF in the documents folder, then run:

```powershell
python -m app.index_manual "documents\your_manual.pdf"
```

The index is saved locally in chroma_db.

## Run the App

Keep Ollama running, then start Streamlit:

```powershell
python -m streamlit run app/streamlit_app.py
```

Ask a question and expand the retrieved passages to inspect
the evidence behind the answer.

## How It Works

During indexing, PDF text is split into overlapping chunks.
Each chunk is converted into an embedding and stored alongside
its text, filename, and PDF page number.

For each question, the app creates a query embedding and searches
for similar chunks. The retrieved passages are supplied to the
language model as context for generating a cited answer.

## Limitations

- Scanned PDFs require OCR before indexing.
- PDF tables may lose formatting during text extraction.
- Retrieval can miss relevant passages.
- Generated answers and citations can be incorrect.
- Reindexing a changed document does not automatically remove
  obsolete chunks from its previous version.
