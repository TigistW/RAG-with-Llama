# RAG with PDF Summarization using LLaMA

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline using **LLaMA** and **Streamlit** for interactive summarization and question answering based on uploaded PDF documents. The app extracts text from a PDF, processes it into chunks, creates embeddings, and retrieves relevant information to answer user queries.

---

## Features

* **Upload PDF** : Users can upload a PDF file for processing.
* **Question-Answering** : Ask a question related to the content of the PDF.
* **Text Generation** : Generates detailed answers using the LLaMA language model.
* **Interactive Interface** : Built using Streamlit for a simple, user-friendly experience.

---

## Installation

### Prerequisites

* Python 3.8 or later
* `pip` (Python package manager)

### Install Required Libraries

Run the following command to install all dependencies:

```bash
pip install streamlit langchain sentence-transformers faiss-cpu transformers
```

---

## How It Works

1. **Upload PDF** : The app accepts a PDF file.
2. **Text Extraction** : Text is extracted from the PDF using `PyPDFLoader`.
3. **Chunking** : The text is split into smaller chunks using `RecursiveCharacterTextSplitter`.
4. **Embeddings** : Each chunk is converted into an embedding using `SentenceTransformerEmbeddings`.
5. **Retrieval** : Relevant chunks are retrieved using FAISS for a user-provided query.
6. **Generation** : The LLaMA model generates a detailed answer based on the retrieved chunks.

---

## Usage

### Running the App

1. Save the code in a file named `app.py`.
2. Run the app using Streamlit:
   ```bash
   streamlit run app.py
   ```

### Interacting with the App

1. Upload a PDF document by clicking "Upload your PDF file."
2. Enter a question in the text input box.
3. View the generated summary or answer in the output section.

---

## Key Components

### **1. LLaMA Model**

The LLaMA model is initialized using Hugging Face's `transformers` library:

```python
def load_llama_model():
    llama_pipeline = pipeline("text-generation", model="facebook/opt-1.3b", device=0)
    return HuggingFacePipeline(pipeline=llama_pipeline)
```

### **2. PDF Processing**

The uploaded PDF is processed to extract and split text into manageable chunks:

```python
def process_pdf(uploaded_file):
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(split_docs, embedding_model)
    return vector_store, split_docs
```

### **3. Query Pipeline**

Handles retrieval of relevant information and answer generation:

```python
def query_pipeline(vector_store, query, llm):
    retrieved_docs = vector_store.similarity_search(query, k=3)
    qa_chain = load_qa_chain(llm, chain_type="stuff")
    answer = qa_chain.run(input_documents=retrieved_docs, question=query)
    return answer
```

---

## Dependencies

* **Streamlit** : For building the interactive web app.
* **LangChain** : For document loading, chunking, embedding, and retrieval.
* **SentenceTransformers** : For embedding text chunks.
* **FAISS** : For similarity search.
* **Transformers** : For LLaMA-based text generation.

---

## Notes

1. Ensure that the `facebook/opt-1.3b` model is accessible and supported on your hardware. If the model is gated, ensure you have authenticated via Hugging Face's CLI.
2. This code is designed for demonstration purposes and may need further optimization for large-scale use.
