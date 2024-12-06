import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline


# Initialize the LLaMA model
@st.cache_resource
def load_llama_model():
    llama_pipeline = pipeline("text-generation", model="facebook/opt-1.3b", device=0)
    return HuggingFacePipeline(pipeline=llama_pipeline)

# Function to process the uploaded PDF
def process_pdf(uploaded_file):
    print("pdf file", uploaded_file)
    
    temp_file_path = f"temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load the PDF using PyPDFLoader
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    split_docs = text_splitter.split_documents(documents)


    # Create embeddings and FAISS index
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(split_docs, embedding_model)

    return vector_store, split_docs

# RAG pipeline to query the vector store
def query_pipeline(vector_store, query, llm):
    # Retrieve relevant chunks
    retrieved_docs = vector_store.similarity_search(query, k=3)

    # Generate the response using LLaMA
    qa_chain = load_qa_chain(llm, chain_type="stuff")
    answer = qa_chain.run(input_documents=retrieved_docs, question=query)
    return answer

# Streamlit app
def main():
    st.title("RAG with PDF Summarization using LLaMA")
    st.write("Upload a PDF, ask a question, and get a detailed answer!")

    # Upload PDF
    pdf_file = st.file_uploader("Upload your PDF file", type=["pdf"])
    if pdf_file:
        # Process the PDF
        st.info("Processing the PDF...")
        vector_store, _ = process_pdf(pdf_file)
        query = st.text_input("Enter your question:")
        if query:
            # Load LLaMA model
            llm = load_llama_model()

            # Get the result
            with st.spinner("Generating response..."):
                result = query_pipeline(vector_store, query, llm)

            # Display the result
            st.success("Generated Summary:")
            st.write(result)

if __name__ == "__main__":
    main()
