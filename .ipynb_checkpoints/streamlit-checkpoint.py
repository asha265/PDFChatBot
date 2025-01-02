# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 11:17:53 2024

@author: Shriya
"""
# -*- coding: utf-8 -*-


import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import tempfile

# Streamlit UI
st.title("PDF-based Q&A System with LangChain")

# Upload PDF file
uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

if uploaded_file:
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Load PDF content
    st.write("Processing the uploaded PDF...")
    loader = PyPDFLoader(temp_file_path)  # Use the temporary file path
    pages = []
    for page in loader.load():
        pages.append(page)
    
    st.success(f"Loaded {len(pages)} pages from the PDF.")
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(pages)
    st.write(f"Text split into {len(chunks)} chunks.")

    # Create vector database
    st.write("Creating vector database...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
        collection_name="local-rag"
    )
    st.success("Vector database created successfully.")
    
    # Set up LLM and retrieval
    local_model = "llama3.2"  # or whichever model you prefer
    llm = ChatOllama(model=local_model)
    
    # Query prompt template
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate 2
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )
    
    # Set up retriever
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), 
        llm,
        prompt=QUERY_PROMPT
    )
    
    # RAG prompt template
    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Input question
    question = st.text_input("Ask a question based on the PDF:")
    if question:
        st.write("Retrieving answer...")
        answer = chain.invoke(question)
        st.markdown(answer)
