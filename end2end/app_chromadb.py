import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma

# Load environment variables
from dotenv import dotenv_values
env_values = dotenv_values(".env")
print(env_values)



# Initialize session state
if "vector" not in st.session_state:
    # Load documents from a web source
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs = st.session_state.loader.load()
    
    # Split documents into chunks
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    
    # Initialize ChromaDB vector store with OpenAI embeddings
    st.session_state.embeddings = OpenAIEmbeddings()
    st.session_state.vectors = Chroma.from_documents(
        st.session_state.final_documents, 
        embedding=st.session_state.embeddings,
        collection_name="demo_collection"
    )

st.title("OpenAI with ChromaDB Demo")

# Set up the LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

# Create the document chain and retrieval chain
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Input field for user prompt
user_prompt = st.text_input("Provide Your Prompt Here")

if user_prompt:
    # Measure response time
    start = time.process_time()
    response = retrieval_chain.invoke({"input": user_prompt})
    print("Response time:", time.process_time() - start)
    
    # Display the response
    st.write(response['answer'])
    
    # Show relevant document chunks
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
