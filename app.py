import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

# 1. Google Gemini Imports
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

# 2. HuggingFace Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# 3. Vector Store
from langchain_community.vectorstores import FAISS

# 4. Prompt & Runnable
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document

# ==========================================
# CONFIGURATION
# ==========================================
load_dotenv()

# Safety: Clear proxies
for key in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
    if key in os.environ:
        del os.environ[key]

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ==========================================
# CORE FUNCTIONS
# ==========================================

def get_pdf_text(pdf_docs):
    """Extracts text from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_text_chunks(text):
    """Splits text into chunks so they fit in the model."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300
    )
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """Creates embeddings and saves the FAISS index locally."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device":"cpu"}
    )
    documents = [Document(page_content=chunk) for chunk in text_chunks]
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """RAG chain using RunnableLambda."""
    prompt_template = PromptTemplate(
        template="""
        Answer the question as detailed as possible from the provided context.
        If the answer is not in the provided context, say:
        "answer is not available in the context".

        Context:
        {context}

        Question:
        {question}

        Answer:
        """,
        input_variables=["context", "question"]
    )

    # Using the standard free model
    model = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        temperature=0.3,
    )

    def rag_logic(inputs):
        context = "\n\n".join(doc.page_content for doc in inputs["docs"])
        final_prompt = prompt_template.format(
            context=context,
            question=inputs["question"]
        )
        return model.invoke(final_prompt).content

    return RunnableLambda(rag_logic)

def get_response(user_question):
    """Helper function to get answer without printing directly."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device":"cpu"}
    )

    try:
        new_db = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
        docs = new_db.similarity_search(user_question, k=3)
        chain = get_conversational_chain()

        response = chain.invoke({
            "docs": docs,
            "question": user_question
        })
        return response
    except Exception as e:
        return f"Error: Please upload a PDF file first. ({e})"

# ==========================================
# MAIN APP UI
# ==========================================

def main():
    st.set_page_config(page_title="Gemini PDF Chat", page_icon="🤖")

    # 1. Sidebar for Uploads
    with st.sidebar:
        st.title("📂 Document Hub")
        st.write("Upload your PDFs to start chatting.")
        
        pdf_docs = st.file_uploader(
            "Select PDF Files", 
            accept_multiple_files=True, 
            type=["pdf"]
        )
        
        if st.button("Submit & Process", type="primary"):
            if pdf_docs:
                with st.spinner("Indexing documents..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done! You can now ask questions.")
            else:
                st.warning("Please upload a PDF first.")

    # 2. Main Chat Interface
    st.header("Chat with PDF using Gemini 🤖")

    # Initialize chat history if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 3. User Input & Response
    if prompt := st.chat_input("Ask a question about your PDF..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_response(prompt)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()



