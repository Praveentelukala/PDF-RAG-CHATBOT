# 🤖 Gemini PDF Chatbot (RAG)

🚀 Live Demo: https://pdf-rag-chatbot-nzxseh9i4dec2jwtwlh9aj.streamlit.app/


A powerful Python application that allows you to chat with multiple PDF documents using Google's **Gemini AI** and **LangChain**. 

This project uses **RAG (Retrieval-Augmented Generation)** to index your PDF content, allowing the AI to answer specific questions based *only* on the documents you provide. It features a persistent chat history and a modern, user-friendly interface.

---

## 🚀 Features

* **Multi-PDF Support:** Upload multiple documents at once.
* **Free & Fast Embeddings:** Uses HuggingFace (`all-MiniLM-L6-v2`) running locally on CPU.
* **Vector Search:** Efficient similarity search using **FAISS**.
* **Google Gemini Powered:** Uses `gemini-1.5-flash` for high-speed, accurate, and free-tier friendly responses.
* **Chat History:** Remembers your conversation context within the session.
* **Secure:** API keys are managed safely via `.env` files.

---

## 🛠️ Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/) (for the UI)
* **LLM:** Google Gemini 1.5 Flash (via `google-generativeai`)
* **Framework:** [LangChain](https://www.langchain.com/) (for chaining logic)
* **Embeddings:** HuggingFace `sentence-transformers` (Local & Free)
* **Vector Database:** FAISS (Facebook AI Similarity Search

---


🌐 Deployment

This application is deployed using Streamlit Cloud.

👉 Access it here:
https://pdf-rag-chatbot-nzxseh9i4dec2jwtwlh9aj.streamlit.app/


---

## 📂 Project Structure
* ├── **app.py:**                   Main application code
* ├── **requirements.txt:**         List of Python dependencies
* ├── **.env:**                     API Key (Do NOT upload to GitHub)
* ├── **.gitignore:**               Files to exclude from Git
* ├── **faiss_index/:**             Local vector storage (generated at runtime)
* └── **README.md:**                Project documentation




