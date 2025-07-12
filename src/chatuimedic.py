# chatuimedic.py
import streamlit as st
from allatonce import MedicAI
import os
from dotenv import load_dotenv

# Session states
if 'medic_ai' not in st.session_state:
    st.session_state.medic_ai = None
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def initialize_rag_chain():
    if st.session_state.rag_chain is None:
        with st.spinner("\U0001F504 Initializing Medic AI..."):
            print("\U0001F680 Loading environment...")
            load_dotenv()
            hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
            if not hf_token:
                st.error("\u274C HuggingFace token missing in .env")
                return
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

            print("\U0001F4E5 Creating MedicAI object...")
            medic_ai = MedicAI("C:/Users/nishz/Downloads/medic.ai-main/Data/")
            medic_ai.create_faiss_index()

            print("\U0001F4DA Loading Vector DB...")
            vectorstore = medic_ai.get_docsearch()

            print("\U0001F916 Loading LLM...")
            llm = medic_ai.setup_huggingface_llm()

            print("\U0001F517 Creating RAG Chain...")
            rag_chain = medic_ai.setup_rag_chain(llm, vectorstore)

            st.session_state.medic_ai = medic_ai
            st.session_state.rag_chain = rag_chain
            print("\u2705 Medic AI Ready")

def display_chat():
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else None):
            st.write(message["content"])

def get_user_input():
    user_input = st.chat_input("Ask your medical question here...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
    return user_input

def main():
    st.set_page_config(page_title="Medic AI 🤖💺", layout="wide")
    st.title("Medic AI 🤖💺 - Your Local Medical Assistant")

    initialize_rag_chain()
    display_chat()

    user_input = get_user_input()
    if user_input:
        with st.chat_message("assistant", avatar="🤖"):
            try:
                response = st.session_state.rag_chain.invoke(user_input)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.write(response)
            except Exception as e:
                st.error(f"\u274C Error: {str(e)}")

if __name__ == "__main__":
    main()