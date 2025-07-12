# allatonce.py
import os
import warnings
from dotenv import load_dotenv

from load_chunk import load_pdf, text_split
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
warnings.filterwarnings("ignore")

class MedicAI:
    def __init__(self, data_path):
        print("\U0001F4C4 Loading data...")
        self.data_path = data_path
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.extracted_data = load_pdf(self.data_path)
        self.documents = text_split(self.extracted_data)
        self.vectorstore = None

    def create_faiss_index(self):
        print("\U0001F50D Creating FAISS index...")
        self.vectorstore = FAISS.from_documents(
            documents=self.documents,
            embedding=self.embeddings
        )

    def get_docsearch(self):
        return self.vectorstore

    def setup_huggingface_llm(self, model="google/flan-t5-base"):
        print("\U0001F9E0 Setting up HuggingFace LLM...")
        return HuggingFaceEndpoint(
            repo_id=model,
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            temperature=0.6,
            max_new_tokens=256
        )

    def setup_rag_chain(self, llm, vectorstore):
        print("\U0001F517 Setting up RAG chain...")
        template = """
        You are Dr. Medic, a medical expert assistant.
        - Use only the context provided.
        - Respond clearly and helpfully.
        - Do not make up any information.

        Context: {context}
        Question: {question}
        Answer:
        """
        rag_prompt = PromptTemplate.from_template(template)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        return (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | rag_prompt
            | llm
            | StrOutputParser()
        )