import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

def load_pdf(data: str):
    documents = []
    for file in os.listdir(data):
        if file.endswith(".pdf"):
            file_path = os.path.join(data, file)
            reader = PdfReader(file_path)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            documents.append(Document(page_content=text, metadata={"source": file_path}))
    return documents

def text_split(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)
