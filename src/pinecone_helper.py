from pinecone.grpc import PineconeGRPC as Pinecone  # ✅ gRPC-based Pinecone class
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from create_embeddings import download_hf_embeddings  # make sure this returns HuggingFaceEmbeddings
import os

# Load .env variables
load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
index_name = "tanishka"

# Init Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

def create_pinecone_index(index_name: str):
    """
    Create a Pinecone index with specific configuration.
    Only run once unless deleting and recreating.
    """
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    else:
        print(f"Index '{index_name}' already exists.")
    return pc.Index(index_name)

def get_pinecone_docsearch(index_name: str, embeddings):
    """
    Return PineconeVectorStore object (LangChain-compatible).
    """
    return PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )

def describe_pinecone_stats(index):
    """
    Describe the stats (vectors, dimensions, etc.) of an index.
    """
    return index.describe_index_stats()

def search_docs(docsearch, query: str):
    """
    Perform similarity search on vectorstore.
    """
    return docsearch.similarity_search(query)

# Optional test block
if __name__ == "__main__":
    embeddings = download_hf_embeddings()
    
    # Create index only once
    # create_pinecone_index(index_name)
    
    docsearch = get_pinecone_docsearch(index_name, embeddings)
    
    results = search_docs(docsearch, "What is acne?")
    
    for res in results:
        print(f"\n[Source Document]\n{res.page_content}")
