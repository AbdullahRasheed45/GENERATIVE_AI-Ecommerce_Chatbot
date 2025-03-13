from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_astradb import AstraDBVectorStore
from dotenv import load_dotenv
import os
import pandas as pd
from ecommercebot.data_converter import dataconveter

# Load environment variables
load_dotenv()

# Fetch Astra DB variables from .env file
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")

# Use Hugging Face pre-trained model for embeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def ingestdata(status):
    # Initialize AstraDBVectorStore with Hugging Face Embeddings
    vstore = AstraDBVectorStore(
        embedding=embedding,
        collection_name="chatbotecomm",
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        token=ASTRA_DB_APPLICATION_TOKEN,
        namespace=ASTRA_DB_KEYSPACE,
    )
    
    if status is None:
        docs = dataconveter()  # Assuming dataconveter function converts your data properly
        inserted_ids = vstore.add_documents(docs)
    else:
        return vstore
    
    return vstore, inserted_ids

if __name__ == '__main__':
    # Perform data ingestion and document insertion
    vstore, inserted_ids = ingestdata(None)
    print(f"\nInserted {len(inserted_ids)} documents.")
    
    # Perform a similarity search
    results = vstore.similarity_search("can you tell me the low budget sound basshead.")
    
    # Print the results
    for res in results:
        print(f"* {res.page_content} [{res.metadata}]")
