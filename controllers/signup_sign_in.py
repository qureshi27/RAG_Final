import os
import bcrypt
import chromadb
from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    Settings
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader  # <-- add this import


# Initialize ChromaDB client

chroma_client = chromadb.PersistentClient(path="chroma_db")
users_collection = chroma_client.get_or_create_collection("users")




UPLOAD_FOLDER = "./uploaded_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
EMBEDDINGS_DIR = os.path.join(UPLOAD_FOLDER, "embeddings")
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)




def signup_user(email: str, password: str):
    # Check if user exists
    results = users_collection.get(where={"email": email})
    if results["ids"]:
        raise Exception("User already exists.")
    # Hash the password
    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    # Save user
    users_collection.add(
        documents=[email],
        metadatas=[{"email": email, "password": hashed_pw.decode('utf-8')}],
        ids=[email]
    )
    return True

def signin_user(email: str, password: str):
    results = users_collection.get(where={"email": email})
    if not results["ids"]:
        raise Exception("User not found.")
    user_meta = results["metadatas"][0]
    hashed_pw = user_meta["password"].encode('utf-8')
    if not bcrypt.checkpw(password.encode('utf-8'), hashed_pw):
        raise Exception("Incorrect password.")
    return True






from llama_index.readers.file import PDFReader
from llama_index.core import SimpleDirectoryReader

def handle_file_embedding_by_email_and_filename(email: str, file_name: str, chroma_client):
    file_path = os.path.join(UPLOAD_FOLDER, file_name)
    # embeddings_dir is ALWAYS EMBEDDINGS_DIR, not user-specific!
    Settings.llm = OpenAI(model="gpt-4o-mini")
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    documents = []
    if file_path.lower().endswith('.pdf'):
        try:
            reader = PDFReader()
            documents = reader.load_data(file_path)
        except Exception as e:
            raise Exception(f"PDF extraction error: {str(e)}")
    else:
        try:
            documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        except Exception as e:
            raise Exception(f"Text file extraction error: {str(e)}")

    if not documents:
        raise Exception("No text could be extracted from the file.")

    for doc in documents:
        if "source" not in doc.metadata or not doc.metadata["source"]:
            doc.metadata["source"] = file_name

    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=EMBEDDINGS_DIR)

    embedding_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # If you still want per-user collections in Chroma, use this line:
    collection_name = "all_embeddings"

    collection = chroma_client.get_or_create_collection(collection_name)
    for i, doc in enumerate(documents):
        embedding = embedding_model.get_text_embedding(doc.text)
        doc_id = f"{file_name}_{i}" if len(documents) > 1 else file_name
        collection.add(
            documents=[doc.text],
            metadatas=[{"email": email, "file": file_name, "source": doc.metadata["source"]}],
            ids=[doc_id],
            embeddings=[embedding]
        )
    return "embeddings saved"
