from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# Embedding model
embeddings = OllamaEmbeddings(
    model="qwen3-embedding:8b"
)


# Load existing Chroma database
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)


# Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# Test question
question = "What is your refund policy?"


# Search relevant documents
documents = retriever.invoke(question)


print("\nRelevant Documents:\n")


for i, document in enumerate(documents, start=1):

    print(f"--- Document {i} ---")
    print(document.page_content)
    print()