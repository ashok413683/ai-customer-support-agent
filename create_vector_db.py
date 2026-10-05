from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


# 1. Load FAQ
loader = TextLoader("faq.txt")
documents = loader.load()

print("Documents:", len(documents))


# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print("Chunks:", len(chunks))


# 3. Create Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)


# 4. Store embeddings in Chroma
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("Vector database created successfully!")