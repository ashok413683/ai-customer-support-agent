from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


loader = TextLoader("faq.txt")

documents = loader.load()

print("Documents:", len(documents))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print("Chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print("\n--- Chunk", i + 1, "---")
    print(chunk.page_content)