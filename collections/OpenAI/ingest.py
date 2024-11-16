from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

# Load and split documents
loader = PyPDFLoader("/Users/unnati/Downloads/Embeddings_project/collections/data.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# Load the OpenAI embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

print("Embedding model loaded")

# Initialize Qdrant
url = "http://localhost:6333"
qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    url=url,
    prefer_grpc=False,
    collection_name="vector_db_openai"
)

print("Vector DB Successfully Created!")
