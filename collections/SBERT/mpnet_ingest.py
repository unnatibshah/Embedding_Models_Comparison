from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

# Load the PDF document
loader = PyPDFLoader("/Users/unnati/Downloads/Embeddings_project/collections/data.pdf")
documents = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# Load the embedding model (all-mpnet-base-v2)
model_name = "all-MiniLM-L6-v2"  
model_kwargs = {'device': 'cpu'} 
encode_kwargs = {'normalize_embeddings': False}   
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

print("Embedding model loaded")

# Initialize Qdrant vector database
url = "http://localhost:6333"
qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    url=url,
    prefer_grpc=False,
    collection_name="mpnet_l6_vector_db"
)

print("Vector DB Successfully Created!")
