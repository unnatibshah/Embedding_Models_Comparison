from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient

# Load the embedding model (all-mpnet-base-v2)
model_name = "all-MiniLM-L6-v2"  # Change model to all-mpnet-base-v2
model_kwargs = {'device': 'cpu'}  # Use 'cuda' if GPU is available
encode_kwargs = {'normalize_embeddings': False}  # Adjust as required
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Connect to Qdrant client
url = "http://localhost:6333"
client = QdrantClient(
    url=url, prefer_grpc=False
)

print(client)
print("##############")

# Load the Qdrant database
db = Qdrant(client=client, embeddings=embeddings, collection_name="mpnet_l6_vector_db")

print(db)
print("######")

# Query the database
query = "Could you give me the user guide on how to submit a training request for ComplianceWire trainings of both MedTech and Vision?"
docs = db.similarity_search_with_score(query=query, k=5)

# Print the results
for i in docs:
    doc, score = i
    print({"score": score, "content": doc.page_content, "metadata": doc.metadata})
