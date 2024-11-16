from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
from qdrant_client import QdrantClient

# Load OpenAI embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Connect to Qdrant
url = "http://localhost:6333"
client = QdrantClient(url=url, prefer_grpc=False)

print(client)
print("##############")

# Load the Qdrant database
db = Qdrant(client=client, embeddings=embeddings, collection_name="vector_db_openai")

print(db)
print("######")

# Query the database
query = "Could you give me the user guide on how to submit a training request for ComplianceWire trainings of both MedTech and Vision?"

docs = db.similarity_search_with_score(query=query, k=5)
for i in docs:
    doc, score = i
    print({"score": score, "content": doc.page_content, "metadata": doc.metadata})
