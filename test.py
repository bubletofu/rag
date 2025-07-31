
import chromadb

client = chromadb.HttpClient(host="localhost", port=8000)

collection = client.get_or_create_collection("test")
collection.add(
    documents=["Xin chào", "Hello ChromaDB"],
    ids=["1", "2"]
)

results = collection.query(query_texts=["chào"], n_results=1)
print(results)
