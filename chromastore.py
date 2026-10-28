import chromadb

chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_or_create_collection(name="documents")


def add_chunks(chunks):

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for chunk in chunks:
        ids.append(f'{chunk["metadata"]["source"]}_{chunk["chunk_id"]}')
        documents.append(chunk["chunk_text"])
        embeddings.append(chunk["chunk_embedding"])
        metadatas.append(chunk["metadata"])

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )


def semantic_search(user_query_embedding, top_k):

    result = collection.query(
        query_embeddings=[user_query_embedding],
        n_results=top_k
    )

    ids = result['ids'][0]
    documents = result['documents'][0]
    distances = result['distances'][0]

    candidates = []

    for i in range(len(ids)):
        candidates.append({
            "chunk_text": documents[i],
            "score": 1 / (1 + distances[i])
        })

    candidates.sort(key=lambda item: item["score"], reverse=True)

    return candidates[:top_k]


def get_all_chunks():
    result = collection.get(include=["documents"])
    return result["documents"] or []
