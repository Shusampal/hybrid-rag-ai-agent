from chromastore import add_chunks
from embedding import create_embedding


def create_rag(kb_text, source, chunk_size=200, overlap=50):

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between zero and chunk_size - 1")

    words = kb_text.split()
    chunks = []
    step = chunk_size - overlap

    for chunk_id, start in enumerate(range(0, len(words), step)):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])
        chunk_embedding = create_embedding(chunk_text)
        chunks.append({
            "chunk_text": chunk_text,
            "chunk_id": chunk_id,
            "chunk_embedding": chunk_embedding,
            "metadata": {
                "source": source
            }
        })
        if end == len(words):
            break

    # ? add chunks to chromadb
    add_chunks(chunks)
