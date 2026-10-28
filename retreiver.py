from answer import get_formatted_answer
from embedding import create_embedding
from chromastore import semantic_search
from keywordSearch import keyword_search
from reranker import rerank_candidates


def get_result(user_query, top_k=3):
    user_query_embedding = create_embedding(user_query)

    semantic_search_results = semantic_search(user_query_embedding, top_k)
    keyword_search_result = keyword_search(user_query)

    combined_candidates = []
    seen_chunks = set()
    for candidate in semantic_search_results + keyword_search_result:
        chunk_text = candidate["chunk_text"]
        if chunk_text not in seen_chunks:
            seen_chunks.add(chunk_text)
            combined_candidates.append(candidate)

    suitable_candidates = rerank_candidates(
        user_query,
        combined_candidates,
        []
    )

    result = get_formatted_answer(user_query, suitable_candidates)

    return result
