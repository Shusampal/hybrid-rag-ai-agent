import re

from chromastore import get_all_chunks


def keyword_search(user_query, top_k=3):
    all_chunks = get_all_chunks()
    user_question_words = set(re.findall(r"\w+", user_query.lower()))
    if not user_question_words:
        return []

    candidates = []

    for chunk in all_chunks:
        chunk_words = set(re.findall(r"\w+", chunk.lower()))
        matching_words = chunk_words & user_question_words
        keymatch_score = len(matching_words) / len(user_question_words)
        if keymatch_score == 0:
            continue
        candidates.append({
            "chunk_text": chunk,
            "score": keymatch_score
        })

    candidates.sort(key=lambda item: item["score"], reverse=True)

    return candidates[:top_k]
