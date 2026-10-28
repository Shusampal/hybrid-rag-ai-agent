from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")


model = SentenceTransformer(EMBEDDING_MODEL)


def create_embedding(text):
    return model.encode(text)
