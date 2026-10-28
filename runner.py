import asyncio
from pathlib import Path

from rag import create_rag
from retreiver import get_result


async def runner():
    print("="*40)
    print("AI Agent With Vector DB")
    print("="*40)
    # ? read text from file
    kb_text = Path("./python.txt").read_text()

    # ? build rag
    """
        #? split text into individual words
        #? create chunks
        #? add embedding to each chunk
        #? store chunks in chroma DB
    """
    create_rag(kb_text, "python.txt")
    # ? start loop
    while True:
        user_query = input("USER: ").strip().lower()

        if user_query in ["exit", "bye", "quit"]:
            print("AI: GoodBye !!")
            break

        # ? get result
        result = get_result(user_query)

        print(f"AI: {result}\n")


if __name__ == "__main__":
    asyncio.run(runner())
