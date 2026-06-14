import chromadb
import ollama


# CHROMA SETUP

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="python_fundamentals"
)


# LOAD + CHUNK DOCUMENT

def load_and_chunk(filepath: str, chunk_size: int = 400, overlap: int = 50):
    """
    Read a text file and split it into overlapping chunks.

    Why chunk? The LLM has a limited context window — we can't dump the
    entire document into the prompt. Instead, we retrieve only the
    most relevant pieces.

    Why overlap? So we don't accidentally cut a sentence in half at a
    chunk boundary and lose the meaning.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# EMBEDDING FUNCTION

def get_embedding(text: str):
    """
    Convert text into vector embeddings using Ollama.
    """
    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=text
    )
    return response["embedding"]


# STORE CHUNKS IN CHROMA

def store_chunks(chunks):
    """
    Embed and store all chunks in Chroma.
    Runs only once unless DB is deleted.
    """
    print(f"\nEmbedding and storing {len(chunks)} chunks...")

    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)

        collection.add(
            ids=[str(i)],
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[{
                "source": "python_fundamentals.txt",
                "chunk_number": i
            }]
        )

        print(f"Stored {i + 1}/{len(chunks)}", end="\r")

    print("\nAll chunks stored successfully.")


# RETRIEVE RELEVANT CHUNKS

def retrieve(question: str, top_k: int = 3):
    """
    Search Chroma for the most relevant chunks.
    """
    question_embedding = get_embedding(question)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )

    return results["documents"][0]


# ASK LLM

def is_small_talk(question):
    small_talk = [
        "hi", "hello", "hey",
        "how are you",
        "what's up",
        "who are you"
    ]

    return question.lower() in small_talk

def ask(question: str, context_chunks):
    """
    Build a RAG prompt and send it to gemma3 via Ollama.

    The key idea: we tell the model to ONLY use the provided context.
    This prevents hallucination — if the answer isn't in our document,
    the model should say so rather than making something up.
    """
    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""
You are a helpful assistant.

Answer ONLY using the context below.
If the answer is not in the context, say:
"I don't have that information in my knowledge base."

Context:
{context}

Question:
{question}

Answer:
"""

    response = ollama.chat(
        model="gemma3:4b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0.1
        }
    )

    return response["message"]["content"]


# MAIN PROGRAM

def main():
    print("=" * 50)
    print("      Simple RAG Chatbot (Chroma + Ollama)")
    print("=" * 50)

    # Only embed if DB is empty
    if collection.count() == 0:
        print("\nNo existing knowledge base found.")

        chunks = load_and_chunk(
            "python_fundamentals.txt",
            chunk_size=400,
            overlap=50
        )

        print(f"Loaded {len(chunks)} chunks.")
        store_chunks(chunks)

    else:
        print("\nKnowledge base already exists.")
        print(f"Loaded {collection.count()} stored chunks.")

    print("\nReady! Ask anything.")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()

        if not question:
            continue

        if question.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if is_small_talk(question):
            response = ollama.chat(
                model="gemma3:4b",
                messages=[
                    {"role": "user", "content": question}
                ]
            )

            print("\nAssistant:", response["message"]["content"])
            print()
            continue

        # Retrieve context
        top_chunks = retrieve(question)

        # Generate answer
        print("\nAssistant:", end=" ", flush=True)
        answer = ask(question, top_chunks)
        print(answer)
        print()


if __name__ == "__main__":
    main()