import ollama


# LOAD AND CHUNK THE DOCUMENT

def load_and_chunk(filepath: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    """
    Read a text file and split it into overlapping chunks.

    Why chunk? The LLM has a limited context window — we can't dump the
    entire document into the prompt. Instead, we retrieve only the
    most relevant pieces.

    Why overlap? So we don't accidentally cut a sentence in half at a
    chunk boundary and lose the meaning.
    """
    text = open(filepath, encoding="utf-8").read()

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap  # step forward, keeping some overlap

    return chunks


# EMBED TEXT INTO VECTORS

def get_embedding(text: str) -> list[float]:
    """
    Convert a string into a vector (list of numbers) using Ollama.

    nomic-embed-text is a model purpose-built for embeddings.
    It turns text into 768 numbers that encode semantic meaning —
    similar text produces similar vectors.
    """
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return response["embedding"]


def embed_chunks(chunks: list[str]) -> list[dict]:
    """
    Embed every chunk and return a list of dicts with text + vector.
    """
    print(f"Embedding {len(chunks)} chunks...")
    embedded = []
    for i, chunk in enumerate(chunks):
        vector = get_embedding(chunk)
        embedded.append({"text": chunk, "vector": vector})
        print(f"  {i + 1}/{len(chunks)} done", end="\r")
    print(f"  All {len(chunks)} chunks embedded.   ")
    return embedded


# FIND MOST RELEVANT CHUNKS

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Measure how similar two vectors are (range: -1 to 1, higher = more similar).

    We implement this manually so you can see exactly what's happening.
    This is what a vector database does internally at massive scale.
    """
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = sum(a * a for a in vec_a) ** 0.5
    magnitude_b = sum(b * b for b in vec_b) ** 0.5

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def retrieve(question: str, embedded_chunks: list[dict], top_k: int = 3) -> list[str]:
    """
    Embed the question and find the top_k most similar chunks.

    This is the core of RAG: instead of searching by keywords,
    we search by *meaning*.
    """
    question_vector = get_embedding(question)

    # Score every chunk against the question
    scored = []
    for chunk in embedded_chunks:
        score = cosine_similarity(question_vector, chunk["vector"])
        scored.append((score, chunk["text"]))

    # Sort by score descending, take the top_k
    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [text for _, text in scored[:top_k]]

    return top_chunks


# ASK THE LLM WITH CONTEXT

def ask(question: str, context_chunks: list[str]) -> str:
    """
    Build a RAG prompt and send it to gemma3 via Ollama.

    The key idea: we tell the model to ONLY use the provided context.
    This prevents hallucination — if the answer isn't in our document,
    the model should say so rather than making something up.
    """
    # Join the retrieved chunks into one context block
    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
    If the answer is not in the context, say "I don't have that information in my knowledge base."

    Context:
    {context}

    Question: {question}

    Answer:"""

    response = ollama.chat(
        model="gemma3:4b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1},  # low = more correct, less creative
    )

    return response["message"]["content"]


def main():
    print("=" * 50)
    print("  Simple RAG Chatbot")
    print("  Knowledge base: python_fundamentals.txt")
    print("=" * 50)

    # Load and chunk the document
    print("\n[1/2] Loading document...")
    chunks = load_and_chunk("python_fundamentals.txt", chunk_size=400, overlap=50)
    print(f"  Split into {len(chunks)} chunks")

    # Embed all chunks (only done once at startup)
    print("\n[2/2] Embedding chunks (one-time setup)...")
    embedded_chunks = embed_chunks(chunks)

    # Chat loop
    print("\nReady! Ask anything about Python fundamentals.")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        # Retrieve relevant context
        top_chunks = retrieve(question, embedded_chunks, top_k=3)

        # Generate answer
        print("\nAssistant: ", end="", flush=True)
        answer = ask(question, top_chunks)
        print(answer)
        print()


if __name__ == "__main__":
    main()
