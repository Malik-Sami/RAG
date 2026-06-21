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

def retrieve(question: str, chat_history, top_k: int = 5):
    """
    Retrieve using current question + recent conversation context.
    """

    recent_history = " ".join(
        [msg["content"] for msg in chat_history[-4:]]
    )

    search_query = f"{recent_history} {question}"

    question_embedding = get_embedding(search_query)

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

def is_followup(question):
    followups = [
        "tell me more",
        "explain more",
        "continue",
        "go on",
        "elaborate",
        "what else"
    ]

    return question.lower() in followups

def ask(question: str, context_chunks, chat_history):
    """
    Build a RAG prompt and send it to gemma3 via Ollama.

    The key idea: we tell the model to ONLY use the provided context.
    This prevents hallucination — if the answer isn't in our document,
    the model should say so rather than making something up.
    """
    context = "\n\n---\n\n".join(context_chunks)

    history_text = "\n".join(
    [f"{msg['role']}: {msg['content']}" for msg in chat_history]
    )

    prompt = f"""You are a Python Fundamentals Tutor.

                Your job is to teach Python clearly using the provided knowledge base.

                Rules:
                1. Answer using the provided context ONLY.
                2. Use conversation history to understand follow-up questions.
                3. If the user asks "tell me more", "explain further", or similar,
                continue expanding the previous topic instead of repeating yourself.
                4. Keep explanations beginner-friendly.
                5. If the answer does not exist in the context, say:
                "I don't have that information in my knowledge base."
                6. If the user asks "tell me more" or "explain further":
                    - DO NOT repeat the previous answer.
                    - Expand with NEW details, examples, deeper explanation, or related concepts.
                7. Avoid repeating wording from the previous answer.

                Conversation History:
                {history_text}

                Knowledge Base Context:
                {context}

                User Question:
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
    print("      Simple RAG Chatbot")
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
    chat_history = []

    while True:
        question = input("You: ").strip()

        if not question:
            continue

        if question.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if is_small_talk(question):
            print("\nAssistant: Hello 👋 I can help you learn Python fundamentals.")
            print("Ask me about variables, loops, functions, OOP, errors, or anything Python-related.\n")
            continue

        if is_followup(question) and chat_history:
            question = chat_history[-2]["content"] + " " + question

        # Retrieve context
        top_chunks = retrieve(question, chat_history, top_k=5)

        # Generate answer
        print("\nAssistant is thinking...", flush=True)
        answer = ask(question, top_chunks, chat_history)
        print(f"\nAssistant: {answer}")
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": answer})
        


if __name__ == "__main__":
    main()