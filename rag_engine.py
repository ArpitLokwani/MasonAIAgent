
from pathlib import Path

import chromadb

from config import (
    client,
    OPENAI_MODEL,
    EMBEDDING_MODEL,
    FAQ_FILE,
    CHROMA_DB_PATH
)


class MasonBroRAG:

    def __init__(self):
        # Connect to persistent ChromaDB
        self.db_client = chromadb.PersistentClient(
            path=str(CHROMA_DB_PATH)
        )

        self.collection = self.db_client.get_or_create_collection(
            name="masonbro_knowledge"
        )

    # ----------------------------------
    # 1. Load knowledge base
    # ----------------------------------

    def load_documents(self):

        if not FAQ_FILE.exists():
            raise FileNotFoundError(
                f"Knowledge file not found: {FAQ_FILE}"
            )

        text = FAQ_FILE.read_text(encoding="utf-8")

        print("MasonBro knowledge base loaded!")

        return text

    # ----------------------------------
    # 2. Split into chunks
    # ----------------------------------

    def create_chunks(self, text, chunk_size=500):

        paragraphs = [
            paragraph.strip()
            for paragraph in text.split("\n\n")
            if paragraph.strip()
        ]

        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:

            if len(current_chunk) + len(paragraph) + 1 > chunk_size:

                if current_chunk:
                    chunks.append(current_chunk.strip())

                current_chunk = paragraph

            else:

                if current_chunk:
                    current_chunk += "\n" + paragraph
                else:
                    current_chunk = paragraph

        if current_chunk:
            chunks.append(current_chunk.strip())

        print(f"Created {len(chunks)} chunks")

        return chunks

    # ----------------------------------
    # 3. Generate embeddings
    # ----------------------------------

    def create_embeddings(self, texts):

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts
        )

        embeddings = [
            item.embedding
            for item in response.data
        ]

        return embeddings

    # ----------------------------------
    # 4. Store embeddings in ChromaDB
    # ----------------------------------

    def index_knowledge(self):

        document = self.load_documents()

        chunks = self.create_chunks(document)

        if not chunks:
            raise ValueError("Knowledge base is empty!")

        # Generate embeddings
        embeddings = self.create_embeddings(chunks)

        # Remove previous FAQ entries
        existing = self.collection.get(
            where={"source": "masonbro_faq.txt"}
        )

        if existing["ids"]:
            self.collection.delete(ids=existing["ids"])

        # Store new chunks
        ids = [
            f"masonbro_faq_{i}"
            for i in range(len(chunks))
        ]

        metadata = [
            {"source": "masonbro_faq.txt"}
            for _ in chunks
        ]

        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadata
        )

        print("MasonBro knowledge indexed successfully!")

    # ----------------------------------
    # 5. Retrieve relevant information
    # ----------------------------------

    def retrieve_documents(self, question):

        if self.collection.count() == 0:
            raise RuntimeError("Knowledge base is not indexed.")

        question_embedding = self.create_embeddings(
            [question]
        )[0]

        results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=min(3, self.collection.count())
        )

        documents = results["documents"][0] or []

        print("Retrieved relevant documents:", len(documents))

        return documents

    # ----------------------------------
    # 6. Generate answer using LLM
    # ----------------------------------

    def generate_answer(self, question, documents):

        if not documents:
            return (
                "I couldn't find that information in the "
                "MasonBro knowledge base. Please contact "
                "the MasonBro team for confirmation."
            )

        context = "\n\n".join(documents)

        instructions = """
        You are the official MasonBro customer support assistant.

        Your job is to answer customer questions about MasonBro.

        Follow these rules:

        1. Answer using only the supplied knowledge context.
        2. Never invent prices, services, availability or policies.
        3. If the answer is not in the context, say that the
           MasonBro team needs to confirm it.
        4. Be polite, clear and helpful.
        5. Keep answers concise.
        6. Do not claim a booking has been made or confirmed.
        """

        prompt = f"""
        MasonBro knowledge context:

        {context}

        Customer question:

        {question}
        """

        response = client.responses.create(
            model=OPENAI_MODEL,
            instructions=instructions,
            input=prompt
        )

        return response.output_text

    # ----------------------------------
    # 7. Complete RAG pipeline
    # ----------------------------------

    def ask(self, question):

        relevant_documents = self.retrieve_documents(
            question
        )

        answer = self.generate_answer(
            question,
            relevant_documents
        )

        return {
            "answer": answer,
            "sources": relevant_documents
        }
