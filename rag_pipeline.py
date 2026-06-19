"""
rag_pipeline.py
---------------
RAG orchestration layer.

Responsibilities:
    - Split cleaned document text into overlapping chunks.
    - Embed chunks with a local sentence-transformers model (no API key).
    - Store / query an in-memory ChromaDB collection.
    - Retrieve top-k chunks for a user question and build a grounded prompt.

The pipeline is stateful: build_index() resets and rebuilds the collection
each time a new document is uploaded.
"""

from __future__ import annotations

import uuid
from typing import List, Tuple

import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


# --------------------------------------------------------------------------- #
# Prompt template
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You are a precise document assistant. Answer the user's question using ONLY "
    "the provided context from their document. If the context does not contain "
    "the answer, reply exactly: \"I couldn't find relevant information in the "
    "document.\" Do not invent facts. Keep answers concise and cite quoted text "
    "verbatim when helpful."
)

PROMPT_TEMPLATE = """{system}

# Context (retrieved from the document)
{context}

# User question
{question}

# Answer
"""


class RAGPipeline:
    """End-to-end retrieval pipeline backed by ChromaDB + sentence-transformers."""

    NO_CONTEXT_MESSAGE = "I couldn't find relevant information in the document."

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 3,
    ) -> None:
        """
        Args:
            embedding_model_name: HF model id for sentence-transformers.
            chunk_size: target characters per chunk.
            chunk_overlap: overlap (characters) between adjacent chunks.
            top_k: number of chunks to retrieve per query.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        # Embedder (downloads on first use, then cached).
        self.embedder = SentenceTransformer(embedding_model_name)

        # Chunker
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

        # In-memory Chroma client (ephemeral; resets between uploads).
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection_name = "rag_docs"
        self.collection = None  # created in build_index()
        self.is_indexed = False

    # ------------------------------------------------------------------ #
    # Indexing
    # ------------------------------------------------------------------ #
    def build_index(self, text: str) -> int:
        """
        Split `text` into chunks, embed them, and store in ChromaDB.

        Any existing collection is deleted first so each upload starts fresh.

        Args:
            text: Cleaned document text.

        Returns:
            Number of chunks indexed.
        """
        # Reset collection
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass  # didn't exist yet

        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        chunks = self.splitter.split_text(text)
        if not chunks:
            self.is_indexed = False
            return 0

        embeddings = self.embedder.encode(
            chunks,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()

        ids = [f"chunk-{i}-{uuid.uuid4().hex[:8]}" for i in range(len(chunks))]
        metadatas = [{"chunk_index": i} for i in range(len(chunks))]

        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        self.is_indexed = True
        return len(chunks)

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def retrieve(self, question: str) -> List[Tuple[str, float]]:
        """
        Retrieve top-k chunks most similar to `question`.

        Returns:
            List of (chunk_text, distance) tuples. Empty if no index exists.
        """
        if not self.is_indexed or self.collection is None:
            return []

        q_emb = self.embedder.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()

        result = self.collection.query(
            query_embeddings=q_emb,
            n_results=min(self.top_k, max(1, self.collection.count())),
        )

        docs = result.get("documents", [[]])[0]
        dists = result.get("distances", [[]])[0]
        return list(zip(docs, dists))

    # ------------------------------------------------------------------ #
    # Prompt construction
    # ------------------------------------------------------------------ #
    def build_prompt(self, question: str, chunks: List[str]) -> str:
        """Format retrieved chunks + question into the final LLM prompt."""
        if not chunks:
            context = "(no context available)"
        else:
            context = "\n\n---\n\n".join(
                f"[Chunk {i + 1}]\n{c}" for i, c in enumerate(chunks)
            )
        return PROMPT_TEMPLATE.format(
            system=SYSTEM_PROMPT,
            context=context,
            question=question.strip(),
        )

    # ------------------------------------------------------------------ #
    # Convenience: full retrieve-and-prompt step
    # ------------------------------------------------------------------ #
    def prepare_query(self, question: str) -> Tuple[str, List[str]]:
        """
        Retrieve chunks and build the LLM prompt in one call.

        Returns:
            (prompt, list_of_chunk_texts). If nothing is indexed yet, returns
            a prompt that instructs the model to emit NO_CONTEXT_MESSAGE.
        """
        retrieved = self.retrieve(question)
        chunks = [c for c, _ in retrieved]
        prompt = self.build_prompt(question, chunks)
        return prompt, chunks
