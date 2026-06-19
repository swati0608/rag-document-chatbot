"""
app.py
------
Gradio entry point for the RAG Document Chatbot.

Layout:
    Left column:  file upload + document info card.
    Right column: chat interface + collapsible "sources" panel + clear button.

Run locally:
    python app.py
On HuggingFace Spaces:
    The Space SDK will auto-detect this file and launch it.
"""

from __future__ import annotations

import os
import traceback
from typing import List, Tuple

import gradio as gr
from dotenv import load_dotenv

from document_processor import (
    DocumentProcessor,
    EmptyDocumentError,
    UnsupportedFileTypeError,
)
from llm_handler import HuggingFaceLLM, LLMAPIError, LLMConfigError
from rag_pipeline import RAGPipeline

# Load .env if present (no-op on HF Spaces, which uses Secrets).
load_dotenv()

# --------------------------------------------------------------------------- #
# Globals (single-user demo state; for multi-user, wrap in gr.State)
# --------------------------------------------------------------------------- #
MAX_HISTORY_TURNS = 5  # last N (user, assistant) exchanges shown

processor = DocumentProcessor()
pipeline = RAGPipeline(chunk_size=500, chunk_overlap=50, top_k=3)

try:
    llm = HuggingFaceLLM()
    LLM_INIT_ERROR: str | None = None
except LLMConfigError as e:
    llm = None
    LLM_INIT_ERROR = str(e)


# --------------------------------------------------------------------------- #
# Handlers
# --------------------------------------------------------------------------- #
def handle_upload(file_obj) -> Tuple[str, List, str]:
    """
    Process an uploaded document and (re)build the vector index.

    Returns:
        (status_markdown, cleared_chat_history, cleared_sources_markdown)
    """
    if file_obj is None:
        return "⚠️ No file uploaded.", [], ""

    file_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)

    try:
        meta = processor.process(file_path)
    except UnsupportedFileTypeError as e:
        return f"❌ {e}", [], ""
    except EmptyDocumentError as e:
        return f"❌ {e}", [], ""
    except FileNotFoundError as e:
        return f"❌ {e}", [], ""
    except Exception as e:
        traceback.print_exc()
        return f"❌ Failed to parse document: {e}", [], ""

    try:
        n_chunks = pipeline.build_index(meta["text"])
    except Exception as e:
        traceback.print_exc()
        return f"❌ Failed to index document: {e}", [], ""

    status = (
        f"✅ **Document loaded**\n\n"
        f"- **File:** `{meta['file_name']}`\n"
        f"- **Type:** {meta['file_type'].upper()}\n"
        f"- **Pages:** {meta['page_count']}\n"
        f"- **Chunks indexed:** {n_chunks}\n\n"
        f"Ask a question on the right →"
    )
    return status, [], ""


def handle_question(
    question: str,
    history: List[List[str]],
) -> Tuple[List[List[str]], str, str]:
    """
    Run one RAG turn.

    Args:
        question: user's message.
        history: Gradio chat history (list of [user, assistant] pairs).

    Returns:
        (updated_history, sources_markdown, cleared_textbox)
    """
    history = history or []

    if llm is None:
        history.append([question or "", f"❌ {LLM_INIT_ERROR}"])
        return history[-MAX_HISTORY_TURNS:], "", ""

    if not question or not question.strip():
        return history[-MAX_HISTORY_TURNS:], "", ""

    if not pipeline.is_indexed:
        history.append([question, "⚠️ Please upload a document first."])
        return history[-MAX_HISTORY_TURNS:], "", ""

    try:
        prompt, chunks = pipeline.prepare_query(question)
    except Exception as e:
        traceback.print_exc()
        history.append([question, f"❌ Retrieval error: {e}"])
        return history[-MAX_HISTORY_TURNS:], "", ""

    if not chunks:
        answer = pipeline.NO_CONTEXT_MESSAGE
    else:
        try:
            answer = llm.generate(prompt)
        except LLMAPIError as e:
            answer = f"❌ {e}"
        except Exception as e:
            traceback.print_exc()
            answer = f"❌ Unexpected LLM error: {e}"

    history.append([question, answer])
    sources_md = _format_sources(chunks)
    return history[-MAX_HISTORY_TURNS:], sources_md, ""


def handle_clear() -> Tuple[List, str]:
    """Clear chat history and sources panel."""
    return [], ""


def _format_sources(chunks: List[str]) -> str:
    """Render retrieved chunks as a numbered markdown block."""
    if not chunks:
        return "_No source chunks were retrieved._"
    parts = ["### 📚 Retrieved source chunks", ""]
    for i, c in enumerate(chunks, start=1):
        snippet = c.strip()
        parts.append(f"**Chunk {i}**\n\n> {snippet}\n")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
CUSTOM_CSS = """
.gradio-container { max-width: 1200px !important; margin: auto; }
#title-md h1 { margin-bottom: 0; }
#title-md p { margin-top: 4px; color: #9aa4b2; }
.doc-card { border: 1px solid #2a2f3a; border-radius: 12px; padding: 16px; }
footer { display: none !important; }
"""


def build_ui() -> gr.Blocks:
    """Construct the Gradio Blocks app."""
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
        neutral_hue="slate",
    ).set(body_background_fill="#0f1115", body_text_color="#e6e8ee")

    with gr.Blocks(theme=theme, css=CUSTOM_CSS, title="RAG Document Chatbot") as demo:
        gr.Markdown(
            "# 📄 RAG Document Chatbot\n"
            "Upload a PDF or TXT file and ask grounded questions about its contents. "
            "Powered by Mistral-7B + ChromaDB + sentence-transformers.",
            elem_id="title-md",
        )

        if LLM_INIT_ERROR:
            gr.Markdown(f"> ⚠️ **LLM not configured:** {LLM_INIT_ERROR}")

        with gr.Row():
            # ---------- Left column ---------- #
            with gr.Column(scale=1):
                gr.Markdown("### 1. Upload your document")
                file_in = gr.File(
                    label="PDF or TXT",
                    file_types=[".pdf", ".txt"],
                    file_count="single",
                )
                doc_status = gr.Markdown(
                    "_No document loaded yet._",
                    elem_classes=["doc-card"],
                )

            # ---------- Right column ---------- #
            with gr.Column(scale=2):
                gr.Markdown("### 2. Ask questions")
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=420,
                    show_copy_button=True,
                    bubble_full_width=False,
                )
                with gr.Row():
                    question_in = gr.Textbox(
                        placeholder="Ask a question about the document…",
                        show_label=False,
                        scale=5,
                        autofocus=True,
                    )
                    ask_btn = gr.Button("Ask", variant="primary", scale=1)
                clear_btn = gr.Button("🧹 Clear conversation", variant="secondary")

                with gr.Accordion("📚 Sources used for the last answer", open=False):
                    sources_md = gr.Markdown("_No sources yet._")

        # ---------- Wiring ---------- #
        file_in.upload(
            fn=handle_upload,
            inputs=[file_in],
            outputs=[doc_status, chatbot, sources_md],
            show_progress="full",
        )

        ask_btn.click(
            fn=handle_question,
            inputs=[question_in, chatbot],
            outputs=[chatbot, sources_md, question_in],
            show_progress="minimal",
        )
        question_in.submit(
            fn=handle_question,
            inputs=[question_in, chatbot],
            outputs=[chatbot, sources_md, question_in],
            show_progress="minimal",
        )

        clear_btn.click(fn=handle_clear, outputs=[chatbot, sources_md])

    return demo


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    app = build_ui()
    app.queue(max_size=16).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        show_api=False,
    )
