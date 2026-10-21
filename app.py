import os
import sys
import warnings
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)
warnings.filterwarnings(
    "ignore",
    message=".*resume_download.*",
    category=FutureWarning,
    module="huggingface_hub.file_download"
)

import streamlit as st

from rag.embeddings import EmbeddingModel
from rag.llm_client import create_llm_client
from rag.processor import answer_question
from rag.processor import process_document
from rag.processor import process_evaluation
from rag.processor import deduplicate_passages
from rag.processor import format_reference_answer
from rag.processor import format_retrieved_passage
from rag.vector_store import VectorStore, query_db


st.set_page_config(
    page_title="Contexta",
    page_icon="💬",
    layout="wide"
)

EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))


@st.cache_resource
def load_rag_resources():
    vector_store = VectorStore()
    embedding_model = EmbeddingModel()
    return vector_store, embedding_model


def index_uploaded_documents(uploaded_files, embedding_model, vector_store):
    total_chunks = 0

    for uploaded_file in uploaded_files:
        document_dir = Path(__file__).resolve().parents[1] / "data" / "documents"
        document_dir.mkdir(parents=True, exist_ok=True)
        document_path = document_dir / uploaded_file.name
        document_path.write_bytes(uploaded_file.getvalue())

        chunks = process_document(str(document_path))
        if not chunks:
            continue

        for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            batch = chunks[start:start + EMBEDDING_BATCH_SIZE]
            embeddings = embedding_model.encode(batch)
            metadatas = [
                {"source": uploaded_file.name, "chunk_id": start + index}
                for index in range(len(batch))
            ]
            vector_store.add_documents(batch, embeddings, metadatas)

        total_chunks += len(chunks)

    return total_chunks


llm = create_llm_client()

try:
    vector_store, embedding_model = load_rag_resources()
except Exception as error:
    st.error("Contexta could not load its document index or embedding model.")
    st.exception(error)
    st.stop()

collection = vector_store.collection

if "messages" not in st.session_state:
    st.session_state.messages = []

if "contexta_metrics" not in st.session_state:
    st.session_state.contexta_metrics = {
        "questions": 0,
        "searches": 0,
        "grades": 0
    }

if "contexta_mode" not in st.session_state:
    st.session_state.contexta_mode = "Ask from uploaded docs"

st.title("Contexta")
st.caption("Grounded answers, search, and grading for your uploaded documents only")

with st.sidebar:
    st.header("Knowledge Base")
    uploaded_files = st.file_uploader(
        "Add one or more question banks or study documents",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("Index documents", use_container_width=True):
        with st.spinner("Building document memory..."):
            chunk_count = index_uploaded_documents(
                uploaded_files,
                embedding_model,
                vector_store
            )
        if chunk_count:
            st.success(f"Added {chunk_count} searchable sections.")
        else:
            st.error("No readable text was found.")

    st.divider()
    st.subheader("Mode")
    st.checkbox("Show retrieved sources", value=False, key="show_sources")
    modes = ["Ask from uploaded docs", "Search retrieved passages", "Grade answer"]
    selected_mode = st.radio(
        "Choose how you want to use Contexta",
        modes,
        index=modes.index(st.session_state.contexta_mode)
    )
    st.session_state.contexta_mode = selected_mode

    st.divider()
    st.metric("Indexed sections", collection.count())
    st.metric("Questions asked", st.session_state.contexta_metrics["questions"])
    st.metric("Searches run", st.session_state.contexta_metrics["searches"])
    st.metric("Grades given", st.session_state.contexta_metrics["grades"])

    if llm is None:
        st.warning("LLM not configured. Add settings to `.env` for AI answers.")
    else:
        st.success(f"AI model: {llm.model}")

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


st.subheader(f"{st.session_state.contexta_mode} mode")

if st.session_state.contexta_mode == "Grade answer":
    with st.container():
        st.write("Compare an answer against the retrieved academic reference material.")
        evaluation_question = st.text_input("Question", key="evaluation_question")
        maximum_marks = st.selectbox(
            "Maximum marks for this question",
            options=list(range(1, 11)),
            index=4,
            format_func=lambda marks: f"{marks} mark" if marks == 1 else f"{marks} marks",
            key="evaluation_maximum_marks"
        )
        student_answer = st.text_area("Student answer", key="student_answer")

        if st.button("Grade answer"):
            if not evaluation_question.strip() or not student_answer.strip():
                st.warning("Enter both a question and a student answer.")
            elif collection.count() == 0:
                st.warning("Index study material before grading an answer.")
            else:
                try:
                    result = process_evaluation(
                        evaluation_question,
                        student_answer,
                        collection,
                        embedding_model,
                        llm,
                        maximum_marks
                    )
                    evaluation = result["evaluation"]
                    st.session_state.contexta_metrics["grades"] += 1
                    st.subheader("Evaluation")
                    score_columns = st.columns(2)
                    score_columns[0].metric(
                        "Marks awarded",
                        f"{evaluation['total']} / {maximum_marks}"
                    )
                    score_columns[1].metric(
                        "Percentage",
                        f"{evaluation['total'] / maximum_marks:.0%}"
                    )
                    st.markdown(evaluation["feedback"])
                    missing_points = evaluation.get("missing_points", [])
                    if missing_points:
                        st.markdown("**Points to review**")
                        for point in missing_points:
                            st.markdown(f"- {point}")
                    if result.get("reference_answer"):
                        with st.expander("Reference answer"):
                            st.markdown(
                                format_reference_answer(result["reference_answer"])
                            )
                except Exception as error:
                    st.error("Answer evaluation failed.")
                    st.exception(error)
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if st.session_state.get("show_sources") and message.get("sources"):
                with st.expander("Retrieved sources"):
                    for source in message["sources"]:
                        st.caption(source)
    question = st.chat_input(
        "Search the retrieved passages from your uploaded documents..."
        if st.session_state.contexta_mode == "Search retrieved passages"
        else "Ask about the uploaded documents..."
    )

    if question:
        if collection.count() == 0:
            st.warning("Upload and index a question bank or study document first.")
        else:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            try:
                with st.chat_message("assistant"):
                    with st.spinner("Searching your documents..."):
                        docs = query_db(
                            collection,
                            embedding_model,
                            question,
                            k=4
                        )

                    if st.session_state.contexta_mode == "Search retrieved passages":
                        passages = deduplicate_passages(docs)
                        formatted_passages = [
                            f"### Passage {index}\n\n{format_retrieved_passage(passage)}"
                            for index, passage in enumerate(passages, start=1)
                        ]
                        response = {
                            "answer": (
                                "Here are the most relevant passages from your uploaded documents.\n\n"
                                + "\n\n---\n\n".join(formatted_passages)
                            ),
                            "sources": passages
                        }
                        st.markdown(response["answer"])
                        st.session_state.contexta_metrics["searches"] += 1
                    else:
                        response = answer_question(
                            question,
                            collection,
                            embedding_model,
                            llm
                        )
                        st.session_state.contexta_metrics["questions"] += 1
                        if llm is None:
                            st.info(response["answer"])
                        else:
                            st.markdown(response["answer"])

                    if st.session_state.get("show_sources"):
                        with st.expander("Retrieved sources"):
                            for source in response["sources"]:
                                st.markdown(format_retrieved_passage(source))

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["answer"],
                    "sources": response["sources"]
                })
            except Exception as error:
                st.error("The request could not be processed.")
                st.exception(error)