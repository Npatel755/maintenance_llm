import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import streamlit as st

from rag import (
    extract_pdf,
    chunk_pages,
    embed_chunks,
    save_embeddings,
    answer_question,
)


st.set_page_config(
    page_title="Maintenance Manual Assistant",
    page_icon="📘",
    layout="centered",
)

st.title("Maintenance Manual Assistant")
st.caption(
    "Upload a manual, index it, and ask questions with source citations."
)


# Remember indexed documents and the latest answer during this session.
if "indexed_sources" not in st.session_state:
    st.session_state.indexed_sources = set()

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None


uploaded_file = st.file_uploader(
    "Choose a maintenance manual",
    type=["pdf"],
)

if uploaded_file is None:
    st.info("Upload a PDF to get started.")
    st.stop()


# Identify the document using its contents and filename.
pdf_bytes = uploaded_file.getvalue()
filename = Path(uploaded_file.name).name
document_hash = hashlib.sha256(pdf_bytes).hexdigest()

source = f"{document_hash}/{filename}"
is_indexed = source in st.session_state.indexed_sources


if st.button("Index manual", disabled=is_indexed):
    try:
        with st.spinner("Reading the PDF and creating embeddings..."):
            # Save temporarily so extract_pdf can open a file path.
            with TemporaryDirectory() as temporary_folder:
                pdf_path = Path(temporary_folder) / "manual.pdf"
                pdf_path.write_bytes(pdf_bytes)

                pages = extract_pdf(pdf_path)

            chunks = chunk_pages(
                pages,
                chunk_size=1000,
                overlap=200,
            )

            if not chunks:
                raise ValueError(
                    "No readable text was found. "
                    "A scanned PDF may need OCR first."
                )

            embeddings = embed_chunks(chunks)

            save_embeddings(
                chunks=chunks,
                embeddings=embeddings,
                source=source,
            )

        # Mark the document ready only after indexing succeeds.
        st.session_state.indexed_sources.add(source)
        st.session_state.last_answer = None
        is_indexed = True

        st.success(
            f"Indexed {len(pages)} pages into {len(chunks)} chunks."
        )

    except Exception as error:
        st.error(f"Could not index the manual: {error}")
        st.stop()


if not is_indexed:
    st.info("Click 'Index manual' before asking questions.")
    st.stop()


st.caption(f"Answering questions about: {filename}")

with st.form("question_form"):
    question = st.text_input(
        "Your question",
        placeholder="What can cause an electric motor to overheat?",
    )

    submitted = st.form_submit_button("Ask")


if submitted:
    question = question.strip()

    if not question:
        st.warning("Enter a question first.")
        st.stop()

    st.session_state.last_answer = None

    try:
        with st.spinner("Searching the manual and writing an answer..."):
            result = answer_question(
                question,
                n_results=4,
                source=source,
            )

        st.session_state.last_answer = {
            "source": source,
            "question": question,
            "result": result,
        }

    except Exception as error:
        st.error(f"Could not generate an answer: {error}")
        st.stop()


saved_answer = st.session_state.last_answer

# Only show an answer belonging to the selected document.
if saved_answer is not None and saved_answer["source"] == source:
    result = saved_answer["result"]

    st.subheader("Answer")
    st.caption(f"Question: {saved_answer['question']}")
    st.markdown(result["answer"])

    st.subheader("Retrieved passages")

    for number, passage in enumerate(result["sources"], start=1):
        passage_filename = Path(passage["source"]).name

        label = (
            f"[{number}] {passage_filename}"
            f" — PDF page {passage['page']}"
        )

        with st.expander(label):
            st.text(passage["text"])