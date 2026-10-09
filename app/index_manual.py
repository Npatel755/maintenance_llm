import argparse
from pathlib import Path
from app.rag import extract_pdf, chunk_pages, embed_chunks, save_embeddings


def main():
    parser = argparse.ArgumentParser(
        description="Index a PDF for the maintenance manual assistant."
    )

    parser.add_argument(
        "pdf_path",
        type=Path,
        help="Path to the PDF you want to index."
    )

    args = parser.parse_args()
    pdf_path = args.pdf_path

    if not pdf_path.is_file():
        parser.error(f"File does not exist: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        parser.error("Please provide a PDF file.")

    print(f"\nReading: {pdf_path.name}")
    pages = extract_pdf(pdf_path)
    print(f"Pages extracted: {len(pages)}")

    chunks = chunk_pages(pages,chunk_size=1000,overlap=200)
    print(f"Chunks created: {len(chunks)}")

    if not chunks:
        parser.error(
            "No text was extracted. The PDF may need OCR if it is scanned."
        )

    embeddings = embed_chunks(chunks)

    collection = save_embeddings(
        chunks=chunks,
        embeddings=embeddings,
        source=str(pdf_path)
    )
    print(f"\nSuccessfully indexed: {pdf_path.name}")
    print(f"Total chunks in the database: {collection.count()}")

if __name__ == "__main__":
    main()