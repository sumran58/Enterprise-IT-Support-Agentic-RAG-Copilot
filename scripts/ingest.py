
from pathlib import Path
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.ingestion import load_file, chunk_documents
from app.rag.vectorstore import add_documents
from app.core.config import get_settings


settings = get_settings()

BASE_DIR = Path(__file__).resolve().parents[1]
KB_DIR = BASE_DIR / "data" / "sample_kb"


def main():
    print(f"KB directory: {KB_DIR}")

    if not KB_DIR.exists():
        raise FileNotFoundError(f"KB directory not found: {KB_DIR}")

    all_chunks = []

    for path in KB_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in {
            ".pdf",
            ".txt",
            ".md",
            ".docx",
        }:
            print(f"\nLoading: {path.name}")

            docs = load_file(path)
            print(f"Loaded documents: {len(docs)}")

            chunks = chunk_documents(docs)
            print(f"Created chunks: {len(chunks)}")

            all_chunks.extend(chunks)

    if not all_chunks:
        raise RuntimeError("No documents/chunks found in the KB directory.")

    print(f"\nTotal chunks to upload: {len(all_chunks)}")
    print(f"Pinecone index: {settings.pinecone_index_name}")
    print(f"Pinecone namespace: {settings.pinecone_namespace}")

    add_documents(all_chunks)

    print("\nIngestion completed successfully.")


if __name__ == "__main__":
    main()

