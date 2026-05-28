from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.chunker import build_chunks_for_document
from app.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preview chunking results for one cleaned JSON document.")
    parser.add_argument("--file", required=True, help="Path to one cleaned JSON file.")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--max-chunks-per-doc", type=int, default=None)
    parser.add_argument("--max-section-splits", type=int, default=None)
    parser.add_argument("--min-paragraph-len", type=int, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(
        {
            "chunk_size": args.chunk_size,
            "max_chunks_per_doc": args.max_chunks_per_doc,
            "max_section_splits": args.max_section_splits,
            "min_paragraph_len": args.min_paragraph_len,
        }
    )

    file_path = Path(args.file).expanduser().resolve()
    with file_path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)

    chunks = build_chunks_for_document(document, str(file_path), config)
    print(f"file: {file_path}")
    print(f"doc_id: {document.get('doc_id', '')}")
    print(f"chunk_count: {len(chunks)}")
    print()

    for record in chunks:
        preview = record["chunk_text"][:220].replace("\n", " ")
        print(
            f"[{record['chunk_index']}] type={record['chunk_type']} "
            f"section={record['section_name']} len={len(record['chunk_text'])} "
            f"range=({record['char_start']},{record['char_end']})"
        )
        print(f"preview: {preview}")
        print()


if __name__ == "__main__":
    main()
