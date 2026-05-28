from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.builder import build_index_batch, configure_logging
from app.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build one RAG index batch into local Qdrant and OpenSearch.")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--state-file", default=None)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--cleaned-corpus-dir", default=None)
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument("--opensearch-url", default=None)
    parser.add_argument("--opensearch-index-name", default=None)
    parser.add_argument("--qdrant-collection-name", default=None)
    parser.add_argument("--embed-batch-size", type=int, default=None)
    parser.add_argument("--embed-max-seq-length", type=int, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(
        {
            "sample_size": args.sample_size,
            "start_offset": args.start_offset,
            "cleaned_corpus_dir": args.cleaned_corpus_dir,
            "model_name_or_path": args.model_name_or_path,
            "opensearch_url": args.opensearch_url,
            "opensearch_index_name": args.opensearch_index_name,
            "qdrant_collection_name": args.qdrant_collection_name,
            "embed_batch_size": args.embed_batch_size,
            "embed_max_seq_length": args.embed_max_seq_length,
        }
    )
    log_file = args.log_file or config.logs_dir / f"run_build_{args.start_offset}_{args.sample_size}.log"
    configure_logging(log_file)
    result = build_index_batch(
        sample_size=args.sample_size,
        start_offset=args.start_offset,
        config=config,
        state_file=args.state_file,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
