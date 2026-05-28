#!/usr/bin/env bash
set -e

PROJECT_ROOT="/AII-heyan/ragtestv01_server_bundle_release"
PYTHON="/torch/venv3/pytorch/bin/python"
VENV_ACTIVATE="/torch/venv3/pytorch/bin/activate"

QDRANT_BIN="/AII-heyan/qdrant_bin_musl/qdrant"
QDRANT_STORAGE="/AII-heyan/qdrant_server_storage"
QDRANT_LOG="/AII-heyan/qdrant_server.log"
QDRANT_URL="http://127.0.0.1:6333"

COLLECTION="rag_chunks_full"

echo "================================================================================"
echo "AII-Heyan Hybrid RAG Demo Launcher"
echo "================================================================================"

cd "$PROJECT_ROOT"
source "$VENV_ACTIVATE"

echo
echo "[1/5] Checking Qdrant server..."

if curl -s "$QDRANT_URL/collections" >/dev/null 2>&1; then
    echo "Qdrant is already running."
else
    echo "Qdrant is not running. Starting Qdrant with storage:"
    echo "$QDRANT_STORAGE"

    nohup env QDRANT__STORAGE__STORAGE_PATH="$QDRANT_STORAGE" \
        "$QDRANT_BIN" \
        > "$QDRANT_LOG" 2>&1 &

    echo "Waiting for Qdrant to recover collection..."
    sleep 40
fi

echo
echo "[2/5] Checking Qdrant collections..."

curl -s "$QDRANT_URL/collections"
echo

echo
echo "[3/5] Checking Qdrant collection status..."

"$PYTHON" - <<'PY'
from qdrant_client import QdrantClient

client = QdrantClient(url="http://127.0.0.1:6333")
info = client.get_collection("rag_chunks_full")

print("Qdrant collection =", "rag_chunks_full")
print("status =", info.status)
print("points =", info.points_count)
print("indexed =", info.indexed_vectors_count)
print("optimizer_status =", info.optimizer_status)

if str(info.status).lower().find("green") == -1:
    raise SystemExit("ERROR: Qdrant collection is not green.")

if info.points_count != info.indexed_vectors_count:
    raise SystemExit("ERROR: Qdrant collection is not fully indexed.")
PY

echo
echo "[4/5] Checking OpenSearch..."

"$PYTHON" - <<'PY'
from app.config import load_config
from app.stores import _build_opensearch_client

cfg = load_config()
client = _build_opensearch_client(cfg)

exists = client.indices.exists(index=cfg.opensearch_index_name)

print("OpenSearch url =", cfg.opensearch_url)
print("OpenSearch index =", cfg.opensearch_index_name)
print("index_exists =", exists)

if not exists:
    raise SystemExit("ERROR: OpenSearch index does not exist.")

count = client.count(index=cfg.opensearch_index_name)["count"]
print("count =", count)

if count <= 0:
    raise SystemExit("ERROR: OpenSearch index is empty.")
PY

echo
echo "[5/5] Starting interactive demo..."
echo "================================================================================"
echo "You can now type a medical question."
echo "Press Enter without input to use the default FUO demo question."
echo "================================================================================"
echo

"$PYTHON" "$PROJECT_ROOT/demo_qa.py"