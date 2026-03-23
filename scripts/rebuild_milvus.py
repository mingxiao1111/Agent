from __future__ import annotations

import time
from pathlib import Path
import sys

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import tcm


def _remove_file(path: Path) -> None:
    if path.exists():
        path.unlink()
        print(f"[clean] removed meta: {path}")
    else:
        print(f"[clean] meta not found (skip): {path}")


def _assert_backend_is_milvus() -> None:
    backend = tcm._vector_backend()
    if backend != "milvus":
        raise SystemExit(
            "TCM_VECTOR_BACKEND is not 'milvus'. "
            "Please set TCM_VECTOR_BACKEND=milvus in .env before rebuilding."
        )


def _warm_rebuild_case_collection() -> None:
    t0 = time.time()
    tcm._tcm_vector_store.cache_clear()
    store = tcm._tcm_vector_store()
    if store is None:
        raise RuntimeError("case collection rebuild failed: _tcm_vector_store() returned None")
    dt = time.time() - t0
    print(f"[ok] case collection rebuilt in {dt:.2f}s")


def _warm_rebuild_patent_collection() -> None:
    t0 = time.time()
    tcm._patent_medicine_docs.cache_clear()
    tcm._patent_vector_store.cache_clear()
    store = tcm._patent_vector_store()
    if store is None:
        raise RuntimeError("patent collection rebuild failed: _patent_vector_store() returned None")
    dt = time.time() - t0
    print(f"[ok] patent collection rebuilt in {dt:.2f}s")


def main() -> None:
    load_dotenv()
    _assert_backend_is_milvus()

    case_collection = tcm._milvus_case_collection_name()
    patent_collection = tcm._milvus_patent_collection_name()
    print(f"[info] backend=milvus")
    print(f"[info] case_collection={case_collection}")
    print(f"[info] patent_collection={patent_collection}")
    print(f"[info] index_type={tcm._milvus_index_params().get('index_type')}")

    # Remove local meta to force drop_old rebuild path for both collections.
    _remove_file(tcm._vector_meta_path())
    _remove_file(tcm._patent_vector_meta_path())

    # Clear caches that may hold previous store handles or documents.
    tcm._tcm_embeddings.cache_clear()
    tcm._case_records.cache_clear()
    tcm._tcm_vector_store.cache_clear()
    tcm._patent_medicine_docs.cache_clear()
    tcm._patent_vector_store.cache_clear()

    _warm_rebuild_case_collection()
    _warm_rebuild_patent_collection()
    print("[done] Milvus collections rebuilt successfully.")


if __name__ == "__main__":
    main()
