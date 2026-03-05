#!/usr/bin/env python3
"""Monitor LoCoMo Sonnet memory DB build progress.

Checks whether per-conversation memory caches are ready under db root:
  <db_root>/<conv_id>/{core_memory.json, vector_store/index.faiss, vector_store/payload.json}
or legacy layout:
  <db_root>/<conv_id>/{core_memory.json, index.faiss, payload.json}
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_locomo_conversations(data_path: Path) -> List[Dict[str, Any]]:
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_conv_ids(conversations: List[Dict[str, Any]]) -> List[str]:
    return [str(c["conversation_id"]) for c in conversations]


def _has_store_files(conv_dir: Path) -> bool:
    core_ok = (conv_dir / "core_memory.json").exists()

    # Prefer new layout
    vs_dir = conv_dir / "vector_store"
    if (vs_dir / "index.faiss").exists() and (vs_dir / "payload.json").exists() and core_ok:
        return True

    # Backward-compatible old layout
    if (conv_dir / "index.faiss").exists() and (conv_dir / "payload.json").exists() and core_ok:
        return True

    return False


def _memory_count(conv_dir: Path) -> int:
    payload_candidates = [
        conv_dir / "vector_store" / "payload.json",
        conv_dir / "payload.json",
    ]
    for payload_path in payload_candidates:
        if not payload_path.exists():
            continue
        try:
            payload = json.load(open(payload_path, "r", encoding="utf-8"))
            memories = payload.get("memories", [])
            if isinstance(memories, list):
                return len(memories)
        except Exception:
            return -1
    return -1


def collect_status(db_root: Path, conv_ids: List[str]) -> Dict[str, Any]:
    ready: List[str] = []
    pending: List[str] = []
    per_conv: Dict[str, Dict[str, Any]] = {}

    for conv_id in conv_ids:
        conv_dir = db_root / conv_id
        ok = _has_store_files(conv_dir)
        mem_count = _memory_count(conv_dir) if ok else -1
        per_conv[conv_id] = {
            "ready": ok,
            "memory_count": mem_count,
            "path": str(conv_dir),
        }
        if ok:
            ready.append(conv_id)
        else:
            pending.append(conv_id)

    pct = (len(ready) / len(conv_ids) * 100.0) if conv_ids else 100.0
    return {
        "time": now_str(),
        "db_root": str(db_root),
        "total_conversations": len(conv_ids),
        "ready_conversations": len(ready),
        "pending_conversations": len(pending),
        "progress_percent": round(pct, 2),
        "ready_ids": ready,
        "pending_ids": pending,
        "per_conversation": per_conv,
    }


def print_status(status: Dict[str, Any]) -> None:
    print("============================================================")
    print(f"time:                 {status['time']}")
    print(f"db_root:              {status['db_root']}")
    print(f"total_conversations:  {status['total_conversations']}")
    print(f"ready_conversations:  {status['ready_conversations']}")
    print(f"pending_conversations:{status['pending_conversations']}")
    print(f"progress_percent:     {status['progress_percent']}%")

    pending_ids = status.get("pending_ids", [])
    if pending_ids:
        sample = ", ".join(pending_ids[:5])
        suffix = " ..." if len(pending_ids) > 5 else ""
        print(f"pending_sample:       {sample}{suffix}")
    print("============================================================")


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor LoCoMo Sonnet DB build progress")
    parser.add_argument(
        "--db-root",
        type=str,
        default="faiss_data/locomo/claude-sonnet-4-5",
        help="Root directory of Sonnet-built memory DB",
    )
    parser.add_argument(
        "--locomo-data",
        type=str,
        default="data/locomo/locomo_conversations.json",
        help="LoCoMo conversation json path",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=0,
        help="Polling interval in seconds; 0 means run once",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional json output path (rewritten every polling tick)",
    )
    args = parser.parse_args()

    db_root = Path(args.db_root)
    data_path = Path(args.locomo_data)
    if not data_path.exists():
        raise FileNotFoundError(f"LoCoMo data not found: {data_path}")

    conv_ids = get_conv_ids(load_locomo_conversations(data_path))

    while True:
        status = collect_status(db_root=db_root, conv_ids=conv_ids)
        print_status(status)

        if args.output_json:
            out = Path(args.output_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(status, f, ensure_ascii=False, indent=2)

        if args.interval <= 0:
            break
        time.sleep(args.interval)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
