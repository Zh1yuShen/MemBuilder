#!/usr/bin/env python3
"""Sequential multi-model LOCOMO QA evaluation over Sonnet-built DB.

Workflow:
1) Monitor/wait until Sonnet memory DB is fully ready for all conversations.
2) Probe candidate model aliases via MetaAI and resolve one valid API model id per target model.
3) Run eval.runner in answer mode sequentially for each resolved model.
4) Save per-model logs and a consolidated summary report.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from llm_client_internal import create_llm_client
except ImportError:
    from llm_client import create_llm_client  # type: ignore


DEFAULT_MODEL_SPECS: List[Dict[str, Any]] = [
    {
        "label": "Claude 4.5 Sonnet",
        "aliases": [
            "claude-sonnet-4-5",
            "claude-sonnet-4-5-20250929",
            "claude-4.5-sonnet",
        ],
    },
    {
        "label": "GPT-4.1",
        "aliases": [
            "gpt-4.1",
            "gpt-4.1-2025-04-14",
        ],
    },
    {
        "label": "GPT-4.1-mini",
        "aliases": [
            "gpt-4.1-mini",
        ],
    },
    {
        "label": "Qwen3-30B-A3B",
        "aliases": [
            "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "qwen3-30b-a3b",
            "qwen3_30b_a3b",
            "Qwen/Qwen3-30B-A3B",
            "qwen3-30b-a3b-instruct-2507",
        ],
    },
    {
        "label": "Qwen3-4B Base",
        "aliases": [
            "Qwen/Qwen3-4B-Base-2507",
            "Qwen/Qwen3-4B-Base",
            "Qwen/Qwen3-4B-Instruct-2507",
            "Qwen/Qwen3-4B",
            "qwen3-4b-base",
            "qwen3-4b-instruct",
            "qwen3-4b",
            "qwen3_4b",
        ],
    },
]


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ts_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_").lower()


def load_locomo_conversations(data_path: Path) -> List[Dict[str, Any]]:
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def conv_ids_from_data(data: List[Dict[str, Any]]) -> List[str]:
    return [str(c["conversation_id"]) for c in data]


def _conv_ready(conv_dir: Path) -> bool:
    if not conv_dir.exists():
        return False
    core_ok = (conv_dir / "core_memory.json").exists()

    # new layout
    new_ok = (conv_dir / "vector_store" / "index.faiss").exists() and (conv_dir / "vector_store" / "payload.json").exists()
    # old layout
    old_ok = (conv_dir / "index.faiss").exists() and (conv_dir / "payload.json").exists()
    return core_ok and (new_ok or old_ok)


def collect_build_status(db_root: Path, conv_ids: List[str]) -> Dict[str, Any]:
    ready: List[str] = []
    pending: List[str] = []
    for conv_id in conv_ids:
        if _conv_ready(db_root / conv_id):
            ready.append(conv_id)
        else:
            pending.append(conv_id)

    pct = (len(ready) / len(conv_ids) * 100.0) if conv_ids else 100.0
    return {
        "time": now_str(),
        "db_root": str(db_root),
        "total": len(conv_ids),
        "ready": len(ready),
        "pending": len(pending),
        "progress_percent": round(pct, 2),
        "ready_ids": ready,
        "pending_ids": pending,
    }


def wait_for_build_ready(
    db_root: Path,
    conv_ids: List[str],
    poll_seconds: int,
    timeout_hours: float,
    status_json_path: Optional[Path] = None,
) -> Dict[str, Any]:
    deadline = datetime.now() + timedelta(hours=timeout_hours)

    while True:
        status = collect_build_status(db_root=db_root, conv_ids=conv_ids)
        print(
            f"[Build Monitor] {status['time']} ready={status['ready']}/{status['total']} "
            f"({status['progress_percent']}%) pending={status['pending']}"
        )
        if status_json_path:
            status_json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(status_json_path, "w", encoding="utf-8") as f:
                json.dump(status, f, ensure_ascii=False, indent=2)

        if status["pending"] == 0:
            return status

        if datetime.now() >= deadline:
            raise TimeoutError(
                f"Build not complete within {timeout_hours}h. Pending conversations: {status['pending_ids']}"
            )
        time.sleep(max(1, poll_seconds))


def probe_model_aliases(
    label: str,
    aliases: List[str],
    provider: str,
    max_probe_retries: int = 2,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "label": label,
        "aliases": aliases,
        "resolved_model": None,
        "attempts": [],
        "ok": False,
    }
    prompt = "Reply with exactly one token: OK"

    for alias in aliases:
        t0 = time.time()
        attempt_info: Dict[str, Any] = {"alias": alias, "ok": False}
        try:
            client = create_llm_client(provider=provider, model=alias)
            resp = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=8,
                max_retries=max_probe_retries,
            )
            attempt_info.update(
                {
                    "ok": True,
                    "latency_sec": round(time.time() - t0, 3),
                    "response_preview": str(resp).strip()[:120],
                }
            )
            report["attempts"].append(attempt_info)
            report["resolved_model"] = alias
            report["ok"] = True
            break
        except Exception as e:
            attempt_info.update(
                {
                    "ok": False,
                    "latency_sec": round(time.time() - t0, 3),
                    "error": str(e)[:400],
                }
            )
            report["attempts"].append(attempt_info)

    return report


def run_command_stream(cmd: List[str], cwd: Path, log_path: Path) -> int:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            logf.write(line)
        return proc.wait()


def find_latest_locomo_summary(model_output_dir: Path) -> Optional[Path]:
    candidates = sorted(
        model_output_dir.glob("locomo_*/summary.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def run_single_model_eval(
    project_root: Path,
    db_root: Path,
    output_root: Path,
    provider: str,
    judge_provider: str,
    judge_model: str,
    top_k: int,
    concurrency: int,
    questions: Optional[int],
    resolved_model: str,
    label: str,
    verbose: bool,
) -> Dict[str, Any]:
    slug = safe_slug(label)
    model_out_dir = output_root / slug
    model_out_dir.mkdir(parents=True, exist_ok=True)

    log_path = model_out_dir / "run.log"
    cmd = [
        sys.executable,
        "-m",
        "eval.runner",
        "--dataset",
        "locomo",
        "--mode",
        "answer",
        "--provider",
        provider,
        "--judge-provider",
        judge_provider,
        "--model",
        resolved_model,
        "--judge-model",
        judge_model,
        "--db-path",
        str(db_root),
        "--top-k",
        str(top_k),
        "--concurrency",
        str(concurrency),
        "--output-dir",
        str(model_out_dir),
    ]
    if questions is not None and questions > 0:
        cmd.extend(["--questions", str(questions)])
    if verbose:
        cmd.append("--verbose")

    started = now_str()
    t0 = time.time()
    print(f"\n[Eval] {label} ({resolved_model})")
    print(f"[Eval] log: {log_path}")
    rc = run_command_stream(cmd=cmd, cwd=project_root, log_path=log_path)
    elapsed = time.time() - t0
    finished = now_str()

    result: Dict[str, Any] = {
        "label": label,
        "resolved_model": resolved_model,
        "status": "ok" if rc == 0 else "failed",
        "return_code": rc,
        "started_at": started,
        "finished_at": finished,
        "elapsed_sec": round(elapsed, 2),
        "output_dir": str(model_out_dir),
        "log_path": str(log_path),
    }

    summary_path = find_latest_locomo_summary(model_out_dir)
    if summary_path and summary_path.exists():
        try:
            summary = json.load(open(summary_path, "r", encoding="utf-8"))
            result["summary_path"] = str(summary_path)
            result["overall_accuracy"] = summary.get("overall_accuracy")
            result["total_questions"] = summary.get("total_questions")
            result["total_correct"] = summary.get("total_correct")
        except Exception as e:
            result["summary_parse_error"] = str(e)[:300]
    else:
        result["summary_path"] = None

    return result


def load_model_specs(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None:
        return DEFAULT_MODEL_SPECS
    with open(path, "r", encoding="utf-8") as f:
        specs = json.load(f)
    if not isinstance(specs, list):
        raise ValueError("Model spec file must be a JSON list.")
    for item in specs:
        if not isinstance(item, dict) or "label" not in item or "aliases" not in item:
            raise ValueError("Each model spec item must contain 'label' and 'aliases'.")
    return specs


def load_model_list_names(model_list_jsonl: Path) -> List[str]:
    names: List[str] = []
    with open(model_list_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                name = row.get("name")
                if isinstance(name, str) and name.strip():
                    names.append(name.strip())
    return names


def align_aliases_with_model_list(
    specs: List[Dict[str, Any]],
    model_list_names: List[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Reorder aliases: names present in model_list.jsonl are tried first."""
    available = set(model_list_names)
    aligned_specs: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []

    for spec in specs:
        label = str(spec["label"])
        aliases = list(spec["aliases"])
        in_list = [a for a in aliases if a in available]
        not_in_list = [a for a in aliases if a not in available]
        reordered = in_list + not_in_list
        aligned_specs.append({"label": label, "aliases": reordered})
        diagnostics.append(
            {
                "label": label,
                "aliases_in_model_list": in_list,
                "aliases_not_in_model_list": not_in_list,
                "has_model_list_match": len(in_list) > 0,
            }
        )
    return aligned_specs, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description="Run sequential multi-model LOCOMO evaluation on Sonnet DB")
    parser.add_argument(
        "--db-root",
        type=str,
        default="faiss_data/locomo/claude-sonnet-4-5",
        help="Sonnet-built memory DB root",
    )
    parser.add_argument(
        "--locomo-data",
        type=str,
        default="data/locomo/locomo_conversations.json",
        help="LoCoMo conversation json path",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="metaai",
        choices=["metaai", "openai", "vllm"],
        help="Provider for answer models",
    )
    parser.add_argument(
        "--judge-provider",
        type=str,
        default="metaai",
        choices=["metaai", "openai", "vllm"],
        help="Provider for judge model",
    )
    parser.add_argument("--judge-model", type=str, default="gpt-4.1")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--questions", type=int, default=0, help="<=0 means all questions")
    parser.add_argument("--probe-retries", type=int, default=2)
    parser.add_argument(
        "--model-spec-file",
        type=str,
        default="",
        help="Optional model spec json file. If empty, use built-in target list",
    )
    parser.add_argument(
        "--model-list-jsonl",
        type=str,
        default="",
        help="Optional model list jsonl path for strict name alignment (e.g. /apdcephfs/group/40150/ZhiyuShen/model_list.jsonl)",
    )
    parser.add_argument(
        "--only-label",
        action="append",
        default=[],
        help="Run only selected model labels (repeatable), e.g. --only-label 'GPT-4.1'",
    )
    parser.add_argument(
        "--no-wait-for-build",
        action="store_true",
        help="Do not wait for DB build completion; fail if DB is incomplete",
    )
    parser.add_argument(
        "--skip-build-check",
        action="store_true",
        help="Skip DB build completeness check (useful for probe-only stage)",
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--wait-timeout-hours", type=float, default=8.0)
    parser.add_argument(
        "--skip-unavailable-models",
        action="store_true",
        help="Continue evaluation when some target models are unavailable",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="logs/multimodel_eval",
        help="Root output directory",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Only resolve/probe model aliases, do not run evaluation",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    db_root = (project_root / args.db_root).resolve() if not Path(args.db_root).is_absolute() else Path(args.db_root)
    locomo_data = (project_root / args.locomo_data).resolve() if not Path(args.locomo_data).is_absolute() else Path(args.locomo_data)
    run_root = (project_root / args.output_root / f"run_{ts_str()}").resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    if not locomo_data.exists():
        raise FileNotFoundError(f"LoCoMo data not found: {locomo_data}")

    conversations = load_locomo_conversations(locomo_data)
    conv_ids = conv_ids_from_data(conversations)

    # 1) Build monitoring / waiting
    status_json = run_root / "build_status.json"
    if not args.skip_build_check:
        if args.no_wait_for_build:
            status = collect_build_status(db_root=db_root, conv_ids=conv_ids)
            with open(status_json, "w", encoding="utf-8") as f:
                json.dump(status, f, ensure_ascii=False, indent=2)
            if status["pending"] > 0:
                raise RuntimeError(
                    f"DB build incomplete: ready={status['ready']}/{status['total']}, pending={status['pending_ids']}"
                )
            print(f"[Build] Ready: {status['ready']}/{status['total']}")
        else:
            status = wait_for_build_ready(
                db_root=db_root,
                conv_ids=conv_ids,
                poll_seconds=args.poll_seconds,
                timeout_hours=args.wait_timeout_hours,
                status_json_path=status_json,
            )
            print(f"[Build] Completed: {status['ready']}/{status['total']}")
    else:
        status = {
            "time": now_str(),
            "db_root": str(db_root),
            "total": len(conv_ids),
            "ready": None,
            "pending": None,
            "progress_percent": None,
            "ready_ids": [],
            "pending_ids": [],
            "skipped": True,
        }
        with open(status_json, "w", encoding="utf-8") as f:
            json.dump(status, f, ensure_ascii=False, indent=2)
        print("[Build] Check skipped (--skip-build-check)")

    # 2) Resolve model aliases
    specs = load_model_specs(Path(args.model_spec_file) if args.model_spec_file else None)
    model_list_diagnostics: List[Dict[str, Any]] = []
    if args.model_list_jsonl:
        model_list_path = Path(args.model_list_jsonl)
        if not model_list_path.exists():
            raise FileNotFoundError(f"model_list_jsonl not found: {model_list_path}")
        model_list_names = load_model_list_names(model_list_path)
        specs, model_list_diagnostics = align_aliases_with_model_list(specs, model_list_names)
        with open(run_root / "model_list_alignment.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_list_jsonl": str(model_list_path),
                    "total_model_list_names": len(model_list_names),
                    "diagnostics": model_list_diagnostics,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[Model List] Loaded {len(model_list_names)} names from {model_list_path}")
        for d in model_list_diagnostics:
            print(
                f"[Model List] {d['label']}: "
                f"matched={d['aliases_in_model_list'] if d['aliases_in_model_list'] else 'NONE'}"
            )
    if args.only_label:
        allow = {x.strip() for x in args.only_label if str(x).strip()}
        specs = [s for s in specs if str(s.get("label", "")) in allow]
        if not specs:
            raise RuntimeError(f"No model spec matches --only-label values: {sorted(allow)}")
    probe_reports: List[Dict[str, Any]] = []
    resolved: List[Dict[str, str]] = []

    print("\n[Probe] Resolving model aliases via live API calls...")
    for spec in specs:
        label = str(spec["label"])
        aliases = list(spec["aliases"])
        report = probe_model_aliases(
            label=label,
            aliases=aliases,
            provider=args.provider,
            max_probe_retries=args.probe_retries,
        )
        probe_reports.append(report)
        if report["ok"]:
            resolved.append({"label": label, "resolved_model": str(report["resolved_model"])})
            print(f"  [OK] {label} -> {report['resolved_model']}")
        else:
            print(f"  [FAIL] {label} -> no valid alias")

    with open(run_root / "probe_report.json", "w", encoding="utf-8") as f:
        json.dump(probe_reports, f, ensure_ascii=False, indent=2)
    with open(run_root / "resolved_models.json", "w", encoding="utf-8") as f:
        json.dump(resolved, f, ensure_ascii=False, indent=2)

    if not resolved:
        raise RuntimeError("No target model could be resolved. See probe_report.json")

    unavailable = [r["label"] for r in probe_reports if not r["ok"]]
    if unavailable and not args.skip_unavailable_models:
        raise RuntimeError(
            "Some target models are unavailable and --skip-unavailable-models is not set: "
            + ", ".join(unavailable)
        )

    if args.probe_only:
        print("\n[Probe] probe-only mode finished.")
        print(f"[Probe] run_root: {run_root}")
        return 0

    # 3) Sequential answer-mode evaluation
    run_summary: Dict[str, Any] = {
        "started_at": now_str(),
        "project_root": str(project_root),
        "db_root": str(db_root),
        "locomo_data": str(locomo_data),
        "provider": args.provider,
        "judge_provider": args.judge_provider,
        "judge_model": args.judge_model,
        "top_k": args.top_k,
        "concurrency": args.concurrency,
        "resolved_models": resolved,
        "unavailable_models": unavailable,
        "results": [],
    }

    for item in resolved:
        result = run_single_model_eval(
            project_root=project_root,
            db_root=db_root,
            output_root=run_root / "per_model",
            provider=args.provider,
            judge_provider=args.judge_provider,
            judge_model=args.judge_model,
            top_k=args.top_k,
            concurrency=args.concurrency,
            questions=args.questions if args.questions > 0 else None,
            resolved_model=item["resolved_model"],
            label=item["label"],
            verbose=args.verbose,
        )
        run_summary["results"].append(result)
        with open(run_root / "run_summary.json", "w", encoding="utf-8") as f:
            json.dump(run_summary, f, ensure_ascii=False, indent=2)

    run_summary["finished_at"] = now_str()
    with open(run_root / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)

    print("\n============================================================")
    print("Multi-model evaluation finished")
    print(f"run_root: {run_root}")
    for res in run_summary["results"]:
        print(
            f"- {res['label']}: status={res['status']}, "
            f"acc={res.get('overall_accuracy')}, elapsed={res['elapsed_sec']}s"
        )
    print("============================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
