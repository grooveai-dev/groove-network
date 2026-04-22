#!/usr/bin/env python3
"""Analyze and visualize Groove inference trace files.

Usage:
    python scripts/analyze_trace.py                    # latest trace
    python scripts/analyze_trace.py ~/.groove/traces/trace_20260422_143804.jsonl
    python scripts/analyze_trace.py --compare trace1.jsonl trace2.jsonl
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def load_trace(path: str) -> tuple[dict, list[dict]]:
    summary = {}
    hops = []
    with open(path) as f:
        for line in f:
            entry = json.loads(line.strip())
            if entry.get("type") == "summary":
                summary = entry
            else:
                hops.append(entry)
    return summary, hops


def bar(value: float, max_val: float, width: int = 40) -> str:
    if max_val <= 0:
        return ""
    filled = int(value / max_val * width)
    return "\u2588" * filled + "\u2591" * (width - filled)


def analyze(path: str) -> dict:
    summary, hops = load_trace(path)
    if not hops:
        print(f"No trace data in {path}")
        return {}

    stages: dict[int, list[dict]] = {}
    for h in hops:
        s = h.get("stage")
        if s is not None:
            stages.setdefault(s, []).append(h)

    print(f"\n{'=' * 72}")
    print(f"  GROOVE INFERENCE TRACE — {Path(path).name}")
    print(f"{'=' * 72}")

    if summary:
        print(f"\n  Tokens: {summary.get('tokens_generated', '?')}")
        print(f"  TPS:    {summary.get('tps', '?')}")
        print(f"  TTFT:   {summary.get('ttft_ms', '?'):.0f} ms")
        print(f"  P2P:    {summary.get('p2p_sends', 0)} sends  |  Relay: {summary.get('relay_sends', 0)} sends")

    print(f"\n{'─' * 72}")
    print(f"  PER-STAGE BREAKDOWN (averages across all tokens)")
    print(f"{'─' * 72}")

    max_rtt = max((h.get("rtt_ms", 0) for h in hops), default=1)

    for stage_idx in sorted(stages.keys()):
        stage_hops = stages[stage_idx]
        n = len(stage_hops)
        node = stage_hops[0].get("node", "?")
        via = stage_hops[0].get("via", "?")

        avg = lambda k: sum(h.get(k, 0) for h in stage_hops) / n
        avg_rtt = avg("rtt_ms")
        avg_fwd = avg("forward_ms")
        avg_queue = avg("queue_ms")
        avg_ser = avg("serialize_ms")
        avg_send = avg("send_ms")
        avg_wait = avg("wait_ms")
        avg_payload = sum(h.get("payload_bytes", 0) for h in stage_hops) / n
        overhead = avg_wait - avg_fwd - avg_queue

        print(f"\n  Stage {stage_idx}: {node}  ({via})")
        print(f"    RTT:        {avg_rtt:7.1f} ms  {bar(avg_rtt, max_rtt)}")
        print(f"    ├ serialize: {avg_ser:6.1f} ms")
        print(f"    ├ send:      {avg_send:6.1f} ms")
        print(f"    ├ wait:      {avg_wait:6.1f} ms")
        print(f"    │  ├ queue:    {avg_queue:6.1f} ms")
        print(f"    │  ├ forward:  {avg_fwd:6.1f} ms")
        print(f"    │  └ overhead: {overhead:6.1f} ms")
        print(f"    payload:    {avg_payload:7.0f} bytes avg")

    print(f"\n{'─' * 72}")
    print(f"  PER-TOKEN TIMELINE (first 20 tokens)")
    print(f"{'─' * 72}")

    tokens: dict[int, list[dict]] = {}
    for h in hops:
        seq = h.get("seq")
        if seq is not None:
            tokens.setdefault(seq, []).append(h)

    sorted_seqs = sorted(tokens.keys())[:20]
    for seq in sorted_seqs:
        parts = tokens[seq]
        total = sum(p.get("rtt_ms", 0) for p in parts)
        stage_strs = []
        for p in sorted(parts, key=lambda x: x.get("stage", 0)):
            s = p.get("stage", 0)
            fwd = p.get("forward_ms", 0)
            rtt = p.get("rtt_ms", 0)
            via_char = "P" if p.get("via") == "p2p" else "R"
            stage_strs.append(f"s{s}={rtt:.0f}ms(fwd={fwd:.0f}){via_char}")
        print(f"  seq {seq:4d} | {' | '.join(stage_strs)} | total={total:.0f}ms")

    print(f"\n{'─' * 72}")
    print(f"  BOTTLENECK ANALYSIS")
    print(f"{'─' * 72}")

    total_serialize = sum(h.get("serialize_ms", 0) for h in hops)
    total_send = sum(h.get("send_ms", 0) for h in hops)
    total_forward = sum(h.get("forward_ms", 0) for h in hops)
    total_queue = sum(h.get("queue_ms", 0) for h in hops)
    total_wait = sum(h.get("wait_ms", 0) for h in hops)
    total_overhead = total_wait - total_forward - total_queue
    grand_total = total_serialize + total_send + total_wait

    if grand_total > 0:
        pcts = {
            "serialize": (total_serialize / grand_total * 100, total_serialize),
            "send":      (total_send / grand_total * 100, total_send),
            "forward":   (total_forward / grand_total * 100, total_forward),
            "queue":     (total_queue / grand_total * 100, total_queue),
            "overhead":  (total_overhead / grand_total * 100, total_overhead),
        }
        for name, (pct, ms) in sorted(pcts.items(), key=lambda x: -x[1][0]):
            print(f"  {name:12s}  {pct:5.1f}%  ({ms:8.0f} ms total)  {bar(pct, 100, 30)}")

    biggest = max(pcts.items(), key=lambda x: x[1][0]) if grand_total > 0 else None
    if biggest:
        print(f"\n  >> Primary bottleneck: {biggest[0]} ({biggest[1][0]:.0f}%)")

    print(f"\n{'=' * 72}\n")

    return {
        "path": path,
        "tps": summary.get("tps", 0),
        "ttft_ms": summary.get("ttft_ms", 0),
        "tokens": summary.get("tokens_generated", 0),
        "bottleneck": biggest[0] if biggest else "unknown",
    }


def compare(paths: list[str]) -> None:
    results = []
    for p in paths:
        results.append(analyze(p))

    if len(results) < 2:
        return

    print(f"{'=' * 72}")
    print(f"  COMPARISON")
    print(f"{'=' * 72}")

    for r in results:
        name = Path(r["path"]).stem
        print(f"  {name:30s}  TPS={r['tps']:6.1f}  TTFT={r['ttft_ms']:6.0f}ms  tokens={r['tokens']}  bottleneck={r['bottleneck']}")
    print()


def find_latest_trace() -> str | None:
    trace_dir = os.path.expanduser("~/.groove/traces")
    if not os.path.isdir(trace_dir):
        return None
    files = sorted(Path(trace_dir).glob("trace_*.jsonl"))
    return str(files[-1]) if files else None


def main():
    if len(sys.argv) < 2:
        path = find_latest_trace()
        if path is None:
            print("No trace files found. Run an inference to generate one.")
            print("Traces are written to ~/.groove/traces/")
            sys.exit(1)
        analyze(path)
    elif sys.argv[1] == "--compare":
        if len(sys.argv) < 4:
            print("Usage: analyze_trace.py --compare trace1.jsonl trace2.jsonl [...]")
            sys.exit(1)
        compare(sys.argv[2:])
    else:
        for path in sys.argv[1:]:
            analyze(path)


if __name__ == "__main__":
    main()
