#!/usr/bin/env python3
"""Analyze and visualize Groove inference trace files.

Usage:
    python scripts/analyze_trace.py                    # latest trace
    python scripts/analyze_trace.py ~/.groove/traces/trace_20260422_143804.jsonl
    python scripts/analyze_trace.py --compare trace1.jsonl trace2.jsonl
    python scripts/analyze_trace.py --benchmark         # compare all traces
    python scripts/analyze_trace.py --label "2-node P2P"  # analyze latest with label
    python scripts/analyze_trace.py --last 5            # compare last N runs
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def load_trace(path: str) -> tuple:
    summary = {}
    hops = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("type") == "summary" or "tps" in entry and "node_count" in entry:
                summary = entry
            else:
                hops.append(entry)
    return summary, hops


def bar(value: float, max_val: float, width: int = 40) -> str:
    if max_val <= 0:
        return ""
    filled = int(min(value / max_val, 1.0) * width)
    return "\u2588" * filled + "\u2591" * (width - filled)


def fmt_mb(mb: int) -> str:
    if mb >= 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{mb} MB"


def analyze(path: str, label: str = "") -> dict:
    summary, hops = load_trace(path)
    if not hops:
        print(f"No trace data in {path}")
        return {}

    stages = {}
    for h in hops:
        s = h.get("stage")
        if s is not None:
            stages.setdefault(s, []).append(h)

    title = label or Path(path).name
    print(f"\n{'=' * 78}")
    print(f"  GROOVE INFERENCE TRACE \u2014 {title}")
    print(f"{'=' * 78}")

    if summary:
        model = summary.get("model", "?")
        nc = summary.get("node_count", "?")
        print(f"\n  Model:  {model}")
        print(f"  Nodes:  {nc}")
        print(f"  Tokens: {summary.get('tokens_generated', '?')}")
        tps = summary.get("tps", 0)
        print(f"  TPS:    {tps}")
        ttft = summary.get("ttft_ms", 0)
        if ttft:
            print(f"  TTFT:   {ttft:.0f} ms")
        p2p = summary.get("p2p_sends", 0)
        relay = summary.get("relay_sends", 0)
        total_sends = p2p + relay
        p2p_pct = (p2p / total_sends * 100) if total_sends > 0 else 0
        print(f"  P2P:    {p2p} sends ({p2p_pct:.0f}%)  |  Relay: {relay} sends")

    # ── NODE RESOURCES ──
    node_resources = summary.get("node_resources", [])
    pipeline = summary.get("pipeline", [])

    if pipeline or node_resources:
        print(f"\n{'─' * 78}")
        print(f"  NODE FLEET")
        print(f"{'─' * 78}")

    if pipeline:
        for i, node in enumerate(pipeline):
            nid = node.get("node_id", "?")
            dev = node.get("device", "?")
            gpu = node.get("gpu_model", "")
            layers = node.get("layers", [None, None])
            score = node.get("score")
            layer_str = f"layers [{layers[0]},{layers[1]})" if layers[0] is not None else ""
            score_str = f"  score={score:.1f}" if score else ""
            print(f"  Stage {i}: {nid}  {dev} {gpu}  {layer_str}{score_str}")

    if node_resources:
        print()
        telem_by_node = {t.get("node_id", ""): t for t in node_resources}
        for nid, t in telem_by_node.items():
            vram_used = t.get("vram_used_mb", 0)
            vram_peak = t.get("vram_peak_mb", 0)
            vram_total = t.get("vram_total_mb", 0)
            ram_used = t.get("ram_used_mb", 0)
            ram_total = t.get("ram_total_mb", 0)
            ram_pct = t.get("ram_pct", 0)
            cpu_pct = t.get("cpu_pct", 0)
            dev = t.get("device", "?")
            gpu = t.get("gpu_model", "")
            layers = t.get("layers", [])

            print(f"  {nid} ({dev} {gpu})")
            if vram_total > 0:
                vram_pct = vram_used / vram_total * 100
                print(f"    VRAM:  {fmt_mb(vram_used)} / {fmt_mb(vram_total)} ({vram_pct:.0f}%)  peak={fmt_mb(vram_peak)}  {bar(vram_pct, 100, 20)}")
            elif vram_used > 0:
                print(f"    VRAM:  {fmt_mb(vram_used)} used  peak={fmt_mb(vram_peak)}")
            if ram_total > 0:
                print(f"    RAM:   {fmt_mb(ram_used)} / {fmt_mb(ram_total)} ({ram_pct:.0f}%)  {bar(ram_pct, 100, 20)}")
            if cpu_pct > 0:
                print(f"    CPU:   {cpu_pct:.0f}%  {bar(cpu_pct, 100, 20)}")
            if layers:
                print(f"    Layers: [{layers[0]}, {layers[1]})")

    # ── Also collect per-hop resource snapshots for live tracking ──
    node_vram_over_time = {}
    for h in hops:
        t = h.get("node_telemetry")
        if not t:
            continue
        nid = t.get("node_id", "")
        vram = t.get("vram_used_mb", 0)
        if nid and vram:
            node_vram_over_time.setdefault(nid, []).append(vram)

    if node_vram_over_time:
        print(f"\n  VRAM over inference:")
        for nid, vrams in node_vram_over_time.items():
            mn, mx = min(vrams), max(vrams)
            avg = sum(vrams) / len(vrams)
            print(f"    {nid}: min={fmt_mb(mn)} avg={fmt_mb(int(avg))} max={fmt_mb(mx)} ({len(vrams)} samples)")

    # ── PER-STAGE BREAKDOWN ──
    print(f"\n{'─' * 78}")
    print(f"  PER-STAGE BREAKDOWN (averages across all tokens)")
    print(f"{'─' * 78}")

    max_rtt = max((h.get("rtt_ms", 0) for h in hops), default=1)

    for stage_idx in sorted(stages.keys()):
        stage_hops = stages[stage_idx]
        n = len(stage_hops)
        node = stage_hops[0].get("node", "?")
        via = stage_hops[0].get("via", "?")

        avg = lambda k: sum(h.get(k, 0) for h in stage_hops) / n  # noqa: E731
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
        print(f"    \u251c serialize: {avg_ser:6.1f} ms")
        print(f"    \u251c send:      {avg_send:6.1f} ms")
        print(f"    \u251c wait:      {avg_wait:6.1f} ms")
        print(f"    \u2502  \u251c queue:    {avg_queue:6.1f} ms")
        print(f"    \u2502  \u251c forward:  {avg_fwd:6.1f} ms")
        print(f"    \u2502  \u2514 overhead: {overhead:6.1f} ms")
        print(f"    payload:    {avg_payload:7.0f} bytes avg")

    # ── PER-TOKEN TIMELINE ──
    print(f"\n{'─' * 78}")
    print(f"  PER-TOKEN TIMELINE (first 20 tokens)")
    print(f"{'─' * 78}")

    tokens = {}
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

    # ── BOTTLENECK ANALYSIS ──
    print(f"\n{'─' * 78}")
    print(f"  BOTTLENECK ANALYSIS")
    print(f"{'─' * 78}")

    total_serialize = sum(h.get("serialize_ms", 0) for h in hops)
    total_send = sum(h.get("send_ms", 0) for h in hops)
    total_forward = sum(h.get("forward_ms", 0) for h in hops)
    total_queue = sum(h.get("queue_ms", 0) for h in hops)
    total_wait = sum(h.get("wait_ms", 0) for h in hops)
    total_overhead = total_wait - total_forward - total_queue
    grand_total = total_serialize + total_send + total_wait

    pcts = {}
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

    biggest = max(pcts.items(), key=lambda x: x[1][0]) if pcts else None
    if biggest:
        print(f"\n  >> Primary bottleneck: {biggest[0]} ({biggest[1][0]:.0f}%)")

    # ── PER-NODE EFFICIENCY ──
    if stages:
        print(f"\n{'─' * 78}")
        print(f"  PER-NODE EFFICIENCY")
        print(f"{'─' * 78}")

        for stage_idx in sorted(stages.keys()):
            stage_hops = stages[stage_idx]
            n = len(stage_hops)
            node = stage_hops[0].get("node", "?")

            fwd_times = [h.get("forward_ms", 0) for h in stage_hops]
            rtt_times = [h.get("rtt_ms", 0) for h in stage_hops]
            avg_fwd = sum(fwd_times) / n
            avg_rtt = sum(rtt_times) / n
            efficiency = (avg_fwd / avg_rtt * 100) if avg_rtt > 0 else 0
            p95_fwd = sorted(fwd_times)[int(n * 0.95)] if n > 1 else fwd_times[0]
            p95_rtt = sorted(rtt_times)[int(n * 0.95)] if n > 1 else rtt_times[0]

            print(f"  {node} (stage {stage_idx})")
            print(f"    avg forward:  {avg_fwd:6.1f} ms  |  p95: {p95_fwd:6.1f} ms")
            print(f"    avg RTT:      {avg_rtt:6.1f} ms  |  p95: {p95_rtt:6.1f} ms")
            print(f"    efficiency:   {efficiency:5.1f}%  (compute / total)")
            print(f"    {bar(efficiency, 100, 40)}")

    print(f"\n{'=' * 78}\n")

    return {
        "path": path,
        "label": label or Path(path).stem,
        "tps": summary.get("tps", 0),
        "ttft_ms": summary.get("ttft_ms", 0),
        "tokens": summary.get("tokens_generated", 0),
        "node_count": summary.get("node_count", len(stages)),
        "model": summary.get("model", "?"),
        "p2p_sends": summary.get("p2p_sends", 0),
        "relay_sends": summary.get("relay_sends", 0),
        "bottleneck": biggest[0] if biggest else "unknown",
        "node_resources": summary.get("node_resources", []),
    }


def compare(paths: list, labels: list = None) -> None:
    results = []
    for i, p in enumerate(paths):
        lbl = labels[i] if labels and i < len(labels) else ""
        results.append(analyze(p, label=lbl))

    results = [r for r in results if r]
    if len(results) < 2:
        return

    print(f"\n{'=' * 78}")
    print(f"  BENCHMARK COMPARISON")
    print(f"{'=' * 78}")

    # Header
    print(f"\n  {'Run':30s}  {'Nodes':>5s}  {'TPS':>7s}  {'TTFT':>8s}  {'Tokens':>6s}  {'P2P%':>5s}  {'Bottleneck':>12s}")
    print(f"  {'-' * 30}  {'-' * 5}  {'-' * 7}  {'-' * 8}  {'-' * 6}  {'-' * 5}  {'-' * 12}")

    for r in results:
        name = r["label"][:30]
        p2p = r["p2p_sends"]
        relay = r["relay_sends"]
        p2p_pct = (p2p / (p2p + relay) * 100) if (p2p + relay) > 0 else 0
        print(
            f"  {name:30s}  {r['node_count']:5d}  {r['tps']:7.2f}  "
            f"{r['ttft_ms']:7.0f}ms  {r['tokens']:6d}  {p2p_pct:4.0f}%  "
            f"{r['bottleneck']:>12s}"
        )

    # TPS comparison bar chart
    max_tps = max(r["tps"] for r in results) if results else 1
    print(f"\n  TPS comparison:")
    for r in results:
        name = r["label"][:20]
        print(f"    {name:20s}  {r['tps']:6.2f}  {bar(r['tps'], max_tps, 35)}")

    # TTFT comparison bar chart
    max_ttft = max(r["ttft_ms"] for r in results) if results else 1
    print(f"\n  TTFT comparison (lower is better):")
    for r in results:
        name = r["label"][:20]
        print(f"    {name:20s}  {r['ttft_ms']:6.0f}ms  {bar(r['ttft_ms'], max_ttft, 35)}")

    # Delta analysis
    if len(results) >= 2:
        base = results[0]
        latest = results[-1]
        tps_delta = latest["tps"] - base["tps"]
        tps_pct = (tps_delta / base["tps"] * 100) if base["tps"] > 0 else 0
        ttft_delta = latest["ttft_ms"] - base["ttft_ms"]
        ttft_pct = (ttft_delta / base["ttft_ms"] * 100) if base["ttft_ms"] > 0 else 0
        direction_tps = "+" if tps_delta >= 0 else ""
        direction_ttft = "+" if ttft_delta >= 0 else ""
        print(f"\n  Delta (first -> last):")
        print(f"    TPS:  {direction_tps}{tps_delta:.2f} ({direction_tps}{tps_pct:.1f}%)")
        print(f"    TTFT: {direction_ttft}{ttft_delta:.0f}ms ({direction_ttft}{ttft_pct:.1f}%)")

    # Node resource comparison
    has_resources = any(r.get("node_resources") for r in results)
    if has_resources:
        print(f"\n  Node resources (final snapshot):")
        for r in results:
            name = r["label"][:25]
            resources = r.get("node_resources", [])
            if not resources:
                continue
            print(f"    {name}:")
            for nr in resources:
                nid = nr.get("node_id", "?")
                dev = nr.get("device", "?")
                gpu = nr.get("gpu_model", "")
                vram = nr.get("vram_used_mb", 0)
                ram_pct = nr.get("ram_pct", 0)
                cpu_pct = nr.get("cpu_pct", 0)
                print(f"      {nid} {dev} {gpu}: VRAM={fmt_mb(vram)} RAM={ram_pct:.0f}% CPU={cpu_pct:.0f}%")

    print()


def find_traces(n: int = 0) -> list:
    trace_dir = os.path.expanduser("~/.groove/traces")
    if not os.path.isdir(trace_dir):
        return []
    files = sorted(Path(trace_dir).glob("trace_*.jsonl"))
    if n > 0:
        files = files[-n:]
    return [str(f) for f in files]


def find_latest_trace() -> str:
    traces = find_traces()
    return traces[-1] if traces else ""


def main():
    args = sys.argv[1:]

    if not args:
        path = find_latest_trace()
        if not path:
            print("No trace files found. Run an inference to generate one.")
            print("Traces are written to ~/.groove/traces/")
            sys.exit(1)
        analyze(path)
        return

    if args[0] == "--benchmark":
        traces = find_traces()
        if len(traces) < 2:
            print("Need at least 2 trace files for benchmark comparison.")
            print(f"Found {len(traces)} trace(s) in ~/.groove/traces/")
            sys.exit(1)
        compare(traces)
        return

    if args[0] == "--last":
        n = int(args[1]) if len(args) > 1 else 5
        traces = find_traces(n)
        if not traces:
            print("No trace files found.")
            sys.exit(1)
        if len(traces) == 1:
            analyze(traces[0])
        else:
            compare(traces)
        return

    if args[0] == "--compare":
        if len(args) < 3:
            print("Usage: analyze_trace.py --compare trace1.jsonl trace2.jsonl [...]")
            sys.exit(1)
        compare(args[1:])
        return

    if args[0] == "--label":
        label = args[1] if len(args) > 1 else ""
        path = args[2] if len(args) > 2 else find_latest_trace()
        if not path:
            print("No trace file found.")
            sys.exit(1)
        analyze(path, label=label)
        return

    if args[0] == "--list":
        traces = find_traces()
        if not traces:
            print("No trace files found.")
            sys.exit(1)
        for t in traces:
            summary, hops = load_trace(t)
            tps = summary.get("tps", 0)
            tokens = summary.get("tokens_generated", 0)
            nc = summary.get("node_count", "?")
            model = summary.get("model", "?")
            name = Path(t).name
            print(f"  {name}  nodes={nc}  tps={tps:.1f}  tokens={tokens}  model={model}")
        return

    for path in args:
        analyze(path)


if __name__ == "__main__":
    main()
