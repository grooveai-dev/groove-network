# Groove Network — Repo Guide

## Directory Layout

```
~/Desktop/groove-network/  <- this repo (node + consumer inference code)
~/Desktop/groove-signal/   <- sibling repo (signal/relay service)
```

These are **separate git repos** with separate remotes, tags, and deploy cycles.
Do NOT add the other repo as a second remote. That causes cross-push accidents.

## What This Repo Contains

- `src/node/` — compute node (shard loader, KV cache, forward pass)
- `src/consumer/` — consumer client (pipeline orchestration, P2P, mesh)
- `src/relay/` — relay server (used for local testing)
- `src/common/` — shared utilities (protocol, tensor transfer, P2P, chunking, benchmark)
- `scripts/` — startup scripts, trace analyzer
- `tests/` — unit and integration tests
- `deploy/coturn/` — TURN server config

## Pushing Changes

```bash
cd ~/Desktop/groove-network
git add <files>
git commit -m "description"
git push origin main --follow-tags
```

There is only one remote (`origin` = groove-network). `git push` always does the right thing.

## Version Tags

Nodes auto-update by comparing their config version against the latest git tag. To release:

```bash
git tag -a v0.X.Y -m "v0.X.Y: description"
git push origin main --follow-tags
```

Always use **annotated tags** (`-a`), not lightweight tags. The update system requires them.

## Running Tests

```bash
cd ~/Desktop/groove-network
.venv/bin/python -m pytest tests/ -x -q
```

## Shared Files (protocol.py, tensor_transfer.py)

`src/common/protocol.py` exists in both repos but has **diverged intentionally**:
- Network version: includes mesh/P2P/inference-specific types (TOKEN_RESULT, MESH_CONNECT, etc.)
- Signal version: includes matcher/signal-specific types
- Base message types (ENVELOPE, HEARTBEAT, REGISTER, etc.) are shared

If you change a base message type, update both repos. Network-only or signal-only
types stay in their respective repo.

`src/common/tensor_transfer.py` should stay identical. After changing it in one repo,
copy it to the other.

## Common Mistakes to Avoid

- Do NOT clone groove-signal as a second remote in this directory
- Do NOT push network tags to the signal repo (the signal cron will check out inference code and crash)
- Do NOT run the signal service from this directory — use groove-signal for that

## Key Dependencies

- `optimum-quanto` — required for INT4 quantized KV cache on MPS (Apple Silicon)
- `aiortc` — WebRTC P2P data channels
- `torch` — installed separately by setup.sh (not in requirements.txt to avoid CPU-only override)
