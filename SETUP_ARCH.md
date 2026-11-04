# code-graph-rag on Arch (atxpc / 192.168.1.61)

## Location
`/home/user/code-graph-rag` — https://github.com/vitali87/code-graph-rag (v0.0.717)

## Tooling
- `uv` in `~/.local/bin`
- venv: `~/code-graph-rag/.venv`
- Docker + docker-compose 5.5.0
- Stack compose: `~/.cgr/docker-compose.yaml`
- Env: `~/code-graph-rag/.env` — both `ORCHESTRATOR_PROVIDER` and `CYPHER_PROVIDER` are `codex`; both models are `gpt-5.6-luna` with low reasoning effort. The local Codex CLI uses its existing subscription login for hosted inference. No model API server or OpenAI API key is needed for these roles.
- Embeddings: Arch cluster `http://127.0.0.1:18190/v1`, model `embedding` (BGE-M3 FP32 CPU), 1024 dimensions. Keep both `OPENAI_EMBEDDING_DIMENSIONS` and `QDRANT_VECTOR_DIM` at `1024`; collection `code_embeddings_bge_m3_arch_1024` is local at `http://127.0.0.1:6333`.
- MCP: `TARGET_REPO_PATH=/home/user/watermelon-mania` and `MCP_PROJECT_NAME=watermelon-mania` select the existing index. Set these together when changing repositories. An unset `MCP_PROJECT_NAME` uses the CLI's default path-hashed name; CLI questions about an existing custom index also need `--project-name`.

## Distcc (for tree-sitter / native builds)
```bash
export PATH="$HOME/.local/bin:/usr/lib/ccache/bin:/usr/local/bin:/usr/bin:/bin"
export DISTCC_HOSTS="localhost/4"
export CCACHE_PREFIX=distcc
export CC=gcc
export CXX=g++
```

## Daily use
```bash
cd ~/code-graph-rag
export PATH="$HOME/.local/bin:$PATH"
uv run cgr daemon up          # Memgraph :7687 + Qdrant :6333
uv run cgr doctor
uv run cgr start --repo-path /path/to/repo --update-graph
# offline protobuf index only:
uv run cgr index --repo-path /path/to/repo -o ./index-out
uv run cgr stats
uv run cgr daemon down
```

## CGR repair verified 2026-09-19

Prepared → `git status --short` showed an already dirty worktree. Existing changes were preserved; task-only backups and review evidence are in `/tmp/cgr-repair-sep19/`. No dependencies or global Codex configuration were changed.

Observed → `cgr status` reported both databases reachable and MCP `list_projects` returned three projects. A direct Spark request failed with `model is not supported when using Codex with a ChatGPT account`. Its JSON stdout error was previously discarded. Qdrant also contained 3,467 Watermelon vectors whose graph node identity no longer matched after incremental updates.

Model choice → both roles now use `gpt-5.6-luna` via Codex subscription login, with low reasoning effort. Official subscription rates list Luna at 5 input / 0.5 cached input / 30 output credits per million tokens, the lowest listed general-purpose option: https://learn.chatgpt.com/docs/pricing . Actual account requests succeeded. The existing local Qwen2.5-Coder-7B Q4_K_M fits this RTX 3050 8 GB (about 6.2 GB used by the server), but a two-query probe returned a wrong, unscoped Module-to-Module CALLS query for function callers. It was not selected for CGR; its service was left unchanged.

Changes → `.env`, `codebase_rag/providers/codex.py`, `codebase_rag/prompts.py`, `codebase_rag/vector_store.py`, `codebase_rag/graph_updater.py`, `codebase_rag/tests/test_codex_provider.py`, `codebase_rag/tests/test_prune_stale_embeddings.py`, and this file. The provider preserves failure output. Prompts include both callable labels and require resolving exact source names. Each Qdrant embedding pass prunes obsolete IDs within its project after successful replacement writes, skips cleanup on embedding/provider or storage failure, handles empty projects, and preserves other projects. A collection snapshot was downloaded to `/tmp/cgr-repair-sep19/code_embeddings_bge_m3_1024-7081346559335260-2026-09-19-13-37-45.snapshot` before the one-time cleanup removed 3,467 stale points. The graph was not rebuilt.

Verified → this command returned **118 passed, 5 skipped**:

```bash
.venv/bin/pytest -q codebase_rag/tests/test_codex_provider.py codebase_rag/tests/test_prune_stale_embeddings.py codebase_rag/tests/test_graph_updater_embeddings.py codebase_rag/tests/test_semantic_search.py codebase_rag/tests/test_semantic_search_project_filter.py codebase_rag/tests/test_skip_embeddings.py codebase_rag/tests/test_reconcile_embeddings.py codebase_rag/tests/test_llm_service_unit.py codebase_rag/tests/test_mcp_query_and_index.py codebase_rag/tests/test_prompt_tool_names.py
```

Live verification after `systemctl --user restart cgr-mcp.service`:
- MCP `query_code_graph` for callers of `GameManager.start_shift`: four callers, correct project filters and caller-to-callee direction, cross-checked against current source.
- MCP `semantic_search("start a shift in watermelon-mania", top_k=3)`: three matches, `GameManager.start_shift` ranked first.
- MCP `ask_agent`: retrieved source, correctly explained `if is_shift_active: return`, and listed the four callers.
- Direct collection/graph identity audit: **0 stale vectors**, 1,383 total at audit time (an existing embedding job was still adding current vectors).
- Ruff on the changed Python files and `git diff --check`: passed.

Independent review → `codex review` identified cleanup ordering and empty-project client cleanup concerns. Both were addressed: pruning follows successful writes, and the empty-project branch closes the Qdrant client in `finally`, including on errors. Regression tests cover both. Review logs are in `/tmp/cgr-repair-sep19/{final-review,revised-review}.log`; the reviewer's sandbox could not complete Codex subprocess tests, which passed in the main test run. A live unsupported-model probe also verified that the actual JSON error now reaches the caller.

Limits → cleanup targets the deployed Qdrant backend; Milvus behavior is unchanged. Pruning scans collection payloads once per indexing pass. Existing graph coverage and dynamic-call resolution limits still apply. A separately started Watermelon embedding job was left running; no new indexing job was launched.

## Spark switch verified 2026-09-10

Prepared → preserved the already dirty worktree and added the `codex` provider using the installed CLI and Pydantic AI. Both CGR roles select Spark; MCP explicitly selects the existing `watermelon-mania` index. No dependencies were added. Global `~/.codex/AGENTS.md` now makes CGR the default code-context check, followed by source inspection and tests.

Changed files: `.env`, `codebase_rag/config.py`, `codebase_rag/constants/providers.py`, `codebase_rag/providers/{base,codex}.py`, `codebase_rag/mcp/{server,tools}.py`, `codebase_rag/tests/test_{codex_provider,mcp_startup_key_validation,mcp_query_and_index}.py`, this document, and `~/.codex/AGENTS.md`.

Observed → `cgr status` reports Memgraph and Qdrant reachable on Arch. `curl --connect-timeout 3 --max-time 10 -sS http://127.0.0.1:6333/collections/code_embeddings_bge_m3_1024` reports `green`, `1053` points, and vector size `1024`. The previous incorrectly sized collection was empty; only that empty collection was recreated before refreshing both existing project indexes.

Verified → this focused command completed with **146 passed, 1 warning** (an unawaited `_query_with_errlog` coroutine warning in the existing suite):

```bash
env OLLAMA_BASE_URL=http://localhost:11434 .venv/bin/pytest -q \
  codebase_rag/tests/test_codex_provider.py \
  codebase_rag/tests/test_provider_classes.py \
  codebase_rag/tests/test_mcp_startup_key_validation.py \
  codebase_rag/tests/test_mcp_query_and_index.py \
  codebase_rag/tests/test_mcp_update_and_search.py \
  codebase_rag/tests/test_mcp_crash_correlation.py \
  codebase_rag/tests/test_llm_service_unit.py
```

After the final transport-prompt clarification, `.venv/bin/pytest -q codebase_rag/tests/test_codex_provider.py` completed with **4 passed**. Ruff checks on the changed Python files and `git diff --check` passed. Independent source review found no remaining blockers.

Live MCP `ask_agent` called source retrieval and graph querying through Spark, correctly reported the `is_shift_active` guard in `GameManager.start_shift`, and returned two indexed direct callers. These were cross-checked with current source. MCP `semantic_search` returned three real matches through the OPI embedding endpoint. Graph results still require source checks: for example, the current GDScript index does not resolve the stockroom's dynamic `gm.start_shift()` call to GameManager correctly.

The CLI path was also checked with `cgr --quiet start --repo-path /home/user/wails-of-whitby-ue5 --project-name wails-of-whitby-ue5 --no-sync --no-start-stack --ask-agent 'Read-only: use query_graph to list up to three C++ class qualified names from wails-of-whitby-ue5. Do not edit files or run shell commands. Be concise.'` It exited successfully and returned `FDistrictAtmosphereOverride`, `FCharacterProfile`, and `FDocumentData` from that project's graph.

Spark uses the existing Codex login and its usage limits. Each model turn launches one read-only CLI process with a 120-second timeout; CGR executes the requested CGR tools. Native command tools are disabled, and developer instructions keep embedded calls in the model role while preserving the existing Codex home/login.

## Verified 2026-08-22
- `uv sync --extra treesitter-full --extra test --group dev` OK (grammars built)
- Unit tests: **7484 passed**, 184 skipped, 2 failed (qdrant/dart optional), 3 errors (optional deps)
- Offline index smoke OK
- Memgraph graph update on demo repo: **13 nodes / 18 relationships**
- Doctor: 7/9 (only optional Gemini/OpenAI keys missing)

## Arch-only CGR test profile — 2026-10-21

This separate profile uses `http://127.0.0.1:8001/mcp` and the
`ai-cluster-blueprint` index at
`/home/user/projects/ai-cluster/ai-cluster-blueprint`. The legacy port-8000
Watermelon profile and `.env` remain unchanged: **that legacy profile still uses
hosted query generation and OPI embeddings; do not use it for private tests.**

Start the local profile:

```sh
systemctl --user start cgr-local-chat cgr-local-embed cgr-local-mcp
.venv/bin/python deploy/check_local_cgr.py
```

For scoped CLI questions, use the local launcher rather than bare `cgr`:

```sh
.venv/bin/python deploy/local_cgr.py start \
  --repo-path /home/user/projects/ai-cluster/ai-cluster-blueprint \
  --project-name ai-cluster-blueprint --no-sync --no-start-stack \
  --ask-agent 'Read-only: explain the parent process watcher using the source.'
```

Both query generation and the CGR assistant use the existing pinned Qwen3-4B
Q4_K_M through an authenticated loopback llama.cpp server, with a 16,384-token
window, temperature zero and a 768-token reasoning budget. Embeddings use the existing pinned Nomic v1.5
Q4_K_M, 768 dimensions, and distinct `search_query:`/`search_document:` prefixes.
The new `code_embeddings_nomic_v15_q4_768` collection preserves the old BGE
collection. MCP semantic search now passes the selected project to retrieval.
The model files were copied into `.tmp/local-cgr/` and their hashes matched the
approved cluster manifests; no model was downloaded. Keep that directory: it
contains the service models and the private shared API key. No dependencies or
global Codex settings were changed.

This is a CPU functional baseline, not a production worker qualification. Chat
has four threads, 400% CPU quota, 8 GiB memory cap and nice 10; embeddings have two
threads, 100% quota, 2 GiB cap and nice 10. CUDA is hidden and GPU offload disabled.
The embedding backend enforces 2,048 input tokens despite the configured larger
context; inputs are bounded to 1,600 characters with the existing explicit
source-elision marker. Retrieval quality for long functions needs separate
measurement. Full source remains available through source-reading tools.
Graph results must still be checked against current source.

The endpoint currently exposes the standard trusted-host CGR toolset, including
mutation tools. The live smoke calls only read-only tools. Before untrusted worker
integration, enforce a tool allowlist and repository authorization in the worker
bridge; project filtering is retrieval correctness, not a security boundary.
This configuration has no hosted inference fallback; it is not an OS-enforced
network isolation sandbox. Services are started for testing, not enabled at boot.

Prepared → loaded worker agreements and Ponytail; `git status --short` showed an
already heavily modified worktree. Task-only baselines are preserved under
`.tmp/local-cgr/baseline/`. `cgr status` found both local databases reachable;
`list_projects` initially had no AI Cluster/CGR implementation index, so targeted
source inspection was used before changes. The AI Cluster graph was added without
clearing existing projects.

Observed → the original hosted configuration was preserved. Initial embedding
requests failed because the backend's effective input limit was 2,048 tokens;
bounding embedding excerpts resolved that failure. Two initial caller queries
failed (invalid alias, then a comment rejected by safety validation and wrong
return variable). Those failures remain in `.tmp/local-cgr/mcp-smoke*.err`.
The shared prompt now explains caller direction/return aliases and prohibits
comments; query safety validation was not weakened.

Verified → focused command:

```sh
.venv/bin/pytest -q \
  codebase_rag/tests/test_local_embedding_prefixes.py \
  codebase_rag/tests/test_embedder_openai.py \
  codebase_rag/tests/test_mcp_update_and_search.py \
  codebase_rag/tests/test_semantic_search.py \
  codebase_rag/tests/test_semantic_search_project_filter.py \
  codebase_rag/tests/test_prompt_tool_names.py \
  codebase_rag/tests/test_llm_service_unit.py
```

Result: **98 passed, 2 skipped, 1 warning**, 3.90 seconds; full output:
`.tmp/local-cgr/tests-final.log`. The warning is the existing unawaited
`_query_with_errlog` coroutine warning. Ruff on task files and `git diff --check`
passed. `systemctl --user is-active cgr-local-chat cgr-local-embed cgr-local-mcp`
returned three `active` lines. Unauthenticated `/v1/models` requests to both
inference ports returned 401. Vector reconciliation confirmed all 99 expected
embeddings. MCP semantic search ranked `cluster.parent_watch.watch` first for
"watch parent process and stop child when parent exits", with two other results
from the same project. The local CGR assistant successfully called its graph tool
and listed three real API function symbols.

Independent reviews of embedding/project-filter changes and deployment files
found no actionable defects (`.tmp/local-cgr/review.log` and `review-final.log`).
Their restricted environments could not verify running services; the live checks
above were run separately.

Final live result → `.venv/bin/python deploy/check_local_cgr.py` completed exit 0
with the reasoning-enabled configuration; output is
`.tmp/local-cgr/mcp-smoke-reasoning.json`. The model generated a bounded,
project-scoped `caller-[:CALLS]->callee` query returning the caller. This cold CPU
request took **217.4 seconds** (99.0 seconds prefill, 118.4 seconds generation).
Thinking-disabled caller probes failed even after the prompt clarification; their
results are retained. This is one functional smoke, not a model success rate.
The separate prompt review (`prompt-review.log`) found no actionable regressions.

Source cross-check found a graph precision limitation: `parent_watch.watch`
really calls `identity`, but returned candidates such as `cluster.process.bind`
do not. Do not treat the 14 returned rows as 14 verified callers. The smoke proves
transport, retrieval and inclusion of a known caller, not exact graph precision.
Worker evaluations must require source verification and use source-derived gold
answers. No graph-parser rewrite or production worker promotion was performed.

Task files: `codebase_rag/config.py`, `embedder.py`, `mcp/tools.py`, `prompts.py`,
`tools/semantic_search.py`, `tests/test_mcp_update_and_search.py`, new
`tests/test_local_embedding_prefixes.py`, `deploy/local_cgr.py`,
`deploy/check_local_cgr.py`, the three `deploy/cgr-local-*.service` units, and this
setup record. Pre-existing changes in these files and elsewhere were preserved.


## Arch cluster embedding cutover — 2026-10-25

CGR now uses the authenticated cluster CPU BGE-M3 endpoint, alias `embedding`,
1024dimensions, empty prefixes and batch size1. API key is private in.env;
permission0600. Collection: `code_embeddings_bge_m3_arch_1024`. The old OPI
collection remains intact;2154points were copied after eight vector comparisons
and three scoped top5-hit-set comparisons passed. This is compatible BGE-M3
precision migration, not fresh re-embedding or a graph rebuild.

Services: `ai-cluster-embedding.service` (enabled gateway),
`ai-cluster-bge-embedding-model.service` (on-demand resident CPU model). The CGR
unit drop-in `deploy/cgr-embedding.conf` starts the gateway dependency. Model
limits:8192native tokens per request in total,1024dimensions; CPU2threads/nice10,
8GiB/no swap. Requests have real cluster job IDs/leases; model weights remain
resident between requests and unload on workstation/pressure vetoes.

Verified live semantic search `start a shift` includes
`watermelon-mania.project.autoloads.GameManager.start_shift`; fresh requests were
observed as succeeded real embedding jobs with released leases.

Rollback: restore the private backup at
`/home/user/projects/ai-cluster/ai-cluster-blueprint/.runtime/cgr-embedding-cutover-20261025/cgr.env.before`
to this repository.env, keeping0600, remove only the new
`~/.config/systemd/user/cgr-mcp.service.d/embedding.conf` symlink, daemon-reload
and restart cgr-mcp. Stop/disable the embedding gateway if abandoning this
deployment; PartOf stops its model. The old collection/provider were retained.
LightRAG configuration is unchanged and deferred until BC250.
