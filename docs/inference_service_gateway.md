# Inference Service Gateway Design

## Context

`atom-viz` currently has a React frontend and a lightweight FastAPI backend in
`atom-viz/api.py`. The frontend can display and edit atom reconfiguration plans,
call backend cost/grouping endpoints, generate random boards, and request
baseline plans. The online `Generate Plan` path supports `kouhei` and `smt`,
but the `mcts` branch is intentionally not implemented yet and returns `501`.

Separately, the core project already has working checkpoint inference through
`pipeline_eval.py` and the `neutral_atoms` / `fast_mcts` stack. A trained
checkpoint can be loaded on CUDA, run MCTS, and converted into atom-viz JSON via
`neutral_atoms.experiment.game_to_solution`.

The missing piece is not basic model inference. The missing piece is deployment
architecture: the UI-facing backend should not assume that the high-performance
inference machine is online, local, or co-located with the low-cost cloud server
hosting the web app.

## Recommendation

Use `atom-viz/api.py` as a stable gateway and proxy `mcts` plan-generation
requests to a separate inference service.

The recommended production shape is:

```text
Browser
  |
  | HTTPS
  v
Low-cost cloud server
  - serves atom-viz frontend
  - runs lightweight FastAPI gateway
  - computes cheap validation/cost endpoints locally
  - forwards MCTS requests when an inference service is configured
  |
  | private HTTPS / VPN / Tailscale / SSH tunnel
  v
Inference service
  - runs on GPU workstation, GPU cloud instance, or HPC-adjacent node
  - loads checkpoint(s)
  - runs MCTS / neural inference
  - returns atom-viz-compatible plans
```

This keeps the browser API stable while allowing the inference backend to be
started only when needed. If the inference service is offline, the gateway can
return a clear, user-facing message instead of failing with a generic network
or CORS error.

## Why Gateway Forwarding Is Preferable

The frontend should not call the GPU inference service directly.

Reasons:

- The browser should only know about one public API origin: the cloud backend.
- The cloud backend can hide private inference-service addresses, tokens, and
  network topology.
- CORS and TLS stay simple.
- The gateway can normalize errors into UI-friendly messages.
- The gateway can enforce request limits before expensive GPU work starts.
- The gateway can support multiple inference providers later without changing
  frontend contracts.
- The inference service can be offline most of the time without breaking the
  rest of the app.

The cloud backend should also not run heavyweight MCTS locally in production.
It may keep cheap endpoints such as `/compute`, `/compute_all_layers`,
`/random_board`, `kouhei`, and possibly small SMT requests. Checkpoint-backed
MCTS should be delegated.

## Service Responsibilities

### React Frontend

The frontend remains responsible for interaction and visualization:

- Maintain and edit `{board, circuit, plan}` state.
- Show plan generation controls for `kouhei`, `smt`, and `mcts`.
- For `mcts`, collect or expose:
  - checkpoint/model id, if multiple models are available;
  - simulation count;
  - optional backend mode, such as `classic` or `fast`;
  - optional device preference, such as `auto`, `cuda`, or `cpu`.
- Display gateway error messages directly:
  - inference service offline;
  - model unavailable;
  - request too large;
  - timeout;
  - internal inference failure.
- Keep generated plans as separate candidates. A returned plan must not overwrite
  the editable `{board, circuit, plan}` state unless the user explicitly applies
  that candidate.
- Allow the user to select up to two ready candidates and inspect them in a
  read-only upper/lower comparison view at the same layer index.

The frontend should not know whether the gateway computed a plan locally or
forwarded it.

### Gateway Backend

The `atom-viz` FastAPI backend is the stable public API:

- Keep existing cheap endpoints local:
  - `GET /`
  - `POST /compute`
  - `POST /compute_all_layers`
  - `POST /random_board`
  - `POST /generate_plan` for `kouhei` and `smt`
- For `POST /generate_plan` with `method="mcts"`:
  - validate the request shape;
  - enforce maximum board size, number of qubits, layers, gates, and simulation
    count;
  - check whether inference forwarding is configured;
  - call the inference service;
  - translate inference-service responses into the existing
    `GeneratePlanResponse`;
  - return clear HTTP status codes and messages when unavailable.

The gateway should use configuration, not hard-coded paths:

```text
MCTS_INFERENCE_URL=https://inference.example.internal
MCTS_INFERENCE_TOKEN=...
MCTS_DEFAULT_MODEL=v3a01-cycle05
MCTS_DEFAULT_NUM_SIMULATIONS=400
MCTS_REQUEST_TIMEOUT_S=120
MCTS_MAX_NUM_SIMULATIONS=10000
MCTS_MAX_QUBITS=12
MCTS_MAX_LAYERS=3
```

If `MCTS_INFERENCE_URL` is unset, the gateway should return `503 Service
Unavailable` with a precise message such as:

```json
{
  "detail": "MCTS inference service is not configured. Try Kouhei/SMT or start the inference service."
}
```

### Inference Service

The inference service is a separate process and deployment unit:

- Load one or more checkpoints.
- Keep loaded models cached in memory.
- Run MCTS with the requested simulation count and backend.
- Convert `Game` results to atom-viz JSON using `game_to_solution`.
- Return plan cost and optional diagnostics.
- Provide health/model discovery endpoints.
- Fail fast if no CUDA device is available when a CUDA-only model is requested.

The service should be allowed to run only when needed. This supports:

- a local GPU workstation;
- a manually started GPU cloud instance;
- a scheduled GPU node;
- a future job queue if inference latency becomes too long for a direct HTTP
  request.

## API Contract

The gateway should keep the existing frontend-facing endpoint:

```http
POST /generate_plan
```

For `method="mcts"`, the request should allow these fields:

```json
{
  "board": {
    "rows": 5,
    "cols": 5,
    "initialAtoms": {
      "0": {"row": 0, "col": 0}
    }
  },
  "circuit": [
    [[0, 1], [2, 3]]
  ],
  "method": "mcts",
  "model_id": "v3a01-cycle05",
  "checkpoint_path": null,
  "num_simulations": 400,
  "backend": "fast",
  "device": "cuda",
  "deterministic": true
}
```

Notes:

- Prefer `model_id` over arbitrary `checkpoint_path` for production. A public
  backend should not allow clients to request arbitrary filesystem paths on the
  inference host.
- `checkpoint_path` can remain useful for local development if guarded by an
  explicit development flag.
- `deterministic=true` should disable root noise and select argmax actions for
  reproducible visualization.

The gateway response should preserve the current `GeneratePlanResponse` shape:

```json
{
  "board": {
    "rows": 5,
    "cols": 5,
    "initialAtoms": {
      "0": {"row": 0, "col": 0}
    }
  },
  "circuit": [
    [[0, 1], [2, 3]]
  ],
  "plan": [
    [
      {
        "atom": 0,
        "from": {"row": 0, "col": 0},
        "to": {"row": 1, "col": 0}
      }
    ]
  ],
  "method": "mcts",
  "elapsed_ms": 1842.0,
  "plan_cost": 17
}
```

The inference service can expose a narrower internal endpoint:

```http
POST /v1/plan
```

with the same planning fields plus any internal options:

```json
{
  "board": {"rows": 5, "cols": 5, "initialAtoms": {}},
  "circuit": [],
  "model_id": "v3a01-cycle05",
  "num_simulations": 400,
  "backend": "fast",
  "device": "cuda",
  "deterministic": true
}
```

The inference-service response should include both the plan and diagnostics:

```json
{
  "board": {},
  "circuit": [],
  "plan": [],
  "model_id": "v3a01-cycle05",
  "checkpoint_run_id": "v3a01",
  "training_steps": 148455,
  "plan_cost": 17,
  "elapsed_ms": 1842.0,
  "device": "cuda",
  "backend": "fast",
  "num_simulations": 400,
  "diagnostics": {
    "avg_mcts_depth": 5.1,
    "root_value_mean": 0.42
  }
}
```

The gateway may drop internal diagnostics initially, or pass them through later
under an optional field.

## Error Semantics

The gateway should translate inference failures into stable HTTP responses:

| Scenario | Gateway status | User-facing meaning |
| --- | ---: | --- |
| Inference URL not configured | `503` | MCTS inference is not configured on this deployment. |
| Inference service offline or connection refused | `503` | MCTS inference service is offline. |
| Inference request timeout | `504` | MCTS inference timed out; reduce simulations or try later. |
| Unknown model id | `404` | Requested model is not available. |
| Request exceeds configured limits | `413` or `422` | Problem or simulation count is too large. |
| Inference service reports no CUDA | `503` | Inference worker has no usable GPU. |
| Unexpected inference error | `502` | Inference service failed while processing the request. |

Example gateway response when the GPU service is offline:

```json
{
  "detail": "MCTS inference service is offline. Start the inference worker or choose Kouhei/SMT."
}
```

This is preferable to exposing raw stack traces, low-level connection errors, or
browser-side CORS failures.

## Request Lifecycle

The synchronous lifecycle is:

1. User clicks `Generate Plan` with `MCTS`.
2. Frontend sends `POST /generate_plan` to the cloud gateway.
3. Gateway validates the board/circuit and MCTS parameters.
4. Gateway checks `MCTS_INFERENCE_URL`.
5. Gateway forwards the request to `POST /v1/plan` on the inference service.
6. Inference service loads or reuses the requested model.
7. Inference service constructs the `neutral_atoms.Game`.
8. Inference service runs MCTS using the requested backend/device.
9. Inference service converts the result with `game_to_solution`.
10. Gateway returns the plan using `GeneratePlanResponse`.
11. Frontend stores the returned plan as a candidate and calls
    `/compute_all_layers` for that candidate.
12. Frontend lets the user compare the candidate against greedy/model results,
    or explicitly apply one candidate to the editable plan.

This is appropriate while inference takes seconds to a couple of minutes.

If inference regularly takes longer than HTTP timeouts, move to an async job
model:

```text
POST /generate_plan_jobs -> {job_id}
GET /generate_plan_jobs/{job_id} -> queued | running | complete | failed
```

The synchronous gateway should be implemented first because it matches the
current frontend flow and keeps the integration small.

## Model Selection

Production should use named models:

```json
{
  "id": "v3a01-cycle05",
  "checkpoint": "/models/pipeline_v3a01/checkpoints/cycle_05.ckpt",
  "map_num": 2,
  "board": "5x5",
  "num_qubits": 12,
  "num_layers": 3,
  "default_backend": "fast",
  "default_num_simulations": 400
}
```

Expose model discovery through:

```http
GET /v1/models
```

Example:

```json
{
  "models": [
    {
      "id": "v3a01-cycle05",
      "run_id": "v3a01",
      "training_steps": 148455,
      "board": "5x5",
      "num_qubits": 12,
      "num_layers": 3
    }
  ]
}
```

The gateway can cache this response briefly and expose it to the frontend later
through a public endpoint such as `GET /mcts_models`.

## Compatibility With Current atom-viz

The current frontend already has most of the necessary shape:

- `GeneratePlanRequest` includes `method`, `checkpoint_path`, and
  `num_simulations`.
- The `Generate Plan` modal already lists `MCTS`.
- Applying a generated plan is already implemented.
- `compute_all_layers` already validates and annotates the returned plan.

The required frontend changes are small:

- Add MCTS-only controls for model id and simulation count.
- Send those fields in `generatePlan(...)`.
- Improve the error display for `503` and `504`.
- Optionally show model metadata and inference diagnostics after success.

The required gateway changes are:

- Replace the current `method == "mcts"` `501` branch with a forwarding call.
- Add configuration for inference URL, token, timeout, and limits.
- Normalize inference errors.
- Keep the existing response schema.

The required inference-service implementation is new, but it can reuse existing
core code:

- `neutral_atoms.config_hpc.get_config`
- `neutral_atoms.config.set_derived_config`
- `neutral_atoms.network.Network`
- `neutral_atoms.game.Game`
- `fast_mcts.backends.get_backend`
- `neutral_atoms.experiment.compute_solution_cost`
- `neutral_atoms.experiment.game_to_solution`

## Security and Operational Constraints

Do not expose arbitrary checkpoint paths in production.

Recommended constraints:

- Only allow model ids from a server-side registry.
- Use a shared service token between gateway and inference service.
- Put inference service behind a private network, VPN, or tunnel.
- Set strict request size and simulation limits.
- Set gateway and inference timeouts.
- Log request id, model id, board shape, simulation count, elapsed time, and
  failure mode.
- Avoid logging full user-submitted circuits if privacy becomes relevant.
- Rate-limit MCTS requests at the gateway.

## Minimal Implementation Plan

### Phase 1: Synchronous Forwarding

Implement the smallest useful version:

- [x] Add `MCTS_INFERENCE_URL`, `MCTS_INFERENCE_TOKEN`, and timeout config.
- [x] Implement `POST /v1/plan` in a new inference-service module.
- [x] Implement gateway forwarding for `method="mcts"`.
- [x] Add frontend controls for `model_id` and `num_simulations`.
- [x] Return `503` when the inference service is not configured or offline.
- [x] Test against `pipelines/pipeline_v3a01/checkpoints/cycle_05.ckpt`.

### Phase 1B: Frontend Candidate Comparison

- [x] Preserve the editable manual plan as the primary working state.
- [x] Store generated greedy/SMT/model plans as independent candidates.
- [x] Keep candidate failures local to the candidate list.
- [x] Allow selection of at most two ready candidates.
- [x] Render selected candidates in a read-only upper/lower layer-inspection
  view.
- [x] Keep both compare panes synchronized to the same layer index.
- [x] Require explicit `Apply` before a generated candidate replaces the editable
  plan.

### Phase 2: Model Registry

- Add `GET /v1/models` on the inference service.
- Add gateway `GET /mcts_models`.
- Let the frontend choose from available model ids.
- Remove production use of raw checkpoint paths.

### Phase 3: Async Jobs

Only add async jobs if synchronous requests become too slow or unreliable:

- `POST /generate_plan_jobs`
- `GET /generate_plan_jobs/{job_id}`
- Optional cancellation.
- Optional persisted result cache keyed by problem hash and model settings.

## Open Questions

- Should the gateway allow local MCTS execution in development mode, or should
  all MCTS requests always go through an inference service?
- What is the maximum acceptable frontend wait time for synchronous MCTS?
- Should the first inference service support only `map_num=2` / 5x5 / 12q /
  3-layer models, or should it generalize immediately?
- Should inference results include raw MCTS diagnostics in the frontend, or only
  the final plan and cost?
- How should model ids be versioned when checkpoints are moved or replaced?

## Decision

Use gateway forwarding for online MCTS plan generation.

The low-cost cloud backend remains the stable public API and cheap computation
host. The high-performance inference service becomes an optional, separately
deployed worker that can be offline without breaking manual editing, greedy
generation, SMT generation, random board generation, or cost visualization.
