# lc0 Selfplay: GPU Usage & Batching Pipeline

## 1. Hardware

**Distributed volunteer network**, not a centralized cluster.

- Hundreds of contributors run the `lc0` binary on their own machines (wrapped by `lc0-training-client`).
- Supported backends (`src/neural/backends/`): CUDA, cuDNN, OpenCL, Metal, DirectX, SYCL, BLAS/CPU, oneDNN, XLA, TensorFlow.
- Any GPU works (NVIDIA / AMD / Apple / Intel). CPU fallback available but slow.
- Trainer lives in a separate repo (`lczero-training`); this repo only generates games and serves search.
- No sync barrier — workers keep generating games on the current net while the trainer publishes new nets asynchronously.

## 2. GPU? Yes.

Backend is pluggable. `BackendAttributes::runs_on_cpu` lets search code adapt (fewer task workers on CPU, more on GPU).

```cpp
// src/neural/backend.h:42
struct BackendAttributes {
  bool has_mlh;
  bool has_wdl;
  bool runs_on_cpu;
  int suggested_num_search_threads;
  int recommended_batch_size;
  int maximum_batch_size;
};
```

## 3. Batching Model

**Per-thread gather-and-compute. No central queue, no dedicated NN-worker thread.**

Each `SearchWorker` owns its own `BackendComputation`, fills its own batch, fires its own GPU call, blocks until done, then backs up results. With N search threads you get N concurrent batches in flight to the GPU.

### 3.1 The `BackendComputation` interface

```cpp
// src/neural/backend.h:75
class BackendComputation {
 public:
  virtual size_t UsedBatchSize() const = 0;
  enum AddInputResult {
    ENQUEUED_FOR_EVAL = 0,    // Will be computed during ComputeBlocking();
    FETCHED_IMMEDIATELY = 1,  // Was in cache, the result is already populated.
  };
  virtual AddInputResult AddInput(
      const EvalPosition& pos,    // Input position.
      EvalResultPtr result) = 0;  // Where to fetch data into.
  virtual void ComputeBlocking() = 0;
};
```

Caller passes a pointer (`EvalResultPtr`) at `AddInput` time; backend writes straight into it during `ComputeBlocking`. No post-hoc result lookup.

### 3.2 SearchWorker construction — task workers scale with GPU

```cpp
// src/search/classic/search.h:210
SearchWorker(Search* search, const SearchParams& params)
    : search_(search), ... {
  task_workers_ = params.GetTaskWorkersPerSearchWorker();
  if (task_workers_ < 0) {
    if (search_->backend_attributes_.runs_on_cpu) {
      task_workers_ = 0;                         // CPU: no helpers
    } else {
      int working_threads = std::max(
          search_->thread_count_.load(...) - 1, 1);
      task_workers_ = std::min(
          std::thread::hardware_concurrency() / working_threads - 1, 4U);
    }
  }
  for (int i = 0; i < task_workers_; i++) {
    task_threads_.emplace_back([this, i]() { this->RunTasks(i); });
  }
  target_minibatch_size_ = params_.GetMiniBatchSize();
  if (target_minibatch_size_ == 0) {
    target_minibatch_size_ =
        search_->backend_attributes_.recommended_batch_size;
  }
}
```

GPU backend gets helper threads for parallel tree descent; CPU backend skips them (no point paying sync cost).

### 3.3 The seven-step iteration

```cpp
// src/search/classic/search.cc:1162
void SearchWorker::ExecuteOneIteration() {
  // 1. Initialize internal structures.
  InitializeIteration(search_->backend_->CreateComputation());

  if (params_.GetMaxConcurrentSearchers() != 0) {
    // ... spin-wait on pending_searchers_ atomic (throttle) ...
  }

  // 2. Gather minibatch.
  GatherMinibatch();
  task_count_.store(-1, std::memory_order_release);
  search_->backend_waiting_counter_.fetch_add(1, std::memory_order_relaxed);

  // 2b. Collect collisions.
  CollectCollisions();

  // 3. Prefetch into cache.
  MaybePrefetchIntoCache();

  // 4. Run NN computation.
  RunNNComputation();
  search_->backend_waiting_counter_.fetch_add(-1, std::memory_order_relaxed);

  // 5. Retrieve NN computations (and terminal values) into nodes.
  FetchMinibatchResults();

  // 6. Propagate the new nodes' information to all their parents in the tree.
  DoBackupUpdate();

  // 7. Update the Search's status and progress information.
  UpdateCounters();
}
```

### 3.4 Gather loop — fills the per-thread batch

```cpp
// src/search/classic/search.cc:1285
void SearchWorker::GatherMinibatch() {
  int minibatch_size = 0;
  // ...

  while (minibatch_size < target_minibatch_size_ &&
         number_out_of_order_ < max_out_of_order_) {
    // If there's something to process without touching slow neural net, do it.
    if (minibatch_size > 0 && computation_->UsedBatchSize() == 0) return;

    // If backend work queued and backend is idle - exit immediately so
    // we fire a small batch now rather than block other threads.
    if (thread_count > 1 && minibatch_size > 0 &&
        static_cast<int>(computation_->UsedBatchSize()) >
            params_.GetIdlingMinimumWork() &&
        thread_count - search_->backend_waiting_counter_.load(
                           std::memory_order_relaxed) >
            params_.GetThreadIdlingThreshold()) {
      return;
    }

    PickNodesToExtend(
        std::min({collisions_left, target_minibatch_size_ - minibatch_size,
                  max_out_of_order_ - number_out_of_order_}));
    // ...
  }
}
```

Key sync signal: `backend_waiting_counter_` (atomic). If many peers are already blocked in `ComputeBlocking`, this thread keeps gathering to make the GPU batch larger. If peers are idle / GPU is free, this thread cuts gathering short and fires its batch now. That's the only coordination between search threads during batch-building.

### 3.5 Adding a leaf to the batch

```cpp
// src/search/classic/search.cc:1460
picked_node.is_cache_hit = computation_->AddInput(
                               EvalPosition{
                                   .pos = history.GetPositions(),
                                   .legal_moves = legal_moves,
                               },
                               picked_node.eval->AsPtr()) ==
                           BackendComputation::FETCHED_IMMEDIATELY;
```

`memcache.h` wrapper short-circuits cache hits (`FETCHED_IMMEDIATELY`) — they never touch the GPU.

### 3.6 Firing the GPU call

```cpp
// src/search/classic/search.cc:2133
void SearchWorker::RunNNComputation() {
  if (computation_->UsedBatchSize() > 0) computation_->ComputeBlocking();
}
```

Single blocking call. Same thread that gathered. No handoff, no queue, no callback.

### 3.7 Pulling results

```cpp
// src/search/classic/search.cc:2139
void SearchWorker::FetchMinibatchResults() {
  for (auto& node_to_process : minibatch_) {
    FetchSingleNodeResult(&node_to_process);
  }
}
```

Results already written to each node's `eval` struct via the pointer passed at `AddInput` time. Backup (`DoBackupUpdate`) then propagates Q/D/M up the tree under `nodes_mutex_`.

## 4. How threads stay busy while GPU runs

- Virtual loss on in-flight leaves → other threads pick different paths, avoid collision.
- Per-thread batches → thread B can be gathering while thread A blocks on GPU.
- `backend_waiting_counter_` → gatherers opportunistically yield early when GPU free.
- `task_workers_` inside one SearchWorker → parallel tree descent feeding one batch (GPU only).
- `batchsplit.cc` wrapper → auto-splits batches exceeding `maximum_batch_size`.

## 5. Key files

| File | Role |
|---|---|
| `src/neural/backend.h` | `Backend` / `BackendComputation` interfaces |
| `src/neural/backends/cuda/` | CUDA + cuDNN impls |
| `src/neural/batchsplit.cc` | Max-batch-size splitter wrapper |
| `src/neural/memcache.h` | Cache wrapper (FETCHED_IMMEDIATELY path) |
| `src/search/classic/search.h:208` | `SearchWorker` class |
| `src/search/classic/search.cc:1162` | `ExecuteOneIteration` — the 7-step pipeline |
| `src/search/classic/search.cc:1285` | `GatherMinibatch` |
| `src/search/classic/search.cc:2133` | `RunNNComputation` |
