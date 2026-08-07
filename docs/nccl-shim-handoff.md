# NCCL C/R Shim Integration — Handoff

Status as of 2026-08-07. Branch: `nccl-shim-cr` (off `main`), first commit `77c129c`.
This document gives full context for continuing the work; it assumes no prior
knowledge of the investigation that led here.

## Problem

Multi-GPU workloads (Megatron trainers, vLLM/SGLang TP>1) cannot be
checkpointed by the `cuda` backend today: NCCL's P2P/SHM transports create
cross-process CUDA IPC state (legacy `cuIpcGetMemHandle` mappings and/or
cuMem shareable handles) that cuda-checkpoint cannot freeze. The current
Slime integration works around this by holding both Trainer and Sampler
locks during NCCL weight sync (`guides/rl-frameworks/slime/sync/SLIME_CHANGES.md`)
so a checkpoint never lands while IPC channels are live — meaning no VRAM is
freed in those windows and multi-GPU trainers are effectively
un-checkpointable.

## Routes evaluated (research summary)

1. **Driver r610 "jobs"** (cuda-checkpoint 6.10 features,
   https://github.com/NVIDIA/cuda-checkpoint): processes launched under
   `cuda-checkpoint --launch-job` (or with `CUDA_CHECKPOINT_JOB_FILE`) may
   share legacy CUDA IPC and remain checkpointable. Protocol: lock all
   processes sequentially, checkpoint each, restore+unlock in the same
   order. Requires driver ≥ 610 (new-feature branch since May 2026, thin
   managed-platform availability), `NCCL_CUMEM_ENABLE=0` (cuMem-exported
   memory still unsupported), one-invocation-per-PID sequencing changes in
   the agent. Deferred: future transparent path, not implemented.
2. **Framework-level teardown/reinit** (NVRx in-process-restart style,
   https://nvidia.github.io/nvidia-resiliency-ext/inprocess/usage_guide.html):
   destroy torch.distributed groups before freeze, re-init after restore.
   Hard part is re-aligning stale ProcessGroup references cached across
   Megatron (DDP buffers, DistributedOptimizer, TE modules, pipeline
   schedules). Superseded by route 3, which solves this at a lower layer.
3. **v2 LD_PRELOAD shim** (CHOSEN):
   https://github.com/aishukamal/rl-time-slicing/blob/main/multi-gpu-cr-poc/REPORT.md
   — `universal_cr_shim_v2.c`, an LD_PRELOAD library that intercepts NCCL
   calls and keeps an app-handle → live-handle translation table, so the
   framework's cached `ncclComm_t` handles never go stale. On SIGRTMIN+1 it
   destroys all tracked communicators (~350–450 ms, removes all
   cross-process GPU state); cuda-checkpoint then freezes only
   process-private state. On SIGRTMIN+2 (post-restore) it arms lazy
   recreation: the next collective triggers a fresh ncclUniqueId rendezvous
   (rank 0 publishes via `/dev/shm`) and rebuilds comms. Validated in the
   PoC on 2×H100, driver 580.126.20, NCCL 2.30.7: FSDP DP=2 ×3 cycles and
   vLLM TP=2, 100% VRAM release, NVLink P2P at steady state with zero
   performance tax. Zero framework code changes required.

### Hard constraints discovered (apply regardless of route)

- `NCCL_NVLS_ENABLE=0` is mandatory: driver multicast binding is broken
  post-restore (`cuMulticastAddDevice` error 101, process-wide). NVLink
  P2P is unaffected; only NVSwitch in-network reductions (NVLS) are lost.
- CUDA graphs cannot be captured/held across a multi-GPU freeze (driver
  limitation) — trainers must run eager (Megatron default).
- TE `tp_comm_overlap` must stay off: its userbuffers use CUDA IPC/multicast
  outside NCCL, invisible to the shim.
- `NCCL_DEBUG=INFO` currently masks a CommCheck race in communicator
  recreation (report: recreate fails with rc=1/3/6 without it). Treat as
  load-bearing until an NVIDIA driver fix; worth filing/tracking upstream.
- Shim rendezvous is same-host (`/dev/shm`) — multi-node TP needs a shim
  extension (TCP or shared-volume bootstrap). Multi-node also needs an
  orchestrator cross-node barrier (lock all ranks on all nodes before
  checkpointing any; today `reconcileNode` in
  `pkg/accelerator-orchestrator/controller/controller.go` is serial per
  node with no barrier). Out of scope for this branch.

## What is implemented on `nccl-shim-cr` (commit 77c129c)

Agent-side signaling, disabled by default, flag/env/Helm controlled:

- `pkg/snapshot-agent/backends/cuda-checkpoint.go`: `NCCLShimConfig`
  (Enabled, DestroySignal=35/SIGRTMIN+1, RecreateSignal=36/SIGRTMIN+2,
  DestroyWait=2s) + `WithNCCLShim(...)` option on `NewCudaCheckpoint`.
  - `Snapshot`: signal destroy to every job PID → wait `DestroyWait`
    (context-aware) → existing `--action lock` / `--action checkpoint`.
    Signal failure aborts before any freeze.
  - `Restore`: existing `--toggle` → signal recreate to every PID. Signal
    failure returns an error (job goes FAULTED rather than resuming with
    dead comms).
- `cmd/snapshot-agent/main.go`: flags `--nccl-shim`,
  `--nccl-shim-destroy-signal`, `--nccl-shim-recreate-signal`,
  `--nccl-shim-destroy-wait`; env overrides `NCCL_SHIM_ENABLED`,
  `NCCL_SHIM_DESTROY_SIGNAL`, `NCCL_SHIM_RECREATE_SIGNAL`,
  `NCCL_SHIM_DESTROY_WAIT` (chart configures via env, matching
  DEPLOYMENT_MODE/AGENT_PORT pattern).
- Helm: `deploy/snapshot-agent/values.yaml` `ncclShim.*` →
  `templates/daemonset.yaml` env block (rendered only when enabled).
  NOTE: `helm template` was NOT run (helm unavailable on the dev machine) —
  verify rendering before deploying.
- Tests (`cuda_checkpoint_test.go`): ordering (destroy → lock → checkpoint;
  toggle → recreate), no signals when disabled, no exec after signal
  failure. `go build ./...`, `go vet`, and the full snapshot-agent test
  suite pass.

Design notes: signals are sent via injectable `signalProcess`
(default `syscall.Kill`; agent has hostPID + privileged, so host PIDs are
reachable). The destroy-completion "ack" is currently a fixed wait
(`DestroyWait`); PoC measured ~400 ms for 2 GPUs, default 2 s. A real ack
(state file from the shim, or app-channel message) is a known improvement —
see TODOs.

## Remaining work

### Repo side
1. Verify Helm rendering (`helm template deploy/snapshot-agent --set
   ncclShim.enabled=true`).
2. Docs: add a section to `guides/snapshot-agent/README.md` describing the
   shim mode, its prerequisites, and the env preset below.
3. Optional hardening: replace fixed `DestroyWait` with an ack mechanism;
   surface shim mode in agent `Status` / `rlts inspect`.
4. Obtain/build the shim artifact: `multi-gpu-cr-poc/universal_cr_shim_v2.c`
   from https://github.com/aishukamal/rl-time-slicing (glibc must match the
   workload image). Decide distribution: bake into trainer images vs
   hostPath from the agent DaemonSet.

### Slime/Megatron side (to use this purely for the Megatron trainer)
Megatron itself needs zero code changes; everything is launch config plus
one Slime patch:
1. Trainer pod: `LD_PRELOAD=<path>/universal_cr_shim_v2.so` set at pod level
   so Ray actor (rank) processes inherit it. Verify PyTorch dynamically
   links `libnccl.so.2` (`ldd libtorch_cuda.so`) — pip wheels do via the
   `nvidia-nccl` package; statically-linked builds make LD_PRELOAD a silent
   no-op.
2. Env preset: `NCCL_NVLS_ENABLE=0`, `NCCL_DEBUG=INFO`,
   `TORCH_NCCL_RETHROW_CUDA_ERRORS=0`, NCCL watchdog/heartbeat timeout
   longer than the longest expected frozen window (trainer is suspended for
   entire sampler slices). NCCL ≥ 2.29.7, driver ≥ 580. CUDA graphs off,
   TE `tp_comm_overlap` off.
3. Slime patch — ephemeral weight-sync group: the trainer↔sampler broadcast
   group used in `update_weights` must be created inside the sync (both
   locks held, as today) and destroyed before locks are released. If it
   stays persistent, the shim destroys the trainer's end at checkpoint and
   lazy recreate hangs on resume (sampler peers are a different job and
   cannot rendezvous mid-slice). The hold-both-locks pattern during
   `update_weights` itself remains (live transfer needs both sides awake);
   the shim removes the need to avoid checkpointing *between* syncs.
4. Ensure the trainer's `Yield` is preceded by a full drain
   (`torch.cuda.synchronize()` after distributed-optimizer async grad ops):
   the destroy signal must land on a quiesced process; the shim's destroy
   is unsafe mid-collective.
5. Deployment scoping: `ncclShim.enabled=true` only on trainer-group nodes
   (or rely on the fact that the flag only affects `cuda`-backend jobs —
   the SGLang sampler uses the `app-endpoint` backend and is untouched).

### Validation plan (first real run)
- During trainer freeze: `nvidia-smi` shows trainer VRAM = 0.
- After resume: loss continuity; NVLink traffic resumes
  (`nvidia-smi nvlink -gt d`).
- ≥ 3 consecutive slice cycles (PoC validated repeated cycles on FSDP;
  Megatron distributed optimizer + Ray actors is the untested combination).
- Watch shim destroy duration in logs; bump `--nccl-shim-destroy-wait` if
  teardown approaches the wait at higher per-node rank counts.

## Key references

- PoC report (shim v2, all measured numbers, failure modes):
  https://github.com/aishukamal/rl-time-slicing/blob/main/multi-gpu-cr-poc/REPORT.md
- cuda-checkpoint README (610 job/IPC features, CLI, version matrix):
  https://github.com/NVIDIA/cuda-checkpoint
- NVRx in-process restart (prior art for teardown/reinit; version caveats,
  pytorch#150690 P2P-group fix):
  https://nvidia.github.io/nvidia-resiliency-ext/inprocess/usage_guide.html
- vLLM checkpoint RFC (NCCL reinit precedent):
  https://github.com/vllm-project/vllm/issues/34303
- Existing repo anchors: backend flow
  `pkg/snapshot-agent/backends/cuda-checkpoint.go`; PID discovery/caching
  `pkg/snapshot-agent/server/server.go` (`resolvePIDs`, restore uses cached
  PIDs because NVML cannot rediscover after freeze); per-node
  snapshot→restore ordering
  `pkg/accelerator-orchestrator/controller/controller.go`; Slime workaround
  `guides/rl-frameworks/slime/sync/SLIME_CHANGES.md`.
