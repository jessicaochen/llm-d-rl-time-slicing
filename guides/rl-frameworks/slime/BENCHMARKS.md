# Slime RL Job Benchmark Results & Execution Guide

This document records benchmark results and metrics (job execution durations, GPU duty cycles, and setup comparisons) for running parallel **THUDM Slime** RL training jobs on Kubernetes, along with instructions to reproduce the benchmarks.

---

## 1. Benchmark Results Table

Below is the summary matrix comparing different Slime RL cluster setups:

| Setup & Configuration | Parallel RL Jobs | RL Loops / Job | GPU Machines Involved | Total Time (Full Set) | Avg Job Duration | Max Job Duration | GPU Duty Cycle (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| [Vanilla Disaggregated Sync (Sequential)](file:///usr/local/google/home/jesschen/git/llm-d-rl-time-slicing/guides/rl-frameworks/slime/sync-disaggregated-sequential/README.md) | 2 | 10 | 2x g2-standard-32 (1x Trainer L4, 1x Sampler L4) | 534.84s (8.91 min) | 417.48s (6.96 min) | 534.84s (8.91 min) | Rollout: 31.78% (29.05% avg util)<br>Trainer: 15.89% (13.12% avg util) |
| [Cooperative Time-Slicing Platform](file:///usr/local/google/home/jesschen/git/llm-d-rl-time-slicing/guides/rl-frameworks/slime/sync-disaggregated-timesliced/README.md) | 2 | 10 | 2x g2-standard-32 (1x Trainer L4, 1x Sampler L4) | 565.10s (9.42 min) | 444.72s (7.41 min) | 560.06s (9.33 min) | Rollout: 35.40% (31.16% avg util)<br>Trainer: 10.62% (7.56% avg util) |

---

## 2. Benchmark Execution Guide (`run_parallel_jobs_benchmark.py`)

The automated benchmark script [`run_parallel_jobs_benchmark.py`](file:///usr/local/google/home/jesschen/git/llm-d-rl-time-slicing/guides/rl-frameworks/slime/run_parallel_jobs_benchmark.py) submits a configurable number of parallel Slime RL jobs, measures job completion timing metrics, and collects real-time GPU duty cycle samples across all worker GPU nodes.

### Prerequisites & Cluster Requirements
* **Kubernetes Version**: `v1.26+` (e.g., `v1.30.5-gke.1443001`).
* **GPU Node Pools**:
  * `trainer-gpu-pool`: 1 machine (e.g. `g2-standard-32` with 1x NVIDIA L4 24GB GPU, 200GB+ disk).
  * `sampler-gpu-pool`: 1 machine (e.g. `g2-standard-32` with 1x NVIDIA L4 24GB GPU, 200GB+ disk).
* **KubeRay Operator**: Installed in the `kuberay-system` namespace.

### Step 1: Deploy the Ray Cluster

Deploy the target Ray cluster manifest:

```bash
kubectl apply -f guides/rl-frameworks/slime/sync-disaggregated-sequential/ray-cluster-disaggregated.yaml
```

Verify that all Ray head and worker pods reach the `1/1 Running` state:
```bash
kubectl get pods -l ray.io/cluster=slime-disaggregated-cluster
```

### Step 2: Run the Benchmark Script

Execute `run_parallel_jobs_benchmark.py` directly from your host environment:

```bash
python3 guides/rl-frameworks/slime/run_parallel_jobs_benchmark.py
```

#### What the script automatically performs:
1. Identifies the Ray Head pod and active GPU worker pods (`trainer` and `rollout` roles).
2. Spawns a background sampling thread that polls `nvidia-smi` GPU utilization every second.
3. Submits `NUM_JOBS` parallel Slime RL jobs to the Ray Cluster API via `ray job submit`.
4. Polls job execution statuses until all jobs complete (`SUCCEEDED` or `FAILED`).
5. Generates a report containing:
   * **Individual Job Durations** ($T_i$)
   * **Average Job Duration** ($\bar{T}$)
   * **Total Set Time** ($T_{\text{total}}$)
   * **GPU Duty Cycle %** ($\frac{\text{Active Samples}}{\text{Total Samples}} \times 100$) per GPU node pool.

### Step 3: Clean Up Benchmark Resources

Delete the deployed RayCluster and clean up temporary storage:

```bash
# Delete Ray cluster custom resources
kubectl delete -f guides/rl-frameworks/slime/sync-disaggregated-sequential/ray-cluster-disaggregated.yaml

# (Optional) Uninstall KubeRay Operator
helm uninstall kuberay-operator --namespace kuberay-system
kubectl delete namespace kuberay-system
```
