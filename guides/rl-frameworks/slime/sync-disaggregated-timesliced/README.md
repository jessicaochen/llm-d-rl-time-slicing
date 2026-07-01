# Time-Sliced Disaggregated Slime RL Job Deployment Guide

This directory contains the manifest files and scripts for deploying multiple **disaggregated Reinforcement Learning (RL)** training jobs using [THUDM Slime](https://github.com/THUDM/slime) integrated with the pre-deployed **`llm-d-rl-time-slicing`** platform.

Cooperative time-slicing enables multiple independent Slime jobs to share GPU resource pools concurrently. When a job enters an idle phase (e.g. trainer waiting for rollout generation, or vice versa), the time-slicing platform safely checkpoints its GPU state and yields the hardware to concurrent jobs.

---

## Job Interleave Strategy: Cooperative Time-Slicing

* **Concurrent Execution**: Ray job concurrency limits are disabled (`RAY_MAX_CONCURRENT_JOBS` default), allowing multiple jobs to run simultaneously.
* **Driver Grant Control**: `slime/train.py` acquires and releases GPU grants (`group-slime-sampler` and `group-slime-trainer`) via the Accelerator Orchestrator before each rollout and training phase.
* **Transparent Eviction**: When a job yields or waits for a grant, the Snapshot Agent context-switches VRAM to host memory, allowing jobs to interleave GPU usage dynamically across RL phases.

---

## 1. Verified Cluster Setup

The disaggregated GRPO job is tested on the following cluster configuration:

* **Kubernetes Version**: `v1.30.5-gke.1443001`
* **Pre-Labeled Node Pools**:
  * **`trainer-gpu-pool`**: 1x `g2-standard-32` (`group.timeslice.io/trainers=true`, 1x NVIDIA L4 GPU, 200 GB disk)
  * **`sampler-gpu-pool`**: 1x `g2-standard-32` (`group.timeslice.io/samplers=true`, 1x NVIDIA L4 GPU, 200 GB disk)

---

## 2. How to Deploy the Ray Cluster

Deploy the Ray cluster manifest:

```bash
kubectl apply -f guides/rl-frameworks/slime/sync-disaggregated-timesliced/ray-cluster-disaggregated.yaml
```

Verify that all pods reach the `1/1 Running` state:

```bash
kubectl get pods -l ray.io/cluster=slime-disaggregated-cluster
```

---

## 3. How to Run the RL Training Job

### Step 1: Update Slime Codebase on Head Pod

Fetch and checkout the time-sliced Slime branch (`timeslice`) from [jessicaochen/slime](https://github.com/jessicaochen/slime/tree/timeslice) onto the Ray Head Pod:

```bash
HEAD_POD=$(kubectl get pods -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')

kubectl exec ${HEAD_POD} -- bash -c "
  cd /root/slime && \
  git remote add timeslice https://github.com/jessicaochen/slime.git 2>/dev/null || true && \
  git fetch timeslice timeslice && \
  git checkout -B timeslice timeslice/timeslice
"
```

### Step 2: Copy the Launch Script to the Head Pod

```bash
kubectl cp guides/rl-frameworks/slime/sync-disaggregated-timesliced/run_disaggregated_grpo.sh ${HEAD_POD}:/tmp/run_disaggregated_grpo.sh
```

### Step 3: Submit the Ray Job

```bash
kubectl exec -it ${HEAD_POD} -- ray job submit \
  --address="http://127.0.0.1:8265" \
  --runtime-env-json='{
    "env_vars": {
      "PYTHONPATH": "/root/Megatron-LM",
      "CUDA_DEVICE_MAX_CONNECTIONS": "1"
    }
  }' \
  -- bash /tmp/run_disaggregated_grpo.sh
```

---

## 4. How to Clean Up

```bash
kubectl delete -f guides/rl-frameworks/slime/sync-disaggregated-timesliced/ray-cluster-disaggregated.yaml
```
