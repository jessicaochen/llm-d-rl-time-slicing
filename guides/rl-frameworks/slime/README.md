# Time-Slicing Integration Guide for Slime Workloads

This guide provides step-by-step instructions on how to integrate and deploy **Slime** (high-performance RL framework for LLMs) with the **llm-d-rl-time-slicing** platform.

### Motivation: Maximizing GPU Utilization
In traditional disaggregated RL setups, GPUs sit idle whenever worker groups wait for another phase to complete (e.g., trainer GPUs idling during rollout generation, or rollout GPUs idling during policy updates). Cooperative time-slicing enables multiple independent Slime jobs to multiplex physical GPU resource pools concurrently. When one job finishes a phase, its GPU context is checkpointed and evicted, allowing another job to immediately utilize the hardware—significantly driving up GPU duty cycle and overall cluster throughput.

For detailed throughput comparisons, execution timings, and GPU duty cycle metrics, see:
* **[Slime RL Job Benchmark Results & Performance Data](BENCHMARKS.md)**

---

## Table of Contents
1. [Cluster Prerequisites](#1-cluster-prerequisites)
2. [Deploying the Time-Slicing Platform](#2-deploying-the-time-slicing-platform)
3. [Code Integration with Slime](#3-code-integration-with-slime)
4. [Deploying the Modified Slime Variant](#4-deploying-the-modified-slime-variant)
5. [Submitting and Observing Time-Sliced Jobs](#5-submitting-and-observing-time-sliced-jobs)
6. [Observing Convergence and Job Completion](#6-observing-convergence-and-job-completion)

---

## 1. Cluster Prerequisites

Before deploying cooperative time-slicing for Slime, ensure your Kubernetes cluster meets the following requirements:

### Kubernetes Version
* Kubernetes **v1.26** or later.

### GPU Node Configuration
* GPU nodes must run **NVIDIA GPU Driver 565 or later**. This is a strict requirement to support **NVIDIA Dynamic Resource Allocation (DRA)**, which enables transparent context switching and snapshot/restore of GPU state.
* GPU memory capacity must be sufficient to hold the active working set of a single Slime job's trainer or sampler at any one time (since inactive jobs will have their GPU states checkpointed and evicted).

### Node Labeling for Time-Slice Pools
The `timeslice` platform relies on node labels to identify resource pools (groups). For disaggregated Slime executions, label your GPU nodes accordingly:
* **Sampler Nodes**:
  ```bash
  kubectl label nodes <node-name> group.timeslice.io/samplers=true
  ```
* **Trainer Nodes**:
  ```bash
  kubectl label nodes <node-name> group.timeslice.io/trainers=true
  ```

---

## 2. Deploying the Time-Slicing Platform

We deploy the core platform components—**Accelerator Orchestrator**, **Snapshot Agent** (DaemonSet), and the **NVIDIA DRA Driver**—using the parent Helm chart.

### Step 1: Update Helm Chart Dependencies
From the root of your `llm-d-rl-time-slicing` workspace, navigate to the `deploy` directory and fetch the required subcharts:
```bash
cd deploy/
helm dependency update .
```

### Step 2: Configure `values.yaml`
Review or modify the parent `values.yaml` file to match your cluster environment:
```yaml
acceleratororchestrator:
  replicaCount: 2
  image:
    tag: latest

snapshot-agent:
  image:
    tag: latest

nvidia-dra-driver-gpu:
  enabled: true
  # Use "/home/kubernetes/bin/nvidia/" for GKE Container-Optimized OS (COS) nodes.
  # Use "/opt/nvidia" for standard Ubuntu/Debian nodes.
  nvidiaDriverRoot: "/home/kubernetes/bin/nvidia/"
```

### Step 3: Install the Helm Chart
Install the chart into a dedicated namespace (`timeslice-system`). This ensures all service accounts, RBAC policies, and daemons are securely isolated:
```bash
helm install timeslice . -n timeslice-system --create-namespace
```

### Step 4: Verify Platform Health
Verify that the orchestrator and agents are running and healthy:
1. **Using the `rlts` CLI**:
   Build the CLI tool and run the verify command:
   ```bash
   go build -o bin/rlts ./cmd/rlts
   ./bin/rlts orchestrator verify
   ```
2. **Using kubectl**:
   Ensure all pods in the `timeslice-system` namespace are `Running`:
   ```bash
   kubectl get pods -n timeslice-system
   ```

---

## 3. Code Integration with Slime

To participate in cooperative time-slicing, the Slime training loop driver requests and yields access to the GPU resource pools at its natural phase boundaries.

Because worker processes (SGLang engines and Megatron-LM trainer actors) run as background servers, **only the main RL loop driver script (`train.py`)** needs to communicate with the Accelerator Orchestrator via `OrchestratorClient`.

### Step 1: Add Time-Slicing Command-Line Arguments
Add time-slicing configuration options to `slime/utils/arguments.py`:

```python
parser.add_argument(
    "--enable-timeslice",
    action="store_true",
    default=False,
    help="Enable llm-d-rl-time-slicing cooperative GPU grant acquisition.",
)
parser.add_argument(
    "--timeslice-orchestrator-addr",
    type=str,
    default="timeslice-acceleratororchestrator.timeslice-system.svc.cluster.local:50051",
    help="Address of the Accelerator Orchestrator gRPC service.",
)
parser.add_argument(
    "--timeslice-job-id",
    type=str,
    default=None,
    help="Unique job identifier for the Accelerator Orchestrator.",
)
parser.add_argument(
    "--timeslice-sampler-group",
    type=str,
    default="group-slime-sampler",
    help="Accelerator Orchestrator time-slice group for rollout samplers.",
)
parser.add_argument(
    "--timeslice-trainer-group",
    type=str,
    default="group-slime-trainer",
    help="Accelerator Orchestrator time-slice group for trainer actors.",
)
```

### Step 2: Initialize `OrchestratorClient` in `train.py`
In `train.py`, instantiate clients for both the sampler and trainer GPU groups when `--enable-timeslice` is passed:

```python
import os
from timeslice import OrchestratorClient

def train(args):
    sampler_client = None
    trainer_client = None
    job_id = getattr(args, "timeslice_job_id", None) or os.getenv("TIMESLICE_JOB_ID", "slime-job-default")

    if getattr(args, "enable_timeslice", False):
        addr = getattr(args, "timeslice_orchestrator_addr", "timeslice-acceleratororchestrator.timeslice-system.svc.cluster.local:50051")
        sampler_group = getattr(args, "timeslice_sampler_group", "group-slime-sampler")
        trainer_group = getattr(args, "timeslice_trainer_group", "group-slime-trainer")

        sampler_client = OrchestratorClient(target=addr, job_id=job_id, group_id=sampler_group)
        trainer_client = OrchestratorClient(target=addr, job_id=job_id, group_id=trainer_group)
```

### Step 3: Wrap the Training and Rollout Phases
Acquire and release GPU grants around the rollout collection and policy training loops in `train.py`:

```python
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # ---------------------------------------------------------
        # Phase 1: Rollout Generation (Rollout GPU Group)
        # ---------------------------------------------------------
        if sampler_client:
            sampler_client.acquire()

        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        if sampler_client:
            sampler_client.release()

        # ---------------------------------------------------------
        # Phase 2: Megatron-LM Policy Training (Trainer GPU Group)
        # ---------------------------------------------------------
        if trainer_client:
            trainer_client.acquire()

        # Run model training and weight updates
        actor_model.async_train(rollout_id, rollout_data_ref)
        actor_model.update_weights()

        if trainer_client:
            trainer_client.release()

    if sampler_client:
        sampler_client.close()
    if trainer_client:
        trainer_client.close()
```

> [!TIP]
> For a tested codebase reference branch containing these exact changes, see [jessicaochen/slime (timeslice branch)](https://github.com/jessicaochen/slime/tree/timeslice).

---

## Concrete Deployment & Benchmark Reference

For a step-by-step reproduction guide using RayCluster Kubernetes manifests, disaggregated GRPO launch scripts, and verified performance benchmark numbers, see:

* **[Disaggregated Time-Sliced Slime Deployment Guide](sync-disaggregated-timesliced/README.md)**

---

## 4. Deploying the Modified Slime Variant

To run your modified Slime workload on the cluster, you must package the `timeslice` client library and configure the Kubernetes deployments.

### Step 1: Package and Containerize
Ensure the `timeslice` Python client is installed in your Slime container image. Add the following to your Slime `Dockerfile`:

<!-- TDB: Less than 98% confident in the exact base image or Dockerfile structure of the Slime workload. Customize this step to fit your existing Docker build process. -->
```dockerfile
# Copy the local timeslice Python client library into the image
COPY pkg/client/python /opt/timeslice-client

# Install the client library and its dependencies (grpcio, protobuf, etc.)
RUN pip install /opt/timeslice-client
```

### Step 2: Configure Kubernetes Deployments
When deploying Slime trainers or samplers as Kubernetes Pods (or via PyTorchJob / KubeRay), you must:
1. Provide the orchestrator address.
2. Inject a unique `TIMESLICE_JOB_ID` for each job.
3. Configure the correct node selectors or tolerations so pods land on the labeled GPU nodes.

Example Pod template snippet:
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: slime-trainer-job-a
  labels:
    # Match the timeslice group
    timeslice.io/group: trainers
spec:
  # Force placement on the trainers GPU pool
  nodeSelector:
    group.timeslice.io/trainers: "true"
  containers:
  - name: slime-container
    image: my-registry/slime-modified:latest
    env:
    - name: TIMESLICE_JOB_ID
      value: "slime-job-a"
    resources:
      limits:
        # Request standard GPU resources. 
        # The DRA driver and Snapshot Agent will handle the physical sharing.
        nvidia.com/gpu: "8" 
```

---

## 5. Submitting and Observing Time-Sliced Jobs

Once the platform is deployed and the Slime code is integrated, you can submit multiple jobs and observe them sharing the GPUs.

### Step 1: Submit Multiple Jobs
Deploy two independent Slime jobs to the cluster (e.g., `slime-job-a` and `slime-job-b`).
Ensure they have unique `TIMESLICE_JOB_ID` environment variables.

### Step 2: Port-Forward the Orchestrator
To monitor the orchestrator state from your local machine, port-forward the gRPC service:
```bash
kubectl port-forward svc/timeslice-acceleratororchestrator 50051:50051 -n timeslice-system
```

### Step 3: Observe Time-Slicing via the CLI
Use the `rlts` CLI tool to watch the active resource allocations in real-time.

1. **Watch the Samplers Pool**:
   ```bash
   watch -n 0.5 ./bin/rlts orchestrator status samplers
   ```
   **Expected Output:**
   You should see the `Active Job` and `Locking Job` alternate between `slime-job-a` and `slime-job-b`. When one job is sampling, the other job's status will show in the `Waiter Queue Depth` (depth = 1).

2. **Watch the Trainers Pool**:
   ```bash
   watch -n 0.5 ./bin/rlts orchestrator status trainers
   ```
   In a pipelined setup, you will observe the jobs interleaving: while `slime-job-a` is using the `trainers` pool, `slime-job-b` is using the `samplers` pool, and vice-versa.

### Step 4: Observe Context Switches in the Logs
You can inspect the platform logs to verify that the Snapshot Agent is actively saving and restoring GPU states during swaps.

1. **Orchestrator Logs (Scheduling Decisions)**:
   ```bash
   kubectl logs -n timeslice-system -l app.kubernetes.io/name=acceleratororchestrator --tail=100 -f
   ```
   Look for lines indicating lock transfers:
   ```text
   [INFO] Acquire request from job "slime-job-b" for group "samplers" - Queued (Lock held by "slime-job-a")
   [INFO] Yield received from job "slime-job-a" for group "samplers"
   [INFO] Granting lock to next waiter "slime-job-b" for group "samplers"
   ```

2. **Snapshot Agent Logs (State Checkpoint & Restore)**:
   ```bash
   kubectl logs -n timeslice-system -l app.kubernetes.io/name=snapshot-agent --tail=100 -f
   ```
   Look for lines showing the actual GPU context switching:
   ```text
   [INFO] Evicting/Snapshotting GPU state for job "slime-job-a" on node "gpu-node-1"
   [INFO] Snapshot completed in 142ms.
   [INFO] Restoring GPU state for job "slime-job-b" on node "gpu-node-1"
   [INFO] Restore completed in 158ms.
   ```

---

## 6. Observing Convergence and Job Completion

Cooperative time-slicing shares the accelerator hardware transparently at the system level. While the wall-clock time per iteration will reflect the shared resource environment, the **algorithmic convergence** (how the model learns over training steps) remains completely unaffected.

### A. Monitoring Training Metrics & Convergence
Slime workloads typically log training metrics to **TensorBoard**, **Weights & Biases (W&B)**, or local stdout logs. You can observe convergence by monitoring standard RL metrics:
1. **Reward/Score Curves**: The mean reward should steadily increase over iterations, indicating the policy is successfully learning.
2. **Policy & Value Loss**: Megatron-LM's training loss curves (actor loss, critic/value loss) should stabilize or decrease as training progresses.
3. **KL Divergence**: Monitor the KL divergence between the active policy and the reference model to ensure it stays within target bounds (e.g., to prevent policy collapse).
4. **Step vs. Wall-Clock Time**:
   * **Step-wise Convergence**: The step-wise convergence graph (e.g., Reward vs. Training Steps) will align perfectly with a standalone (non-timesliced) run. The time-slicing process does not alter the mathematical state transitions.
   * **Wall-Clock Progress**: Because the GPUs are shared, the wall-clock time per step will increase by a factor of $N$ (where $N$ is the number of co-located jobs), minus any gains from overlapping CPU-heavy phases (like reward processing or data loading) of one job with the other job's GPU phases.

### B. Observing Job Completion
When a Slime job completes its designated number of iterations:
1. **Graceful Exit**: The `OrchestratorClient` context manager or the `.close()` method will clean up the gRPC channels and permanently release any remaining locks.
2. **Kubernetes Job Status**:
   If deployed as a Kubernetes `Job` or `PyTorchJob` (via the Kubeflow Training Operator), you can observe the status transition to `Completed` (or `Succeeded`):
   ```bash
   kubectl get jobs -w
   # or for Kubeflow Training Operator:
   kubectl get pytorchjobs -w
   ```
   **Expected Output:**
   ```text
   NAME             COMPLETION   STATUS      AGE
   slime-job-a      1/1          Succeeded   45m
   slime-job-b      0/1          Running     46m
   ```
3. **Release of Lock Pools**:
   Once `slime-job-a` completes and terminates, the orchestrator will notice the channel closure, and `slime-job-b` will get **exclusive, continuous access** to the GPU pools without any further time-slicing delays. You can verify this via:
   ```bash
   ./bin/rlts orchestrator status samplers
   ```
   The `Waiter Queue Depth` will drop to `0` and stay there, and the `Active Job` will remain permanently assigned to `slime-job-b` until it also completes.

