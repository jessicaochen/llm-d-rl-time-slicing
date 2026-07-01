# Disaggregated Sync Slime RL Job Deployment Setup

This directory contains the manifest files, scripts, and step-by-step instructions to reproduce the vanilla **disaggregated synchronous Reinforcement Learning (RL)** training job using [THUDM Slime](https://github.com/THUDM/slime) on Kubernetes.

---

## Job Interleave Strategy: Sequential Execution (`RAY_MAX_CONCURRENT_JOBS=1`)

* **Ray Job Queueing**: `RAY_MAX_CONCURRENT_JOBS: "1"` limits Ray to 1 active job at a time.
* **Sequential Behavior**: Job 1 runs to completion (`SUCCEEDED`). Job 2 waits in `PENDING` status consuming 0 GPU memory, then automatically starts when Job 1 finishes.

---

## 1. Verified Cluster Setup

The disaggregated GRPO job was tested and verified on the following cluster configuration:

* **Kubernetes Version**: `v1.30.5-gke.1443001`
* **Node Pools**:
  * **`trainer-gpu-pool`**:
    * **Machine Count**: 1
    * **Machine Type**: `g2-standard-32` (1x NVIDIA L4 GPU / 24GB VRAM, 32 vCPUs, 128 GB RAM)
    * **Disk Size**: `200 GB` pd-balanced
  * **`sampler-gpu-pool`**:
    * **Machine Count**: 1
    * **Machine Type**: `g2-standard-32` (1x NVIDIA L4 GPU / 24GB VRAM, 32 vCPUs, 128 GB RAM)
    * **Disk Size**: `200 GB` pd-balanced

---

## 2. Deploying KubeRay Operator

Slime relies on Ray for cluster lifecycle management and worker coordination.

Install the KubeRay Operator (version `v1.6.2`) via Helm into the `kuberay-system` namespace:

```bash
# Add KubeRay Helm repo
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update

# Install KubeRay Operator
helm install kuberay-operator kuberay/kuberay-operator --namespace kuberay-system --create-namespace
```

Verify operator deployment:
```bash
kubectl get pods -n kuberay-system
```

---

## 3. Container Image Selection

Slime relies on complex dependencies (Megatron-LM, SGLang, Apex, TransformerEngine, `sgl-router`).

Official THUDM image used in testing:
* **Image**: `slimerl/slime:latest`

---

## 4. How to Deploy the Ray Cluster

The `ray-cluster-disaggregated.yaml` manifest deploys:
* 1 Ray Head Node on standard CPU nodes configured with `RAY_MAX_CONCURRENT_JOBS: "1"`.
* 1 Ray Worker Group (`trainer-group`) targeting `cloud.google.com/gke-nodepool: trainer-gpu-pool`.
* 1 Ray Worker Group (`rollout-group`) targeting `cloud.google.com/gke-nodepool: sampler-gpu-pool`.

Apply the cluster manifest:

```bash
kubectl apply -f guides/rl-frameworks/slime/sync-disaggregated-sequential/ray-cluster-disaggregated.yaml
```

Verify that all pods reach the **`1/1 Running`** state:

```bash
kubectl get pods -l ray.io/cluster=slime-disaggregated-cluster
```

---

## 5. How to Run the RL Training Job

We use **Strategy 2 (AutoBridge)** for zero-preconversion runtime weight loading into Megatron-LM.

### Step 1: Copy the Launch Script to the Head Pod

```bash
# Get head pod name
HEAD_POD=$(kubectl get pods -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')

# Copy run script to head pod
kubectl cp guides/rl-frameworks/slime/sync-disaggregated-sequential/run_disaggregated_grpo.sh ${HEAD_POD}:/tmp/run_disaggregated_grpo.sh
```

### Step 2: Submit the Ray Job

Execute `ray job submit` from the head pod:

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

### Step 3: Monitor Job Execution

You can check job logs or status at any time:

```bash
# Check status
kubectl exec -it ${HEAD_POD} -- ray job status <JOB_ID>

# Stream logs
kubectl exec -it ${HEAD_POD} -- ray job logs <JOB_ID>
```

---

## 6. How to Clean Up

To tear down the RL training job, Ray Cluster, and KubeRay operator:

### Step 1: Delete the RayCluster Custom Resource
```bash
kubectl delete -f guides/rl-frameworks/slime/sync-disaggregated-sequential/ray-cluster-disaggregated.yaml
```

### Step 2: Uninstall KubeRay Operator (Optional)
```bash
helm uninstall kuberay-operator --namespace kuberay-system
kubectl delete namespace kuberay-system
```
