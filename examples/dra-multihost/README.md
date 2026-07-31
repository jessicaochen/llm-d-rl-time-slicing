# Multi-Node DRA Experiments

Experiments validating multi-node Dynamic Resource Allocation (DRA) behavior —
ResourceClaims spanning nodes, consumable shares, and time-slicing settings —
as groundwork for lifting the platform's current one-node-per-group limitation
(`rlts inspect` reports: "max 1 node per group supported due to DRA").

This README records the environment the experiments run against. Experiment
manifests and results are added as subfolders as they are worked out.

Placeholders used below: `<PROJECT_ID>` is the GCP project, `<REGISTRY>` is a
private image registry (e.g. an Artifact Registry docker repository).

## Test Environment

### Cluster

| | |
|---|---|
| Cluster | GKE zonal cluster with L4 GPU node pools |
| GKE version | `1.36.2-gke.1346000`, REGULAR release channel (control plane + all node pools) |

### GPU node pools

Two pools of 3 nodes each (6 GPU nodes total):

| Pool | Nodes | Machine type | Accelerator | Group label |
|---|---|---|---|---|
| `sampler-gpu-pool` | 3 | `g2-standard-32` | 1× NVIDIA L4 | `group.timeslice.io/samplers=true` |
| `trainer-gpu-pool` | 3 | `g2-standard-32` | 1× NVIDIA L4 | `group.timeslice.io/trainers=true` |

All GPU nodes also carry `timeslice.io/enabled=true` and
`cloud.google.com/gke-gpu=true`. Labels are set at the node-pool level, so
nodes added by resizing inherit them.

### DRA driver (custom build)

The chart-managed `nvidia-dra-driver-gpu` dependency of the `timeslice`
release is **disabled**; a custom driver build is installed standalone
instead.

| | |
|---|---|
| Source | [NVIDIA/k8s-dra-driver-gpu](https://github.com/NVIDIA/k8s-dra-driver-gpu) `main` @ `6515fed7` (includes consumable-shares-unlimited) |
| Image | `<REGISTRY>/dra-driver-nvidia-gpu:dev-6515fed7` |
| Namespace | `dra-driver-nvidia-gpu` |
| Feature gates | `ConsumableShares=true`, `TimeSlicingSettings=true` |
| Other settings | `consumableShares=unlimited`, `gpuResourcesEnabledOverride=true`, `nvidiaDriverRoot=/home/kubernetes/bin/nvidia` |

Install command (from a `k8s-dra-driver-gpu` checkout):

```bash
helm upgrade -i --create-namespace --namespace dra-driver-nvidia-gpu \
  dra-driver-nvidia-gpu ./deployments/helm/dra-driver-nvidia-gpu \
  -f dra-driver-values.yaml \
  --set image.repository="${REGISTRY}/dra-driver-nvidia-gpu" \
  --set image.tag=dev-6515fed7 \
  --set image.pullPolicy=Always \
  --set nvidiaDriverRoot="/home/kubernetes/bin/nvidia" \
  --set gpuResourcesEnabledOverride=true \
  --set consumableShares="unlimited" \
  --set featureGates.ConsumableShares=true \
  --set featureGates.TimeSlicingSettings=true
```

Two GKE-specific overrides are required on top of the chart defaults
(`dra-driver-values.yaml`):

```yaml
# GKE quota-restricts system-* priority classes outside kube-system.
controller:
  priorityClassName: ""
kubeletPlugin:
  priorityClassName: ""
  tolerations:
    - operator: Exists
  # Chart default affinity keys on NFD labels GKE does not set.
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
          - matchExpressions:
              - key: cloud.google.com/gke-gpu
                operator: In
                values:
                  - "true"
```

The image is built with docker buildx (container driver — the daemon-side
BuildKit in Cloud Build's default docker builder mishandles the Dockerfile's
`COPY --parents`), single-arch `linux/amd64`.

Baseline after install: one `gpu.nvidia.com` ResourceSlice per GPU node (6
total) with `allowMultipleAllocations` on the L4 devices, plus
`compute-domain.nvidia.com` slices.

### Time-slicing stack

| | |
|---|---|
| Release | `timeslice` (Helm, release metadata in `default`, resources in `timeslice-system`) |
| Chart source | `timeslice-rename-go` branch (PR #136) |
| Components | `timeslice-timesliceorchestrator` (Deployment), `timeslice-snapshot-agent` (DaemonSet, 6/6 on GPU nodes) |
| DRA subchart | `nvidia-dra-driver-gpu.enabled=false` (replaced by the custom driver above) |

## Experiments

To be filled in as we go. Planned shape: one numbered subfolder per
experiment with its manifests and a README stating hypothesis, steps, and
observed result.

| # | Experiment | Status |
|---|---|---|
| — | — | — |
