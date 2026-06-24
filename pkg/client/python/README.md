# Timeslice Python SDK

This is the Python library for Timeslice.

## Installation

```bash
pip install .
```

For development (including gRPC tools):
```bash
pip install .[dev]
```

## Clients

This SDK provides two clients:

1.  **Snapshot Agent Client**: Used to trigger manual snapshots and restores on local nodes. See usage below.
2.  **Accelerator Orchestrator Client**: Used to coordinate shared GPU access between jobs in a time-slice group. Detailed documentation and examples can be found in the [Orchestrator README](timeslice/orchestrator/README.md).

---

## Snapshot Agent Client Usage

```python
from timeslice.snapshot_agent import SnapshotAgentClient

with SnapshotAgentClient(endpoint="localhost:9001") as client:
    # Trigger a snapshot and wait for it to complete
    result = client.snapshot_and_wait(
        job_id="my-job", 
        group="default", 
        backend="CUDA"
    )
    print(f"Snapshot finished with status: {result.status}")
```

---

## Development

To generate gRPC stubs for the Snapshot Agent:

```bash
python3 -m grpc_tools.protoc \
    -I../../snapshot-agent/api/v1alpha1 \
    --python_out=timeslice/snapshot_agent \
    --grpc_python_out=timeslice/snapshot_agent \
    ../../snapshot-agent/api/v1alpha1/snapshot_agent.proto
```

*Note: You may need to fix the imports in the generated files (e.g., `import snapshot_agent_pb2 as snapshot__agent__pb2` -> `from . import snapshot_agent_pb2 as snapshot__agent__pb2`).*

To generate stubs for the Accelerator Orchestrator, see the [Orchestrator README](timeslice/orchestrator/README.md#development).