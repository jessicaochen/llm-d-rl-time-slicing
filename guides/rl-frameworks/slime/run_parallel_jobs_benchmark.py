#!/usr/bin/env python3
"""
run_parallel_jobs_benchmark.py

Runs a specified number of parallel Slime RL jobs on Ray, measures job durations
(individual, average, total set), and collects GPU duty cycle metrics across worker GPUs.
"""

import os
import time
import json
import subprocess
import threading
from datetime import datetime

# Hardcoded Job Settings
NUM_JOBS = 2
POLL_INTERVAL_SEC = 1.0
SCRIPT_PATH = "/tmp/run_disaggregated_grpo.sh"
LOCAL_SCRIPT_SOURCE = "/usr/local/google/home/jesschen/git/llm-d-rl-time-slicing/guides/rl-frameworks/slime/sync-disaggregated-timesliced/run_disaggregated_grpo.sh"

# Global GPU sampling storage
gpu_samples = []
stop_sampling_event = threading.Event()

def get_head_pod():
    cmd = ["kubectl", "get", "pods", "-l", "ray.io/node-type=head", "-o", "jsonpath={.items[0].metadata.name}"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return res.stdout.strip()

def get_gpu_worker_pods():
    cmd = ["kubectl", "get", "pods", "-l", "ray.io/cluster=slime-disaggregated-cluster", "-o", "jsonpath={range .items[*]}{.metadata.name}{' '}{.metadata.labels.slime-role}{'\\n'}{end}"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    pods = []
    for line in res.stdout.strip().split('\n'):
        if line.strip():
            parts = line.split()
            if len(parts) >= 2 and parts[1] in ['trainer', 'rollout']:
                pods.append((parts[0], parts[1]))
    return pods

def sample_gpu_utilization(pods):
    """Background loop to collect GPU utilization samples."""
    while not stop_sampling_event.is_set():
        sample_time = time.time()
        for pod_name, role in pods:
            cmd = ["kubectl", "exec", pod_name, "--", "nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory", "--format=csv,noheader,nounits"]
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if res.returncode == 0 and res.stdout.strip():
                    parts = res.stdout.strip().split(',')
                    gpu_util = float(parts[0].strip())
                    mem_util = float(parts[1].strip())
                    gpu_samples.append({
                        'timestamp': sample_time,
                        'pod': pod_name,
                        'role': role,
                        'gpu_util': gpu_util,
                        'mem_util': mem_util,
                        'active': 1 if gpu_util > 0 else 0
                    })
            except Exception:
                pass
        time.sleep(POLL_INTERVAL_SEC)

def ensure_script_on_head(head_pod):
    print(f"Copying launch script to Head Pod ({head_pod})...")
    subprocess.run(["kubectl", "cp", LOCAL_SCRIPT_SOURCE, f"{head_pod}:{SCRIPT_PATH}"], check=True)

def submit_ray_job(head_pod):
    cmd = [
        "kubectl", "exec", head_pod, "--", "ray", "job", "submit",
        "--address=http://127.0.0.1:8265",
        "--runtime-env-json={\"env_vars\": {\"PYTHONPATH\": \"/root/Megatron-LM\", \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"}}",
        "--no-wait",
        "--", "bash", SCRIPT_PATH
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    for line in res.stdout.split('\n'):
        if "submitted successfully" in line:
            parts = line.split("'")
            if len(parts) >= 2:
                return parts[1]
    raise RuntimeError(f"Could not parse job ID from output: {res.stdout}")

def check_job_status(head_pod, job_id):
    cmd = ["kubectl", "exec", head_pod, "--", "ray", "job", "status", job_id]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    out = res.stdout.lower()
    if "succeeded" in out:
        return "SUCCEEDED"
    elif "failed" in out:
        return "FAILED"
    elif "stopped" in out:
        return "STOPPED"
    elif "running" in out or "pending" in out:
        return "RUNNING"
    return "UNKNOWN"

def main():
    print(f"=========================================================")
    print(f"   PARALLEL RL JOBS BENCHMARK & GPU DUTY CYCLE TRACKER   ")
    print(f"=========================================================")
    print(f"Target parallel jobs: {NUM_JOBS}")
    
    head_pod = get_head_pod()
    gpu_pods = get_gpu_worker_pods()
    print(f"Head Pod: {head_pod}")
    print(f"GPU Worker Pods ({len(gpu_pods)}): {[p[0] + ' (' + p[1] + ')' for p in gpu_pods]}")
    
    ensure_script_on_head(head_pod)
    
    # Start GPU Duty Cycle monitoring thread
    monitor_thread = threading.Thread(target=sample_gpu_utilization, args=(gpu_pods,), daemon=True)
    monitor_thread.start()
    print("Started GPU duty cycle monitoring thread (1s interval)...")
    
    start_time_all = time.time()
    job_records = []
    
    print(f"\nSubmitting {NUM_JOBS} parallel Ray jobs...")
    for i in range(NUM_JOBS):
        submit_time = time.time()
        job_id = submit_ray_job(head_pod)
        print(f"  [Job {i+1}/{NUM_JOBS}] Submitted Job ID: {job_id} at {datetime.fromtimestamp(submit_time).strftime('%H:%M:%S')}")
        job_records.append({
            'index': i + 1,
            'job_id': job_id,
            'start_time': submit_time,
            'end_time': None,
            'duration': None,
            'status': 'RUNNING'
        })
    
    # Poll job statuses until all finish
    active_jobs = len(job_records)
    print("\nMonitoring parallel job execution...")
    while active_jobs > 0:
        time.sleep(2.0)
        active_jobs = 0
        for record in job_records:
            if record['status'] == 'RUNNING':
                status = check_job_status(head_pod, record['job_id'])
                if status in ['SUCCEEDED', 'FAILED', 'STOPPED']:
                    record['end_time'] = time.time()
                    record['duration'] = record['end_time'] - record['start_time']
                    record['status'] = status
                    print(f"  -> Job {record['index']} ({record['job_id']}) finished with status {status} in {record['duration']:.2f}s")
                else:
                    active_jobs += 1

    end_time_all = time.time()
    total_elapsed_set = end_time_all - start_time_all
    
    # Stop GPU sampler
    stop_sampling_event.set()
    monitor_thread.join(timeout=3)
    
    # Calculate Timing Metrics
    durations = [r['duration'] for r in job_records if r['duration'] is not None]
    avg_duration = sum(durations) / len(durations) if durations else 0.0
    
    # Calculate GPU Duty Cycle Metrics per pod
    pod_data = {}
    for sample in gpu_samples:
        pod = sample['pod']
        if pod not in pod_data:
            pod_data[pod] = {'role': sample['role'], 'gpu_utils': [], 'actives': []}
        pod_data[pod]['gpu_utils'].append(sample['gpu_util'])
        pod_data[pod]['actives'].append(sample['active'])

    # Display Summary Report
    print("\n" + "="*65)
    print("                    BENCHMARK RESULTS REPORT                     ")
    print("="*65)
    print(f"Total Parallel Jobs Submitted : {NUM_JOBS}")
    print(f"Total Time for Full Job Set   : {total_elapsed_set:.2f} seconds ({total_elapsed_set/60.0:.2f} min)")
    print(f"Average Duration per Job      : {avg_duration:.2f} seconds ({avg_duration/60.0:.2f} min)")
    print("-" * 65)
    print("INDIVIDUAL JOB TIMINGS:")
    for r in job_records:
        print(f"  * Job #{r['index']} [{r['job_id']}]: {r['status']} | Duration: {r['duration']:.2f}s")
    
    print("-" * 65)
    print("GPU DUTY CYCLE & UTILIZATION (PER NODE):")
    for pod, data in pod_data.items():
        total_samples = len(data['actives'])
        active_samples = sum(data['actives'])
        duty_cycle_pct = (active_samples / total_samples * 100.0) if total_samples > 0 else 0.0
        avg_util = (sum(data['gpu_utils']) / total_samples) if total_samples > 0 else 0.0
        print(f"  * Pod: {pod} ({data['role']})")
        print(f"    - Duty Cycle (>0% util) : {duty_cycle_pct:.2f}% ({active_samples}/{total_samples} samples)")
        print(f"    - Average GPU Util      : {avg_util:.2f}%")
    print("="*65)

if __name__ == "__main__":
    main()
