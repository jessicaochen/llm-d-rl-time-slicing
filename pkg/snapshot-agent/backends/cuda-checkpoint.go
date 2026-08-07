package backends

import (
	"context"
	"fmt"
	"log/slog"
	"os/exec"
	"strconv"
	"sync"
	"syscall"
	"time"

	"github.com/NVIDIA/go-nvml/pkg/nvml"
	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
)

type nvmlClient interface {
	Init() nvml.Return
	Shutdown() nvml.Return
	DeviceGetCount() (int, nvml.Return)
}

type defaultNvmlClient struct{}

func (d *defaultNvmlClient) Init() nvml.Return {
	return nvml.Init()
}

func (d *defaultNvmlClient) Shutdown() nvml.Return {
	return nvml.Shutdown()
}

func (d *defaultNvmlClient) DeviceGetCount() (int, nvml.Return) {
	return nvml.DeviceGetCount()
}

// NCCLShimConfig configures signaling of the NCCL checkpoint/restore shim, an
// LD_PRELOAD library loaded into workload processes. When enabled, the backend
// tells the shim to destroy the workload's NCCL communicators before the
// freeze — removing the cross-process CUDA IPC state that cuda-checkpoint
// cannot preserve — and to arm lazy communicator recreation after restore.
// Workloads without the shim preloaded ignore the signals only if they handle
// them; the flag must therefore only be enabled on nodes whose CUDA-backend
// jobs run with the shim.
type NCCLShimConfig struct {
	Enabled bool
	// DestroySignal is sent to each PID before lock/checkpoint. Defaults to
	// SIGRTMIN+1 as seen by glibc (35), the shim's default.
	DestroySignal syscall.Signal
	// RecreateSignal is sent to each PID after restore. Defaults to
	// SIGRTMIN+2 as seen by glibc (36).
	RecreateSignal syscall.Signal
	// DestroyWait is how long to wait after sending DestroySignal before
	// freezing, giving the shim time to tear down all communicators.
	DestroyWait time.Duration
}

const (
	defaultShimDestroySignal  = syscall.Signal(35) // SIGRTMIN+1 under glibc
	defaultShimRecreateSignal = syscall.Signal(36) // SIGRTMIN+2 under glibc
	defaultShimDestroyWait    = 2 * time.Second
)

// CudaCheckpointOption customizes a CudaCheckpoint backend.
type CudaCheckpointOption func(*CudaCheckpoint)

// WithNCCLShim configures NCCL shim signaling around checkpoint and restore.
// Zero-valued signals and wait duration fall back to the shim defaults.
func WithNCCLShim(cfg NCCLShimConfig) CudaCheckpointOption {
	return func(c *CudaCheckpoint) {
		if cfg.DestroySignal == 0 {
			cfg.DestroySignal = defaultShimDestroySignal
		}
		if cfg.RecreateSignal == 0 {
			cfg.RecreateSignal = defaultShimRecreateSignal
		}
		if cfg.DestroyWait == 0 {
			cfg.DestroyWait = defaultShimDestroyWait
		}
		c.shim = cfg
	}
}

// CudaCheckpoint implements the Backend interface using cuda-checkpoint and optionally CRIU.
type CudaCheckpoint struct {
	mu            sync.Mutex
	execCommand   func(ctx context.Context, name string, args ...string) ([]byte, error)
	nvml          nvmlClient
	lookPath      func(string) (string, error)
	signalProcess func(pid int, sig syscall.Signal) error
	shim          NCCLShimConfig
}

// NewCudaCheckpoint creates a new CudaCheckpoint backend.
func NewCudaCheckpoint(opts ...CudaCheckpointOption) *CudaCheckpoint {
	c := &CudaCheckpoint{
		execCommand: func(ctx context.Context, name string, args ...string) ([]byte, error) {
			return exec.CommandContext(ctx, name, args...).CombinedOutput()
		},
		nvml:          &defaultNvmlClient{},
		lookPath:      exec.LookPath,
		signalProcess: syscall.Kill,
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// Snapshot triggers a snapshot of the accelerator context for a job.
func (c *CudaCheckpoint) Snapshot(ctx context.Context, req Request) error {
	pids := ExtractPIDStrings(req.Config)
	if len(pids) == 0 {
		return fmt.Errorf("at least one PID is required for CUDA snapshot")
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	slog.InfoContext(ctx, "Snapshotting PIDs", "pids", pids)

	t0 := time.Now()
	if c.shim.Enabled {
		if err := c.shimDestroyComms(ctx, pids); err != nil {
			return fmt.Errorf("nccl shim destroy failed: %w", err)
		}
	}
	if err := c.checkpointPIDs(ctx, pids); err != nil {
		return fmt.Errorf("cuda-checkpoint checkpoint failed: %w", err)
	}
	slog.InfoContext(ctx, "cuda-checkpoint action took", "duration", time.Since(t0))
	return nil
}

// Restore triggers a restoration of the accelerator context for a job.
func (c *CudaCheckpoint) Restore(ctx context.Context, req Request) error {
	pids := ExtractPIDStrings(req.Config)
	if len(pids) == 0 {
		return fmt.Errorf("at least one PID is required for CUDA restore")
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	slog.InfoContext(ctx, "Restoring PIDs", "pids", pids)
	t0 := time.Now()
	if err := c.restorePIDs(ctx, pids); err != nil {
		return fmt.Errorf("cuda-checkpoint toggle failed: %w", err)
	}
	if c.shim.Enabled {
		if err := c.shimArmRecreate(ctx, pids); err != nil {
			return fmt.Errorf("nccl shim recreate failed: %w", err)
		}
	}
	slog.InfoContext(ctx, "cuda-checkpoint toggle took", "duration", time.Since(t0), "pids", pids)
	return nil
}

// ExtractPIDStrings extracts PID strings from a BackendConfig.
func ExtractPIDStrings(config *pb.BackendConfig) []string {
	if config == nil {
		return nil
	}
	cuda := config.GetCuda()
	if cuda == nil {
		return nil
	}
	target := cuda.GetExplicitTarget()
	if target == nil {
		return nil
	}
	pids := make([]string, 0, len(target.GetPids()))
	for _, pid := range target.GetPids() {
		pids = append(pids, strconv.Itoa(int(pid)))
	}
	return pids
}

// BuildCudaConfig wraps PID strings into a BackendConfig.
func BuildCudaConfig(pidStrings []string) *pb.BackendConfig {
	pids := make([]int32, 0, len(pidStrings))
	for _, s := range pidStrings {
		if pid, err := strconv.ParseInt(s, 10, 32); err == nil {
			pids = append(pids, int32(pid))
		}
	}
	return &pb.BackendConfig{
		Backend: &pb.BackendConfig_Cuda{
			Cuda: &pb.CudaBackendConfig{
				ExplicitTarget: &pb.ProcessTarget{Pids: pids},
			},
		},
	}
}

func (c *CudaCheckpoint) getCudaCheckpointPath() string {
	// First check if it's in the PATH
	if path, err := exec.LookPath("cuda-checkpoint"); err == nil {
		return path
	}
	// Fallback to the relative path used in development
	return "/usr/local/bin/cuda-checkpoint"
}

func (c *CudaCheckpoint) runSudoCommand(ctx context.Context, name string, args ...string) error {
	if out, err := c.execCommand(ctx, name, args...); err != nil {
		return fmt.Errorf("command failed: %w, output: %s", err, string(out))
	}
	return nil
}

func (c *CudaCheckpoint) checkpointPIDs(ctx context.Context, pids []string) error {
	binaryPath := c.getCudaCheckpointPath()
	pidArgs := make([]string, 0, 2*len(pids))
	for _, pid := range pids {
		pidArgs = append(pidArgs, "--pid", pid)
	}
	if err := c.runSudoCommand(ctx, binaryPath, append([]string{"--action", "lock"}, pidArgs...)...); err != nil {
		return fmt.Errorf("cuda-checkpoint lock failed: %w", err)
	}
	if err := c.runSudoCommand(ctx, binaryPath, append([]string{"--action", "checkpoint"}, pidArgs...)...); err != nil {
		return fmt.Errorf("cuda-checkpoint checkpoint failed: %w", err)
	}
	return nil
}

// shimDestroyComms tells the shim in each workload process to destroy its
// NCCL communicators, then waits DestroyWait so the subsequent freeze only
// sees process-private CUDA state. The workload must already be quiesced (at
// a Yield boundary) — the shim's destroy is unsafe mid-collective.
func (c *CudaCheckpoint) shimDestroyComms(ctx context.Context, pids []string) error {
	if err := c.signalPIDs(pids, c.shim.DestroySignal); err != nil {
		return err
	}
	slog.InfoContext(ctx, "Signaled NCCL shim to destroy communicators",
		"pids", pids, "signal", c.shim.DestroySignal, "wait", c.shim.DestroyWait)
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(c.shim.DestroyWait):
	}
	return nil
}

// shimArmRecreate tells the shim in each restored process to lazily recreate
// its NCCL communicators on the next collective call.
func (c *CudaCheckpoint) shimArmRecreate(ctx context.Context, pids []string) error {
	if err := c.signalPIDs(pids, c.shim.RecreateSignal); err != nil {
		return err
	}
	slog.InfoContext(ctx, "Signaled NCCL shim to arm communicator recreation",
		"pids", pids, "signal", c.shim.RecreateSignal)
	return nil
}

func (c *CudaCheckpoint) signalPIDs(pids []string, sig syscall.Signal) error {
	for _, pidStr := range pids {
		pid, err := strconv.Atoi(pidStr)
		if err != nil {
			return fmt.Errorf("invalid PID %q: %w", pidStr, err)
		}
		if err := c.signalProcess(pid, sig); err != nil {
			return fmt.Errorf("signaling PID %d with signal %d: %w", pid, sig, err)
		}
	}
	return nil
}

func (c *CudaCheckpoint) restorePIDs(ctx context.Context, pids []string) error {
	binaryPath := c.getCudaCheckpointPath()
	pidArgs := make([]string, 0, 2*len(pids))
	for _, pid := range pids {
		pidArgs = append(pidArgs, "--pid", pid)
	}
	if err := c.runSudoCommand(ctx, binaryPath, append([]string{"--toggle"}, pidArgs...)...); err != nil {
		return fmt.Errorf("cuda-checkpoint toggle failed: %w", err)
	}
	return nil
}

// HealthCheck checks if the cuda-checkpoint backend is healthy by initializing the backend
// and the discovery provider.
func (c *CudaCheckpoint) HealthCheck(ctx context.Context) error {
	// 1. Check if cuda-checkpoint executable is available
	binaryPath := c.getCudaCheckpointPath()
	if _, err := c.lookPath(binaryPath); err != nil {
		return fmt.Errorf("cuda-checkpoint executable not found: %w", err)
	}

	// 2. Initialize NVML
	if ret := c.nvml.Init(); ret != nvml.SUCCESS {
		return fmt.Errorf("failed to initialize NVML: %v", nvml.ErrorString(ret))
	}
	defer func() {
		if ret := c.nvml.Shutdown(); ret != nvml.SUCCESS {
			slog.ErrorContext(ctx, "Failed to shutdown NVML", "error", nvml.ErrorString(ret))
		}
	}()

	// 3. Check if there are any GPUs attached to the system
	count, ret := c.nvml.DeviceGetCount()
	if ret != nvml.SUCCESS {
		return fmt.Errorf("failed to get device count: %v", nvml.ErrorString(ret))
	}

	if count == 0 {
		return fmt.Errorf("no GPUs found on the system")
	}

	return nil
}
