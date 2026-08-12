package backends_test

import (
	"context"
	"fmt"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/NVIDIA/go-nvml/pkg/nvml"
	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/backends"
)

func cudaConfig(pids ...int32) *pb.BackendConfig {
	return &pb.BackendConfig{
		Backend: &pb.BackendConfig_Cuda{
			Cuda: &pb.CudaBackendConfig{
				ExplicitTarget: &pb.ProcessTarget{Pids: pids},
			},
		},
	}
}

type mockNvmlClient struct {
	initRet        nvml.Return
	shutdownRet    nvml.Return
	deviceCount    int
	deviceCountRet nvml.Return
}

func (m *mockNvmlClient) Init() nvml.Return                  { return m.initRet }
func (m *mockNvmlClient) Shutdown() nvml.Return              { return m.shutdownRet }
func (m *mockNvmlClient) DeviceGetCount() (int, nvml.Return) { return m.deviceCount, m.deviceCountRet }

func TestNewCudaCheckpoint(t *testing.T) {
	c := backends.NewCudaCheckpoint()
	if c == nil {
		t.Fatal("NewCudaCheckpoint returned nil")
	}
}

func TestSnapshot(t *testing.T) {
	tests := []struct {
		name        string
		config      *pb.BackendConfig
		execErr     error
		expectedErr bool
	}{
		{
			name:   "Success",
			config: cudaConfig(123, 456),
		},
		{
			name:        "ExecFailure",
			config:      cudaConfig(123),
			execErr:     fmt.Errorf("exec error"),
			expectedErr: true,
		},
		{
			name:        "NoPIDs",
			config:      cudaConfig(),
			expectedErr: true,
		},
		{
			name:        "NilConfig",
			config:      nil,
			expectedErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := backends.NewCudaCheckpoint()
			c.SetExecCommand(func(_ context.Context, _ string, _ ...string) ([]byte, error) {
				return nil, tt.execErr
			})

			err := c.Snapshot(context.Background(), backends.Request{JobID: "test-job", Config: tt.config})
			if (err != nil) != tt.expectedErr {
				t.Errorf("Snapshot() error = %v, expectedErr %v", err, tt.expectedErr)
			}
		})
	}
}

func TestRestore(t *testing.T) {
	tests := []struct {
		name        string
		config      *pb.BackendConfig
		execErr     error
		expectedErr bool
	}{
		{
			name:   "Success",
			config: cudaConfig(123),
		},
		{
			name:        "NoPIDs",
			config:      cudaConfig(),
			expectedErr: true,
		},
		{
			name:        "ExecFailure",
			config:      cudaConfig(123),
			execErr:     fmt.Errorf("exec error"),
			expectedErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := backends.NewCudaCheckpoint()
			c.SetExecCommand(func(_ context.Context, _ string, _ ...string) ([]byte, error) {
				return nil, tt.execErr
			})

			err := c.Restore(context.Background(), backends.Request{JobID: "test-job", Config: tt.config})
			if (err != nil) != tt.expectedErr {
				t.Errorf("Restore() error = %v, expectedErr %v", err, tt.expectedErr)
			}
		})
	}
}

// shimRecorder tracks the interleaving of shim signals and cuda-checkpoint
// invocations so tests can assert ordering.
type shimRecorder struct {
	calls []string
}

func (r *shimRecorder) trackedBackend(shim backends.NCCLShimConfig) *backends.CudaCheckpoint {
	c := backends.NewCudaCheckpoint(backends.WithNCCLShim(shim))
	c.SetExecCommand(func(_ context.Context, _ string, args ...string) ([]byte, error) {
		r.calls = append(r.calls, "exec:"+strings.Join(args, " "))
		return nil, nil
	})
	c.SetSignalProcess(func(pid int, sig syscall.Signal) error {
		r.calls = append(r.calls, fmt.Sprintf("signal:%d:%d", pid, sig))
		return nil
	})
	return c
}

func TestSnapshotWithNCCLShim(t *testing.T) {
	rec := &shimRecorder{}
	c := rec.trackedBackend(backends.NCCLShimConfig{Enabled: true, DestroyWait: time.Millisecond})

	// Pass PIDs in reverse order to verify ascending sort normalization
	if err := c.Snapshot(context.Background(), backends.Request{JobID: "test-job", Config: cudaConfig(456, 123)}); err != nil {
		t.Fatalf("Snapshot() error = %v", err)
	}

	want := []string{
		"signal:123:35",
		"signal:456:35",
		"exec:--action lock --pid 123",
		"exec:--action lock --pid 456",
		"exec:--action checkpoint --pid 123",
		"exec:--action checkpoint --pid 456",
	}
	if fmt.Sprint(rec.calls) != fmt.Sprint(want) {
		t.Errorf("call order = %v, want %v", rec.calls, want)
	}
}

func TestRestoreWithNCCLShim(t *testing.T) {
	rec := &shimRecorder{}
	c := rec.trackedBackend(backends.NCCLShimConfig{Enabled: true, DestroyWait: time.Millisecond})

	// Pass PIDs in reverse order to verify ascending sort normalization
	if err := c.Restore(context.Background(), backends.Request{JobID: "test-job", Config: cudaConfig(456, 123)}); err != nil {
		t.Fatalf("Restore() error = %v", err)
	}

	want := []string{
		"exec:--toggle --pid 123",
		"exec:--toggle --pid 456",
		"signal:123:36",
		"signal:456:36",
	}
	if fmt.Sprint(rec.calls) != fmt.Sprint(want) {
		t.Errorf("call order = %v, want %v", rec.calls, want)
	}
}

func TestSnapshotShimDisabledSendsNoSignals(t *testing.T) {
	rec := &shimRecorder{}
	c := backends.NewCudaCheckpoint()
	c.SetExecCommand(func(_ context.Context, _ string, _ ...string) ([]byte, error) {
		return nil, nil
	})
	c.SetSignalProcess(func(pid int, sig syscall.Signal) error {
		rec.calls = append(rec.calls, fmt.Sprintf("signal:%d:%d", pid, sig))
		return nil
	})

	if err := c.Snapshot(context.Background(), backends.Request{JobID: "test-job", Config: cudaConfig(123)}); err != nil {
		t.Fatalf("Snapshot() error = %v", err)
	}
	if len(rec.calls) != 0 {
		t.Errorf("expected no signals with shim disabled, got %v", rec.calls)
	}
}

func TestSnapshotShimSignalFailure(t *testing.T) {
	rec := &shimRecorder{}
	c := rec.trackedBackend(backends.NCCLShimConfig{Enabled: true, DestroyWait: time.Millisecond})
	c.SetSignalProcess(func(int, syscall.Signal) error {
		return fmt.Errorf("no such process")
	})

	if err := c.Snapshot(context.Background(), backends.Request{JobID: "test-job", Config: cudaConfig(123)}); err == nil {
		t.Fatal("Snapshot() expected error when shim signaling fails")
	}
	for _, call := range rec.calls {
		if strings.HasPrefix(call, "exec:") {
			t.Errorf("cuda-checkpoint must not run after shim signal failure, got %v", rec.calls)
		}
	}
}

func TestHealthCheck(t *testing.T) {
	tests := []struct {
		name           string
		initRet        nvml.Return
		deviceCount    int
		deviceCountRet nvml.Return
		expectedErr    bool
	}{
		{
			name:           "Success",
			initRet:        nvml.SUCCESS,
			deviceCount:    1,
			deviceCountRet: nvml.SUCCESS,
		},
		{
			name:        "NVMLInitFailure",
			initRet:     nvml.ERROR_LIBRARY_NOT_FOUND,
			expectedErr: true,
		},
		{
			name:           "NoGPUs",
			initRet:        nvml.SUCCESS,
			deviceCount:    0,
			deviceCountRet: nvml.SUCCESS,
			expectedErr:    true,
		},
		{
			name:           "DeviceCountFailure",
			initRet:        nvml.SUCCESS,
			deviceCount:    0,
			deviceCountRet: nvml.ERROR_UNKNOWN,
			expectedErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := backends.NewCudaCheckpoint()
			c.SetLookPath(func(path string) (string, error) {
				return path, nil
			})
			c.SetNvmlClient(&mockNvmlClient{
				initRet:        tt.initRet,
				shutdownRet:    nvml.SUCCESS,
				deviceCount:    tt.deviceCount,
				deviceCountRet: tt.deviceCountRet,
			})

			err := c.HealthCheck(context.Background())
			if (err != nil) != tt.expectedErr {
				t.Errorf("HealthCheck() error = %v, expectedErr %v", err, tt.expectedErr)
			}
		})
	}
}
