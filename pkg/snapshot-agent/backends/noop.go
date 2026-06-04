package backends

import (
	"context"
	"log"
	"time"
)

// NoopBackend is a dummy implementation of the Backend interface.
type NoopBackend struct{}

// NewNoopBackend creates a new NoopBackend instance.
func NewNoopBackend() *NoopBackend {
	return &NoopBackend{}
}

// Snapshot simulates a snapshot operation.
func (b *NoopBackend) Snapshot(ctx context.Context, pids []string) (int64, int64, error) {
	log.Printf("NoopBackend: Snapshot called for PIDs %v", pids)
	// Simulate some work
	time.Sleep(500 * time.Millisecond)
	return 100, 200, nil
}

// Restore simulates a restore operation.
func (b *NoopBackend) Restore(ctx context.Context, pids []string) error {
	log.Printf("NoopBackend: Restore called for PIDs %v", pids)
	// Simulate some work
	time.Sleep(500 * time.Millisecond)
	return nil
}
