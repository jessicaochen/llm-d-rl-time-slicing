package controller_test

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/api/v1alpha1"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/controller"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/store"
	agentpb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
	"k8s.io/client-go/util/workqueue"
)

type mockInfrastructureOrchestrator struct {
	initFunc    func(ctx context.Context) error
	observeFunc func(ctx context.Context, groupID string) error
}

func (m *mockInfrastructureOrchestrator) Init(ctx context.Context) error {
	if m.initFunc != nil {
		return m.initFunc(ctx)
	}
	return nil
}

func (m *mockInfrastructureOrchestrator) ObserveGroupState(ctx context.Context, groupID string) error {
	if m.observeFunc != nil {
		return m.observeFunc(ctx, groupID)
	}
	return nil
}

// TestController_ReconcileSuccess verifies that the controller calls ObserveGroupState
// and successfully processes the item.
func TestController_ReconcileSuccess(t *testing.T) {
	lockStore := store.NewMemLockStore()
	groupStore := store.NewGroupStore(lockStore)
	jobStore := store.NewJobStore()
	queue := workqueue.NewTypedRateLimitingQueueWithConfig(
		workqueue.DefaultTypedControllerRateLimiter[string](),
		workqueue.TypedRateLimitingQueueConfig[string]{Name: "test"},
	)

	observeCalled := make(chan string, 1)
	mockOrch := &mockInfrastructureOrchestrator{
		observeFunc: func(ctx context.Context, groupID string) error {
			_, _, err := groupStore.GetOrCreate(ctx, groupID)
			if err != nil {
				return err
			}
			observeCalled <- groupID
			return nil
		},
	}

	mockAgentStore := &controller.MockSnapshotAgentStore{}
	c := controller.NewController(groupStore, jobStore, queue, mockOrch, mockAgentStore)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start the controller
	go func() {
		if err := c.Run(ctx, 1); err != nil {
			t.Errorf("Controller Run failed: %v", err)
		}
	}()

	// Add an item to the queue
	queue.Add("group-1")

	// Wait for ObserveGroupState to be called
	select {
	case groupID := <-observeCalled:
		if groupID != "group-1" {
			t.Errorf("Expected ObserveGroupState to be called for group-1, got %s", groupID)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Timed out waiting for ObserveGroupState to be called")
	}

	// Verify the item was marked Done (queue length should become 0)
	time.Sleep(100 * time.Millisecond)
	if queue.Len() != 0 {
		t.Errorf("Expected queue to be empty, got length %d", queue.Len())
	}
}

// TestController_ReconcileFailure_Retries verifies that if ObserveGroupState fails,
// the controller retries by adding the item back to the queue (rate limited).
func TestController_ReconcileFailure_Retries(t *testing.T) {
	lockStore := store.NewMemLockStore()
	groupStore := store.NewGroupStore(lockStore)
	jobStore := store.NewJobStore()

	testQueue := &trackQueue{
		TypedRateLimitingInterface: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{Name: "test"},
		),
	}

	observeCalled := make(chan struct{})
	mockOrch := &mockInfrastructureOrchestrator{
		observeFunc: func(ctx context.Context, groupID string) error {
			close(observeCalled)
			return errors.New("observe failed")
		},
	}

	mockAgentStore := &controller.MockSnapshotAgentStore{}
	c := controller.NewController(groupStore, jobStore, testQueue, mockOrch, mockAgentStore)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go func() {
		if err := c.Run(ctx, 1); err != nil {
			t.Errorf("Controller Run failed: %v", err)
		}
	}()

	testQueue.Add("group-1")

	<-observeCalled

	// Wait for the item to be re-added
	err := waitWithTimeout(func() bool {
		return testQueue.getAddRateLimitedCount() > 0
	}, 2*time.Second)
	if err != nil {
		t.Fatal("Timed out waiting for item to be re-queued")
	}

	if testQueue.getAddRateLimitedCount() != 1 {
		t.Errorf("Expected AddRateLimited to be called once, got %d", testQueue.getAddRateLimitedCount())
	}
}

type trackQueue struct {
	workqueue.TypedRateLimitingInterface[string]
	mu                  sync.Mutex
	addRateLimitedCount int
	doneCount           int
}

func (t *trackQueue) AddRateLimited(item string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.addRateLimitedCount++
	t.TypedRateLimitingInterface.AddRateLimited(item)
}

func (t *trackQueue) getAddRateLimitedCount() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.addRateLimitedCount
}

func (t *trackQueue) Done(item string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.doneCount++
	t.TypedRateLimitingInterface.Done(item)
}

func (t *trackQueue) getDoneCount() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.doneCount
}

func waitWithTimeout(f func() bool, timeout time.Duration) error {
	ch := make(chan struct{})
	go func() {
		for {
			if f() {
				close(ch)
				return
			}
			time.Sleep(50 * time.Millisecond)
		}
	}()
	select {
	case <-ch:
		return nil
	case <-time.After(timeout):
		return errors.New("timeout")
	}
}

func TestController_Reconcile_TwoJobsTakeLockTurns(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	lockStore := store.NewMemLockStore()
	groupStore := store.NewGroupStore(lockStore)
	jobStore := store.NewJobStore()
	testQueue := &trackQueue{
		TypedRateLimitingInterface: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{Name: "test"},
		),
	}

	groupID := "group-1"

	// Mock ObserveGroupState (no-op, we don't need to sync lock state because
	// in-memory and lockStore are kept in sync by GroupSpec methods).
	mockOrch := &mockInfrastructureOrchestrator{
		observeFunc: func(ctx context.Context, gID string) error {
			return nil
		},
	}

	mockAgentStore := &controller.MockSnapshotAgentStore{}
	c := controller.NewController(groupStore, jobStore, testQueue, mockOrch, mockAgentStore)

	// Start the controller
	go func() {
		if err := c.Run(ctx, 1); err != nil {
			t.Errorf("Controller Run failed: %v", err)
		}
	}()

	// 1. Pre-lock the group to "job-1" in lockStore, then create it.
	// This simulates starting with a locked group.
	if err := lockStore.Lock(ctx, groupID, "job-1"); err != nil {
		t.Fatalf("failed to lock in store: %v", err)
	}
	testGroup, _, err := groupStore.GetOrCreate(ctx, groupID)
	if err != nil {
		t.Fatalf("failed to create group: %v", err)
	}

	// Reconcile Phase 1 (job-1 locked)
	initialDone := testQueue.getDoneCount()
	testQueue.Add(groupID)
	err = waitWithTimeout(func() bool { return testQueue.getDoneCount() > initialDone }, 2*time.Second)
	if err != nil {
		t.Fatalf("Timed out waiting for Phase 1 reconcile: %v", err)
	}
	if testGroup.Spec().LockingJob() != "job-1" || testGroup.Spec().ActiveJob() != "job-1" {
		t.Errorf("Phase 1: expected lockingJob=job-1, activeJob=job-1; got lockingJob=%q, activeJob=%q",
			testGroup.Spec().LockingJob(), testGroup.Spec().ActiveJob())
	}

	// 2. job-2 requests lock and gets in queue
	testGroup.Spec().RequestLock("job-2")

	// Reconcile Phase 2 (job-2 enqueued, job-1 still locked)
	initialDone = testQueue.getDoneCount()
	testQueue.Add(groupID)
	err = waitWithTimeout(func() bool { return testQueue.getDoneCount() > initialDone }, 2*time.Second)
	if err != nil {
		t.Fatalf("Timed out waiting for Phase 2 reconcile: %v", err)
	}
	if testGroup.Spec().LockingJob() != "job-1" || testGroup.Spec().ActiveJob() != "job-1" {
		t.Errorf("Phase 2: expected lockingJob=job-1, activeJob=job-1; got lockingJob=%q, activeJob=%q",
			testGroup.Spec().LockingJob(), testGroup.Spec().ActiveJob())
	}
	if !testGroup.Spec().GetWaitingJobQueue().Exists("job-2") {
		t.Errorf("Phase 2: expected job-2 to be in queue")
	}

	// 3. job-1 yields the lock -> job-2 should get the lock
	err = testGroup.Spec().Yield(ctx, "job-1")
	if err != nil {
		t.Fatalf("Yield failed: %v", err)
	}

	// Reconcile Phase 3 (job-2 should be active/locking)
	testQueue.Add(groupID)
	err = waitWithTimeout(func() bool {
		return testGroup.Spec().LockingJob() == "job-2" && testGroup.Spec().ActiveJob() == "job-2"
	}, 3*time.Second)
	if err != nil {
		t.Fatalf("Timed out waiting for Phase 3 reconcile "+
			"(expected lockingJob=job-2, activeJob=job-2): %v. "+
			"Current state: lockingJob=%q, activeJob=%q",
			err, testGroup.Spec().LockingJob(), testGroup.Spec().ActiveJob())
	}
	if testGroup.Spec().GetWaitingJobQueue().Exists("job-2") {
		t.Errorf("Phase 3: expected job-2 to be dequeued")
	}

	// 4. job-1 requests lock again (gets in queue)
	testGroup.Spec().RequestLock("job-1")

	// Reconcile Phase 4 (job-1 enqueued, job-2 still locked)
	initialDone = testQueue.getDoneCount()
	testQueue.Add(groupID)
	err = waitWithTimeout(func() bool { return testQueue.getDoneCount() > initialDone }, 2*time.Second)
	if err != nil {
		t.Fatalf("Timed out waiting for Phase 4 reconcile: %v", err)
	}
	if testGroup.Spec().LockingJob() != "job-2" || testGroup.Spec().ActiveJob() != "job-2" {
		t.Errorf("Phase 4: expected lockingJob=job-2, activeJob=job-2; got lockingJob=%q, activeJob=%q",
			testGroup.Spec().LockingJob(), testGroup.Spec().ActiveJob())
	}
	if !testGroup.Spec().GetWaitingJobQueue().Exists("job-1") {
		t.Errorf("Phase 4: expected job-1 to be in queue")
	}

	// 5. job-2 yields the lock -> job-1 should get the lock again
	err = testGroup.Spec().Yield(ctx, "job-2")
	if err != nil {
		t.Fatalf("Yield failed: %v", err)
	}

	// Reconcile Phase 5 (job-1 should be active/locking again)
	testQueue.Add(groupID)
	err = waitWithTimeout(func() bool {
		return testGroup.Spec().LockingJob() == "job-1" && testGroup.Spec().ActiveJob() == "job-1"
	}, 3*time.Second)
	if err != nil {
		t.Fatalf("Timed out waiting for Phase 5 reconcile "+
			"(expected lockingJob=job-1, activeJob=job-1): %v. "+
			"Current state: lockingJob=%q, activeJob=%q",
			err, testGroup.Spec().LockingJob(), testGroup.Spec().ActiveJob())
	}
	if testGroup.Spec().GetWaitingJobQueue().Exists("job-1") {
		t.Errorf("Phase 5: expected job-1 to be dequeued")
	}
}

func TestController_Reconcile_OneJobLoopRemainsActive(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	lockStore := store.NewMemLockStore()
	groupStore := store.NewGroupStore(lockStore)
	jobStore := store.NewJobStore()
	queue := workqueue.NewTypedRateLimitingQueueWithConfig(
		workqueue.DefaultTypedControllerRateLimiter[string](),
		workqueue.TypedRateLimitingQueueConfig[string]{Name: "test"},
	)

	groupID := "group-1"

	// Mock ObserveGroupState to notify when reconcile runs.
	observeCalled := make(chan struct{}, 10)
	mockOrch := &mockInfrastructureOrchestrator{
		observeFunc: func(ctx context.Context, gID string) error {
			observeCalled <- struct{}{}
			return nil
		},
	}

	mockAgentStore := &controller.MockSnapshotAgentStore{}
	c := controller.NewController(groupStore, jobStore, queue, mockOrch, mockAgentStore)

	// Start the controller
	go func() {
		if err := c.Run(ctx, 1); err != nil {
			t.Errorf("Controller Run failed: %v", err)
		}
	}()

	// 1. Pre-lock the group to "job-1" in lockStore, then create it.
	if err := lockStore.Lock(ctx, groupID, "job-1"); err != nil {
		t.Fatalf("failed to lock job-1: %v", err)
	}
	testGroup, _, err := groupStore.GetOrCreate(ctx, groupID)
	if err != nil {
		t.Fatalf("failed to create group: %v", err)
	}

	// Reconcile Lock
	queue.Add(groupID)
	select {
	case <-observeCalled:
	case <-time.After(2 * time.Second):
		t.Fatalf("Timed out waiting for Lock reconcile")
	}

	// Verify Locked State
	if testGroup.Spec().LockingJob() != "job-1" || testGroup.Spec().ActiveJob() != "job-1" {
		t.Errorf("Expected lockingJob=job-1, activeJob=job-1; got lockingJob=%q, activeJob=%q",
			testGroup.Spec().LockingJob(), testGroup.Spec().ActiveJob())
	}

	// 2. Yield job-1 (no waiters, so it just unlocks)
	err = testGroup.Spec().Yield(ctx, "job-1")
	if err != nil {
		t.Fatalf("Yield failed: %v", err)
	}

	// Reconcile Yield
	queue.Add(groupID)
	select {
	case <-observeCalled:
	case <-time.After(2 * time.Second):
		t.Fatalf("Timed out waiting for Yield reconcile")
	}

	// Verify Yielded State: lockingJob is "", but activeJob REMAINS "job-1" (warm!)
	if testGroup.Spec().LockingJob() != "" {
		t.Errorf("Expected lockingJob to be empty, got %q", testGroup.Spec().LockingJob())
	}
	if testGroup.Spec().ActiveJob() != "job-1" {
		t.Errorf("Expected activeJob to remain job-1, got %q", testGroup.Spec().ActiveJob())
	}
}

func TestController_ObserveJobContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	lockStore := store.NewMemLockStore()
	groupStore := store.NewGroupStore(lockStore)
	jobStore := store.NewJobStore()

	groupID := "group-1"
	nodeName := "node-1"

	// 1. Setup group and nodes in store
	g, _, err := groupStore.GetOrCreate(ctx, groupID)
	if err != nil {
		t.Fatalf("failed to create group: %v", err)
	}
	g.Status().SetNodes([]string{nodeName})

	// 2. Setup job in store (must exist for context to be updated)
	job := store.NewJob(groupID, "job-1")
	if err := jobStore.Put(ctx, job); err != nil {
		t.Fatalf("failed to put job: %v", err)
	}

	// 3. Mock SnapshotAgentStore to return status
	mockAgentStore := &controller.MockSnapshotAgentStore{
		GetStatusFunc: func(ctx context.Context, node string) (*agentpb.StatusResponse, error) {
			if node == nodeName {
				return &agentpb.StatusResponse{
					JobStatuses: []*agentpb.JobStatus{
						{JobId: "job-1", State: agentpb.JobState_JOB_STATE_RUNNING},
					},
				}, nil
			}
			return &agentpb.StatusResponse{}, nil
		},
	}

	c := controller.NewController(groupStore, jobStore, nil, nil, mockAgentStore)

	// 4. Call ObserveJobContext
	err = c.ObserveJobContext(ctx, groupID)
	if err != nil {
		t.Fatalf("ObserveJobContext failed: %v", err)
	}

	// 5. Verify job context state is updated
	updatedJob, err := jobStore.Get(ctx, groupID, "job-1")
	if err != nil {
		t.Fatalf("failed to get job: %v", err)
	}
	state, ok := updatedJob.ContextState()[nodeName]
	if !ok {
		t.Fatalf("Expected context state for job-1 on node-1 to exist")
	}
	if state != pb.SnapshotAgentJobState_STATE_RUNNING {
		t.Errorf("Expected job-1 state to be RUNNING, got %v", state)
	}
}

type reconcileTestCase struct {
	name         string
	activeJobID  string
	lockingJobID string
	initialJobs  []*store.Job
	nodes        []string

	// Mock Agent Behavior
	initialAgentStatuses map[string][]*agentpb.JobStatus
	postOpAgentStatuses  map[string][]*agentpb.JobStatus

	// Expected Actions
	expectedSnapshotJob  string
	expectedSnapshotNode string
	expectedRestoreJob   string
	expectedRestoreNode  string

	// Expected Outcomes
	expectedLoadedJob  string
	expectedGroupState pb.GroupStatus_State
	expectFailure      bool
}

func TestController_Reconcile(t *testing.T) {
	// Helper to make jobs easily in table definition
	makeJob := func(groupID, jobID string, nodeStates map[string]pb.SnapshotAgentJobState_State) *store.Job {
		j := store.NewJob(groupID, jobID)
		for node, state := range nodeStates {
			j.UpdateContextState(node, state)
		}
		return j
	}

	testCases := []reconcileTestCase{
		{
			name:        "Snapshot_EvictRunningJob_1Node",
			activeJobID: "job-1",
			nodes:       []string{"node-1"},
			initialJobs: []*store.Job{
				makeJob("group-1", "job-1", nil),
				makeJob("group-1", "job-2", map[string]pb.SnapshotAgentJobState_State{"node-1": pb.SnapshotAgentJobState_STATE_RUNNING}),
			},
			initialAgentStatuses: map[string][]*agentpb.JobStatus{
				"node-1": {
					{JobId: "job-2", State: agentpb.JobState_JOB_STATE_RUNNING},
					{JobId: "job-1", State: agentpb.JobState_JOB_STATE_IDLE},
				},
			},
			postOpAgentStatuses: map[string][]*agentpb.JobStatus{
				"node-1": {
					{JobId: "job-2", State: agentpb.JobState_JOB_STATE_SAVED},
					{JobId: "job-1", State: agentpb.JobState_JOB_STATE_IDLE},
				},
			},
			expectedSnapshotJob:  "job-2",
			expectedSnapshotNode: "node-1",
			expectedLoadedJob:    "job-1",
			expectedGroupState:   pb.GroupStatus_STATE_IDLE_YIELDED,
		},
		{
			name:        "Error_ActiveJobFaulted",
			activeJobID: "job-1",
			nodes:       []string{"node-1"},
			initialJobs: []*store.Job{
				makeJob("group-1", "job-1", map[string]pb.SnapshotAgentJobState_State{"node-1": pb.SnapshotAgentJobState_STATE_FAULTED}),
			},
			expectFailure: true,
		},
		{
			name:        "Restore_SavedActiveJob_1Node",
			activeJobID: "job-1",
			nodes:       []string{"node-1"},
			initialJobs: []*store.Job{
				makeJob("group-1", "job-1", map[string]pb.SnapshotAgentJobState_State{"node-1": pb.SnapshotAgentJobState_STATE_SAVED}),
			},
			initialAgentStatuses: map[string][]*agentpb.JobStatus{
				"node-1": {
					{JobId: "job-1", State: agentpb.JobState_JOB_STATE_SAVED},
				},
			},
			postOpAgentStatuses: map[string][]*agentpb.JobStatus{
				"node-1": {
					{JobId: "job-1", State: agentpb.JobState_JOB_STATE_RUNNING},
				},
			},
			expectedRestoreJob:  "job-1",
			expectedRestoreNode: "node-1",
			expectedLoadedJob:   "job-1",
			expectedGroupState:  pb.GroupStatus_STATE_IDLE_YIELDED,
		},
		{
			name:        "NoOp_ActiveJobAlreadyRunning_1Node",
			activeJobID: "job-1",
			nodes:       []string{"node-1"},
			initialJobs: []*store.Job{
				makeJob("group-1", "job-1", map[string]pb.SnapshotAgentJobState_State{"node-1": pb.SnapshotAgentJobState_STATE_RUNNING}),
			},
			initialAgentStatuses: map[string][]*agentpb.JobStatus{
				"node-1": {
					{JobId: "job-1", State: agentpb.JobState_JOB_STATE_RUNNING},
				},
			},
			expectedLoadedJob:  "job-1",
			expectedGroupState: pb.GroupStatus_STATE_IDLE_YIELDED,
		},
		{
			name:        "NoOp_ActiveJobAlreadyRunning_2Nodes",
			activeJobID: "job-1",
			initialJobs: []*store.Job{
				makeJob("group-1", "job-1", map[string]pb.SnapshotAgentJobState_State{
					"node-1": pb.SnapshotAgentJobState_STATE_RUNNING,
					"node-2": pb.SnapshotAgentJobState_STATE_RUNNING,
				}),
			},
			initialAgentStatuses: map[string][]*agentpb.JobStatus{
				"node-1": {{JobId: "job-1", State: agentpb.JobState_JOB_STATE_RUNNING}},
				"node-2": {{JobId: "job-1", State: agentpb.JobState_JOB_STATE_RUNNING}},
			},
			expectedLoadedJob:  "job-1",
			expectedGroupState: pb.GroupStatus_STATE_IDLE_YIELDED,
		},
		{
			name:        "Restore_PartiallySavedActiveJob",
			activeJobID: "job-1",
			initialJobs: []*store.Job{
				makeJob("group-1", "job-1", map[string]pb.SnapshotAgentJobState_State{
					"node-1": pb.SnapshotAgentJobState_STATE_RUNNING,
					"node-2": pb.SnapshotAgentJobState_STATE_SAVED,
				}),
			},
			initialAgentStatuses: map[string][]*agentpb.JobStatus{
				"node-1": {{JobId: "job-1", State: agentpb.JobState_JOB_STATE_RUNNING}},
				"node-2": {{JobId: "job-1", State: agentpb.JobState_JOB_STATE_SAVED}},
			},
			postOpAgentStatuses: map[string][]*agentpb.JobStatus{
				"node-2": {{JobId: "job-1", State: agentpb.JobState_JOB_STATE_RUNNING}},
			},
			expectedRestoreJob:  "job-1",
			expectedRestoreNode: "node-2",
			expectedLoadedJob:   "job-1",
			expectedGroupState:  pb.GroupStatus_STATE_IDLE_YIELDED,
		},
		{
			name:        "NoOp_NewActiveJobNoRestore",
			activeJobID: "job-1",
			initialJobs: []*store.Job{
				makeJob("group-1", "job-1", map[string]pb.SnapshotAgentJobState_State{
					"node-1": pb.SnapshotAgentJobState_STATE_RUNNING,
				}),
			},
			initialAgentStatuses: map[string][]*agentpb.JobStatus{
				"node-1": {{JobId: "job-1", State: agentpb.JobState_JOB_STATE_RUNNING}},
				"node-2": {},
			},
			expectedLoadedJob:  "job-1",
			expectedGroupState: pb.GroupStatus_STATE_IDLE_YIELDED,
		},
		{
			name:        "Snapshot_EvictRunningForActiveJob",
			activeJobID: "job-1",
			initialJobs: []*store.Job{
				makeJob("group-1", "job-1", map[string]pb.SnapshotAgentJobState_State{
					"node-1": pb.SnapshotAgentJobState_STATE_RUNNING,
				}),
				makeJob("group-1", "job-2", map[string]pb.SnapshotAgentJobState_State{
					"node-2": pb.SnapshotAgentJobState_STATE_RUNNING,
				}),
			},
			initialAgentStatuses: map[string][]*agentpb.JobStatus{
				"node-1": {{JobId: "job-1", State: agentpb.JobState_JOB_STATE_RUNNING}},
				"node-2": {{JobId: "job-2", State: agentpb.JobState_JOB_STATE_RUNNING}},
			},
			postOpAgentStatuses: map[string][]*agentpb.JobStatus{
				"node-2": {{JobId: "job-2", State: agentpb.JobState_JOB_STATE_SAVED}},
			},
			expectedSnapshotJob:  "job-2",
			expectedSnapshotNode: "node-2",
			expectedLoadedJob:    "job-1",
			expectedGroupState:   pb.GroupStatus_STATE_IDLE_YIELDED,
		},
		{
			name:        "Snapshot_EvictIdleForActiveJob",
			activeJobID: "job-1",
			initialJobs: []*store.Job{
				makeJob("group-1", "job-1", map[string]pb.SnapshotAgentJobState_State{
					"node-1": pb.SnapshotAgentJobState_STATE_RUNNING,
				}),
				makeJob("group-1", "job-2", map[string]pb.SnapshotAgentJobState_State{
					"node-2": pb.SnapshotAgentJobState_STATE_IDLE,
				}),
			},
			initialAgentStatuses: map[string][]*agentpb.JobStatus{
				"node-1": {{JobId: "job-1", State: agentpb.JobState_JOB_STATE_RUNNING}},
				"node-2": {{JobId: "job-2", State: agentpb.JobState_JOB_STATE_IDLE}},
			},
			postOpAgentStatuses: map[string][]*agentpb.JobStatus{
				"node-2": {{JobId: "job-2", State: agentpb.JobState_JOB_STATE_SAVED}},
			},
			expectedSnapshotJob:  "job-2",
			expectedSnapshotNode: "node-2",
			expectedLoadedJob:    "job-1",
			expectedGroupState:   pb.GroupStatus_STATE_IDLE_YIELDED,
		},
		{
			name:        "Error_MultipleRunningJobsOnNode",
			activeJobID: "job-1",
			initialJobs: []*store.Job{
				makeJob("group-1", "job-1", map[string]pb.SnapshotAgentJobState_State{
					"node-1": pb.SnapshotAgentJobState_STATE_RUNNING,
					"node-2": pb.SnapshotAgentJobState_STATE_SAVED,
				}),
				makeJob("group-1", "job-2", map[string]pb.SnapshotAgentJobState_State{
					"node-2": pb.SnapshotAgentJobState_STATE_RUNNING,
				}),
			},
			initialAgentStatuses: map[string][]*agentpb.JobStatus{
				"node-1": {{JobId: "job-1", State: agentpb.JobState_JOB_STATE_RUNNING}},
				"node-2": {
					{JobId: "job-1", State: agentpb.JobState_JOB_STATE_RUNNING},
					{JobId: "job-2", State: agentpb.JobState_JOB_STATE_RUNNING},
				},
			},
			expectFailure: true,
		},
		{
			name:               "NoOp_AllJobsUnspecified",
			activeJobID:        "job-1",
			expectedLoadedJob:  "job-1",
			expectedGroupState: pb.GroupStatus_STATE_IDLE_YIELDED,
		},
		{
			name:               "NoOp_NonExistentActiveJob",
			activeJobID:        "job-1",
			expectedLoadedJob:  "job-1",
			expectedGroupState: pb.GroupStatus_STATE_IDLE_YIELDED,
		},
		{
			name:        "Snapshot_EvictRunningForNewActiveJob",
			activeJobID: "job-1",
			initialJobs: []*store.Job{
				makeJob("group-1", "job-2", map[string]pb.SnapshotAgentJobState_State{
					"node-2": pb.SnapshotAgentJobState_STATE_RUNNING,
				}),
			},
			initialAgentStatuses: map[string][]*agentpb.JobStatus{
				"node-2": {{JobId: "job-2", State: agentpb.JobState_JOB_STATE_RUNNING}},
			},
			postOpAgentStatuses: map[string][]*agentpb.JobStatus{
				"node-2": {{JobId: "job-2", State: agentpb.JobState_JOB_STATE_SAVED}},
			},
			expectedSnapshotJob:  "job-2",
			expectedSnapshotNode: "node-2",
			expectedLoadedJob:    "job-1",
			expectedGroupState:   pb.GroupStatus_STATE_IDLE_YIELDED,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			lockStore := store.NewMemLockStore()
			groupStore := store.NewGroupStore(lockStore)
			jobStore := store.NewJobStore()
			testQueue := &trackQueue{
				TypedRateLimitingInterface: workqueue.NewTypedRateLimitingQueueWithConfig(
					workqueue.DefaultTypedControllerRateLimiter[string](),
					workqueue.TypedRateLimitingQueueConfig[string]{Name: "test"},
				),
			}

			groupID := "group-1"
			nodes := tc.nodes
			if len(nodes) == 0 {
				nodes = []string{"node-1", "node-2"}
			}

			// 1. Setup group and nodes
			group, _, err := groupStore.GetOrCreate(ctx, groupID)
			if err != nil {
				t.Fatalf("failed to create group: %v", err)
			}
			group.Status().SetNodes(nodes)
			if tc.activeJobID != "" {
				group.Spec().SetActiveJob(tc.activeJobID)
			}
			if tc.lockingJobID != "" {
				if err := lockStore.Lock(ctx, groupID, tc.lockingJobID); err != nil {
					t.Fatalf("failed to lock: %v", err)
				}
				group.Spec().RequestLock(tc.lockingJobID)
			}

			// 2. Setup jobs in store
			for _, job := range tc.initialJobs {
				if err := jobStore.Put(ctx, job); err != nil {
					t.Fatalf("failed to put job: %v", err)
				}
			}

			// 3. Mock SnapshotAgentStore
			snapshotCalled := make(chan string, 1)
			restoreCalled := make(chan string, 1)

			callsPerNode := make(map[string]int)
			var mu sync.Mutex

			mockAgentStore := &controller.MockSnapshotAgentStore{
				GetStatusFunc: func(ctx context.Context, node string) (*agentpb.StatusResponse, error) {
					mu.Lock()
					callsPerNode[node]++
					count := callsPerNode[node]
					mu.Unlock()

					if count > 1 && tc.postOpAgentStatuses != nil {
						if statuses, ok := tc.postOpAgentStatuses[node]; ok {
							return &agentpb.StatusResponse{JobStatuses: statuses}, nil
						}
					}

					if tc.initialAgentStatuses != nil {
						if statuses, ok := tc.initialAgentStatuses[node]; ok {
							return &agentpb.StatusResponse{JobStatuses: statuses}, nil
						}
					}
					return &agentpb.StatusResponse{}, nil
				},
				SnapshotFunc: func(ctx context.Context, node, jobID, gID string) (*agentpb.SnapshotResponse, error) {
					if node == tc.expectedSnapshotNode && jobID == tc.expectedSnapshotJob && gID == groupID {
						snapshotCalled <- jobID
					}
					return &agentpb.SnapshotResponse{OperationId: "op-123"}, nil
				},
				RestoreFunc: func(ctx context.Context, node, jobID, gID string) (*agentpb.RestoreResponse, error) {
					if node == tc.expectedRestoreNode && jobID == tc.expectedRestoreJob && gID == groupID {
						restoreCalled <- jobID
					}
					return &agentpb.RestoreResponse{OperationId: "op-123"}, nil
				},
				OperationFunc: func(ctx context.Context, node, operationID string) (*agentpb.GetOperationResponse, error) {
					if operationID == "op-123" {
						return &agentpb.GetOperationResponse{
							Status: agentpb.OperationStatus_OPERATION_STATUS_COMPLETE,
						}, nil
					}
					return &agentpb.GetOperationResponse{}, nil
				},
			}

			mockOrch := &mockInfrastructureOrchestrator{
				observeFunc: func(ctx context.Context, gID string) error {
					return nil
				},
			}

			c := controller.NewController(groupStore, jobStore, testQueue, mockOrch, mockAgentStore)

			// Start the controller
			go func() {
				if err := c.Run(ctx, 1); err != nil {
					t.Errorf("Controller Run failed: %v", err)
				}
			}()

			// Trigger reconcile
			testQueue.Add(groupID)

			if tc.expectFailure {
				err = waitWithTimeout(func() bool {
					return testQueue.getAddRateLimitedCount() > 0
				}, 2*time.Second)
				if err != nil {
					t.Fatal("Timed out waiting for item to be re-queued (reconciliation should have failed)")
				}
				return
			}

			err = waitWithTimeout(func() bool { return testQueue.getDoneCount() > 0 }, 2*time.Second)
			if err != nil {
				t.Fatalf("Timed out waiting for reconcile to complete: %v", err)
			}

			if tc.expectedSnapshotJob != "" {
				select {
				case jobID := <-snapshotCalled:
					if jobID != tc.expectedSnapshotJob {
						t.Errorf("Expected snapshot for %s, got %s", tc.expectedSnapshotJob, jobID)
					}
				case <-time.After(2 * time.Second):
					t.Fatalf("Timed out waiting for expected Snapshot of %s on %s", tc.expectedSnapshotJob, tc.expectedSnapshotNode)
				}
			}

			if tc.expectedRestoreJob != "" {
				select {
				case jobID := <-restoreCalled:
					if jobID != tc.expectedRestoreJob {
						t.Errorf("Expected restore for %s, got %s", tc.expectedRestoreJob, jobID)
					}
				case <-time.After(2 * time.Second):
					t.Fatalf("Timed out waiting for expected Restore of %s on %s", tc.expectedRestoreJob, tc.expectedRestoreNode)
				}
			}

			if group.Status().LoadedJob() != tc.expectedLoadedJob {
				t.Errorf("Expected loadedJob to be %q, got %q", tc.expectedLoadedJob, group.Status().LoadedJob())
			}

			state, _ := group.Status().State()
			if state != tc.expectedGroupState {
				t.Errorf("Expected group state to be %v, got %v", tc.expectedGroupState, state)
			}
		})
	}
}
