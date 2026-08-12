// Copyright 2025 The llm-d Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"context"
	"flag"
	"log/slog"
	"os"
	"strconv"
	"syscall"
	"time"

	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/logging"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/backends"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/server"
)

func main() {
	// Initialize slog with ContextHandler
	jsonHandler := slog.NewJSONHandler(os.Stdout, nil)
	ctxHandler := logging.NewContextHandler(jsonHandler)
	slog.SetDefault(slog.New(ctxHandler))

	port := flag.Int("port", 9001, "The port to listen on")
	deploymentMode := flag.String("deployment-mode", "standalone", "Deployment mode ('standalone' or 'k8s')")
	ncclShim := flag.Bool("nccl-shim", false, "Signal the NCCL C/R shim preloaded into workload processes around cuda-checkpoint operations")
	ncclShimDestroySignal := flag.Int("nccl-shim-destroy-signal", 35, "Signal telling the shim to destroy NCCL communicators before freeze (SIGRTMIN+1 under glibc)")
	ncclShimRecreateSignal := flag.Int("nccl-shim-recreate-signal", 36, "Signal telling the shim to arm lazy NCCL communicator recreation after restore (SIGRTMIN+2 under glibc)")
	ncclShimDestroyWait := flag.Duration("nccl-shim-destroy-wait", 2*time.Second, "How long to wait after the destroy signal before freezing")
	flag.Parse()

	depMode := *deploymentMode
	if envDepMode := os.Getenv("DEPLOYMENT_MODE"); envDepMode != "" {
		depMode = envDepMode
	}

	// AGENT_PORT overrides the flag, mirroring DEPLOYMENT_MODE: the Helm
	// chart configures the agent through env vars, not flags.
	listenPort := *port
	if envPort := os.Getenv("AGENT_PORT"); envPort != "" {
		p, err := strconv.Atoi(envPort)
		if err != nil {
			slog.Error("Invalid AGENT_PORT", "value", envPort, "error", err)
			os.Exit(1)
		}
		listenPort = p
	}

	if depMode != "standalone" && depMode != "k8s" {
		slog.Error("Invalid deployment mode, must be 'standalone' or 'k8s'", "mode", depMode)
		os.Exit(1)
	}

	// NCCL_SHIM_* env vars mirror the flags, like DEPLOYMENT_MODE and
	// AGENT_PORT: the Helm chart configures the agent through env vars.
	shimCfg := backends.NCCLShimConfig{
		Enabled:        *ncclShim,
		DestroySignal:  syscall.Signal(*ncclShimDestroySignal),
		RecreateSignal: syscall.Signal(*ncclShimRecreateSignal),
		DestroyWait:    *ncclShimDestroyWait,
	}
	if env := os.Getenv("NCCL_SHIM_ENABLED"); env != "" {
		v, err := strconv.ParseBool(env)
		if err != nil {
			slog.Error("Invalid NCCL_SHIM_ENABLED", "value", env, "error", err)
			os.Exit(1)
		}
		shimCfg.Enabled = v
	}
	if env := os.Getenv("NCCL_SHIM_DESTROY_SIGNAL"); env != "" {
		v, err := strconv.Atoi(env)
		if err != nil {
			slog.Error("Invalid NCCL_SHIM_DESTROY_SIGNAL", "value", env, "error", err)
			os.Exit(1)
		}
		shimCfg.DestroySignal = syscall.Signal(v)
	}
	if env := os.Getenv("NCCL_SHIM_RECREATE_SIGNAL"); env != "" {
		v, err := strconv.Atoi(env)
		if err != nil {
			slog.Error("Invalid NCCL_SHIM_RECREATE_SIGNAL", "value", env, "error", err)
			os.Exit(1)
		}
		shimCfg.RecreateSignal = syscall.Signal(v)
	}
	if env := os.Getenv("NCCL_SHIM_DESTROY_WAIT"); env != "" {
		d, err := time.ParseDuration(env)
		if err != nil {
			slog.Error("Invalid NCCL_SHIM_DESTROY_WAIT", "value", env, "error", err)
			os.Exit(1)
		}
		shimCfg.DestroyWait = d
	}

	ctx := context.Background()

	// The channel registry is shared between the app-channel backend and the
	// server's WorkloadChannel RPC handler.
	channelRegistry := backends.NewChannelRegistry()
	var cudaOpts []backends.CudaCheckpointOption
	if shimCfg.Enabled {
		cudaOpts = append(cudaOpts, backends.WithNCCLShim(shimCfg))
	}
	registeredBackends := map[backends.BackendType]backends.Backend{
		backends.BackendCuda:        backends.NewCudaCheckpoint(cudaOpts...),
		backends.BackendNoop:        backends.NewNoopBackend(),
		backends.BackendAppEndpoint: backends.NewAppEndpointBackend(),
		backends.BackendAppChannel:  backends.NewAppChannelBackend(channelRegistry),
	}

	slog.InfoContext(ctx, "Starting Snapshot Agent", "port", listenPort, "deploymentMode", depMode, "ncclShim", shimCfg.Enabled)
	if err := server.StartServer(ctx, listenPort, registeredBackends, backends.BackendCuda, depMode, channelRegistry); err != nil {
		slog.ErrorContext(ctx, "Failed to start server", "error", err)
		os.Exit(1)
	}
}
