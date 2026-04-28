// Package agent manages the Python agent subprocess and streams JSON events
// from its stdout to a channel that the Bubble Tea model consumes.
package agent

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sync"

	"github.com/philosopher/comp-neuroscientist/tui/protocol"
)

// Runner manages the Python agent subprocess lifecycle and event streaming.
type Runner struct {
	events    chan *protocol.Event
	done      chan struct{}
	mu        sync.Mutex
	running   bool
	python    string // path to python3 binary
	agentPkg  string // Python module path, e.g. "comp_neuroscientist.cli"
	cwd       string // working directory for subprocess
	cancel    context.CancelFunc // cancels the subprocess context
}

// NewRunner creates a Runner that will launch the Python agent.
// pythonPath: path to python3 binary (e.g., "/usr/local/bin/python3")
// agentModule: Python module to run (e.g., "comp_neuroscientist.cli")
func NewRunner(pythonPath, agentModule string) *Runner {
	return &Runner{
		events:    make(chan *protocol.Event, 256),
		done:      make(chan struct{}),
		python:    pythonPath,
		agentPkg:  agentModule,
	}
}

// Events returns a read-only channel of agent events.
func (r *Runner) Events() <-chan *protocol.Event {
	return r.events
}

// Done returns a channel that closes when the agent process exits.
func (r *Runner) Done() <-chan struct{} {
	return r.done
}

// IsRunning returns whether the agent is currently running.
func (r *Runner) IsRunning() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.running
}

// Start launches the Python agent with the given prompt.
// modelName: Ollama model to use (e.g., "llama3.1" or "deepseek-v4-flash:cloud")
// localMode: if true, sets CN_LOCAL=true env var for the subprocess
func (r *Runner) Start(prompt, modelName string, localMode bool) error {
	r.mu.Lock()
	if r.running {
		r.mu.Unlock()
		return fmt.Errorf("agent already running")
	}
	// Create fresh channels for this run (previous run's channels are closed)
	r.events = make(chan *protocol.Event, 256)
	r.done = make(chan struct{})
	r.running = true
	r.mu.Unlock()

	// Context with cancel — we store cancel to kill the process later
	ctx, cancel := context.WithCancel(context.Background())
	r.cancel = cancel

	// Build command: python3 -m comp_neuroscientist.cli --json "prompt"
	cmd := exec.CommandContext(ctx, r.python, "-m", r.agentPkg, "--json", prompt)

	// Inherit parent env and add our vars (critical: PATH, HOME must be present)
	cmd.Env = append(os.Environ(),
		"ANTHROPIC_AUTH_TOKEN=ollama",
		"ANTHROPIC_BASE_URL=http://localhost:11434",
		"PYTHONPATH=src",
	)

	// Pass model and local mode info
	cmd.Env = append(cmd.Env, "CN_MODEL="+modelName)
	if localMode {
		cmd.Env = append(cmd.Env, "CN_LOCAL=true")
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		cancel()
		return fmt.Errorf("stdout pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		cancel()
		return fmt.Errorf("start agent: %w", err)
	}

	// Stream events in background
	go r.streamEvents(stdout)
	go r.waitExit(cmd)

	return nil
}

// Stop sends an interrupt signal to the agent process via context cancellation.
func (r *Runner) Stop() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if !r.running || r.cancel == nil {
		return nil
	}
	r.cancel()
	return nil
}

func (r *Runner) streamEvents(stdout io.ReadCloser) {
	defer func() {
		stdout.Close()
		r.mu.Lock()
		r.running = false
		r.mu.Unlock()
		close(r.events)
	}()

	scanner := bufio.NewScanner(stdout)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024) // 1MB max line

	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}
		ev, err := protocol.ParseEvent(line)
		if err != nil {
			// Non-JSON lines from the agent (e.g., startup messages) — skip
			continue
		}
		r.events <- ev
	}
}

func (r *Runner) waitExit(cmd *exec.Cmd) {
	err := cmd.Wait()
	if err != nil && err.Error() != "signal: killed" && r.running {
		select {
		case r.events <- &protocol.Event{
			Type:    protocol.EventError,
			Message: fmt.Sprintf("agent exited: %v", err),
		}:
		default:
		}
	}
	close(r.done)
}
