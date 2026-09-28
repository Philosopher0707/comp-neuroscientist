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
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/philosopher/comp-neuroscientist/tui/protocol"
)

// EventBufferSize is the capacity of each run's event channel. The channel is
// buffered so a burst of agent output never blocks the reader goroutine on a
// slow UI consumer.
const EventBufferSize = 4096

// Stream is the handle to a single execution of the Python agent.
//
// Channel ownership is per-run, NOT per-Runner. Every Start allocates a fresh
// pair of channels and hands them to the goroutine that owns that execution.
// This is load-bearing: the previous design kept the channels on the Runner and
// had streamAndWait read r.events / r.done as struct fields at send and close
// time. A second Start reassigned both fields while the first run's goroutine
// was still alive, so run 1 would send its events into run 2's channel and then
// close it — silently swallowing the current run's output and latching the UI
// into "agent active" forever. With channels owned by the Stream value, that
// class of cross-run contamination is unrepresentable.
//
// The fields are exported so a Stream can be assembled from channels the caller
// already owns. That is what lets the UI be tested without spawning a Python
// process.
type Stream struct {
	// ID is the monotonically increasing run number, used by the model to
	// discard stale messages defensively.
	ID int

	// Events carries decoded events. It is closed after the final event.
	Events <-chan *protocol.Event

	// Done is closed once the process has exited and all events have been
	// delivered. It is never closed early, so a waiter registered for this run
	// can never miss the signal.
	Done <-chan struct{}
}

// Runner manages the Python agent subprocess lifecycle and event streaming.
//
// The Runner is a factory, not a container. It holds configuration and a
// process handle, but never the channels — those belong to each Stream.
type Runner struct {
	mu       sync.Mutex
	running  bool
	nextID   int
	python   string // path to python3 binary
	agentPkg string // Python module path, e.g. "comp_neuroscientist.cli"
	cancel   context.CancelFunc
	workDir  string // absolute path to the repo root; empty means inherit CWD
}

// NewRunner creates a Runner that will launch the Python agent.
// pythonPath: path to python3 binary (e.g., "/usr/local/bin/python3")
// agentModule: Python module to run (e.g., "comp_neuroscientist.cli")
func NewRunner(pythonPath, agentModule string) *Runner {
	return &Runner{
		python:   pythonPath,
		agentPkg: agentModule,
	}
}

// SetWorkDir sets the directory the agent subprocess runs in, and the root
// that src/ is resolved against for PYTHONPATH. When set to a relative path it
// is made absolute so the child behaves identically no matter where the TUI was
// launched from. An empty or non-existent directory is ignored, leaving the
// subprocess to inherit the parent's working directory.
func (r *Runner) SetWorkDir(dir string) {
	if dir == "" {
		return
	}
	abs, err := filepath.Abs(dir)
	if err != nil {
		return
	}
	if info, err := os.Stat(abs); err != nil || !info.IsDir() {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.workDir = abs
}

// Start launches the Python agent with the given prompt and returns a handle to
// that run's event stream. An error means nothing was started and the returned
// stream is nil.
//
// modelName: model to use (e.g., "llama3.1" or "deepseek-v4-flash:cloud")
// localMode: if true, sets CN_LOCAL=true env var for the subprocess
func (r *Runner) Start(prompt, modelName string, localMode bool) (*Stream, error) {
	r.mu.Lock()
	if r.running {
		r.mu.Unlock()
		return nil, fmt.Errorf("agent already running")
	}

	ctx, cancel := context.WithCancel(context.Background())
	events := make(chan *protocol.Event, EventBufferSize)
	done := make(chan struct{})
	stream := &Stream{
		ID:     r.nextID + 1,
		Events: events,
		Done:   done,
	}
	r.nextID = stream.ID
	r.cancel = cancel
	r.running = true
	python, pkg, workDir := r.python, r.agentPkg, r.workDir
	r.mu.Unlock()

	args := []string{"-m", pkg, "--json", prompt}
	cmd := exec.CommandContext(ctx, python, args...)
	cmd.Dir = workDir
	// Inherit the parent environment. An earlier version built the child env
	// from a nil slice, which handed the subprocess a one-element environment
	// with no PATH and no CN_*/OLLAMA_* variables, and broke every model and
	// endpoint setting the user had configured.
	cmd.Env = append(os.Environ(), pythonEnv(workDir)...)

	// Pass model and local mode info
	cmd.Env = append(cmd.Env, "CN_MODEL="+modelName)
	if localMode {
		cmd.Env = append(cmd.Env, "CN_LOCAL=true")
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		cancel()
		r.finishRun()
		return nil, fmt.Errorf("stdout pipe: %w", err)
	}

	stderr, err := cmd.StderrPipe()
	if err != nil {
		cancel()
		r.finishRun()
		return nil, fmt.Errorf("stderr pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		cancel()
		r.finishRun()
		return nil, fmt.Errorf("start agent: %w", err)
	}

	// Stream events + wait for exit in a single goroutine. It closes only the
	// channels belonging to this run, so overlapping runs can never interfere.
	go r.streamAndWait(cmd, events, done, stdout, stderr)

	return stream, nil
}

// pythonEnv returns the PYTHONPATH entries pointing the child at the repo's
// src/ directory. The path is absolute so launching the TUI from a
// subdirectory works. If src/ does not exist we return nothing rather than
// setting a PYTHONPATH that points nowhere.
func pythonEnv(workDir string) []string {
	if workDir == "" {
		return nil
	}
	src := filepath.Join(workDir, "src")
	if info, err := os.Stat(src); err != nil || !info.IsDir() {
		return nil
	}
	return []string{"PYTHONPATH=" + src}
}

// finishRun clears the running flag and cancels the process context. It is
// called on every exit path, including the error paths before the goroutine is
// launched, so a failed Start cannot leave the Runner latched as busy.
func (r *Runner) finishRun() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.running = false
	if r.cancel != nil {
		r.cancel()
		r.cancel = nil
	}
}

// Stop cancels the in-flight run, if any. The run's goroutine still drains and
// closes its own channels, so the UI reliably receives the Done signal.
func (r *Runner) Stop() error {
	r.mu.Lock()
	cancel := r.cancel
	r.cancel = nil
	r.mu.Unlock()

	if cancel != nil {
		cancel()
	}
	return nil
}

// IsRunning reports whether a run is currently in flight.
func (r *Runner) IsRunning() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.running
}

// streamAndWait reads JSON events from stdout, captures stderr, then waits for
// the process to exit. It owns — and closes — exactly the channels passed in.
//
// Every send and close is routed through these local channel variables, never
// through a Runner field, so a later Start cannot retarget this goroutine's
// output.
func (r *Runner) streamAndWait(cmd *exec.Cmd, events chan<- *protocol.Event, done chan<- struct{}, stdout, stderr io.ReadCloser) {
	defer func() {
		if rec := recover(); rec != nil {
			fmt.Fprintf(os.Stderr, "agent: panic in streamAndWait: %v\n", rec)
		}
		// Order matters: the process has exited by the time we get here (or the
		// pipes are torn down), so close the event stream first and the done
		// signal last. A waiter selecting on both can then never observe Done
		// while events are still buffered.
		close(events)
		close(done)
		r.finishRun()
	}()

	// Read stderr concurrently — buffer it so we can include it in the error
	// event. The channel is buffered, so the sender never leaks if the reader
	// below has already given up.
	stderrCh := make(chan string, 1)
	go func() {
		defer func() {
			if rec := recover(); rec != nil {
				fmt.Fprintf(os.Stderr, "agent: panic reading stderr: %v\n", rec)
				stderrCh <- fmt.Sprintf("[panic reading stderr: %v]", rec)
			}
		}()
		var sb strings.Builder
		io.Copy(&sb, stderr)
		stderrCh <- sb.String()
	}()

	// Read JSON events from stdout.
	scanner := bufio.NewScanner(stdout)
	scanner.Buffer(make([]byte, 0, 64*1024), 10*1024*1024) // 10MB max line

	// Unparseable lines are counted, not silently dropped. See
	// protocol.Violation for why swallowing them made bugs invisible.
	var (
		violations  int
		violationNo []string
	)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		ev, err := protocol.ParseEvent([]byte(line))
		if err != nil {
			violations++
			if len(violationNo) < maxReportedViolations {
				violationNo = append(violationNo, truncate(line, 200))
			}
			continue
		}
		// A blocked send here is bounded by the buffer size; if the UI is gone
		// the run has been stopped and ctx is cancelled, so cmd.Wait below
		// will unblock. There is no unbounded wait because the channel is only
		// closed after this loop finishes.
		select {
		case events <- ev:
		case <-time.After(sendTimeout):
			// The consumer is wedged. Stop reading and let the deferred close
			// tear the run down rather than holding a goroutine forever.
			violations++
			return
		}
	}
	if serr := scanner.Err(); serr != nil && violations == 0 {
		violations++
		if len(violationNo) < maxReportedViolations {
			violationNo = append(violationNo, "read error: "+serr.Error())
		}
	}

	// Wait for process exit.
	werr := cmd.Wait()

	// Collect stderr (may already be done).
	stderrText := <-stderrCh

	// Report protocol violations before any exit error so a malformed stream
	// is never masked by a downstream failure.
	if violations > 0 {
		detail := fmt.Sprintf("%d unreadable line(s) from agent stdout", violations)
		if len(violationNo) > 0 {
			detail += "; first: " + strings.Join(violationNo, " | ")
		}
		send(events, &protocol.Event{
			Type:    protocol.EventError,
			Message: "protocol violation: " + detail,
		})
	}

	// If the process errored in a way that wasn't a signal kill, notify.
	if werr != nil && !strings.Contains(werr.Error(), "signal: killed") {
		msg := fmt.Sprintf("agent exited: %v", werr)
		if stderrText != "" {
			// Truncate stderr to avoid flooding
			if len(stderrText) > 2000 {
				stderrText = stderrText[:2000] + "\n... (truncated)"
			}
			msg += "\nstderr:\n" + stderrText
		}
		send(events, &protocol.Event{
			Type:    protocol.EventError,
			Message: msg,
		})
	}
}

// send delivers an event to the run, dropping it if the run is already
// finishing. It never blocks: the channel is buffered and only closed by the
// owning goroutine after all sends have completed.
func send(events chan<- *protocol.Event, ev *protocol.Event) {
	select {
	case events <- ev:
	default:
		// Buffer full during teardown; dropping a diagnostic here is better
		// than blocking a goroutine that must close the channel.
	}
}

// truncate shortens s to at most n bytes, marking the elision.
func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

const (
	// maxReportedViolations bounds how many offending lines are quoted back to
	// the user, so one broken line cannot flood the viewport.
	maxReportedViolations = 3
	// sendTimeout bounds how long a send may block on a full buffer.
	sendTimeout = 2 * time.Second
)
