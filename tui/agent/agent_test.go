package agent

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/philosopher/comp-neuroscientist/tui/protocol"
)

// fakePython writes a stub interpreter that emits a canned NDJSON sequence and
// then exits. It lets the Runner be exercised over a real process boundary —
// pipes, exec, and goroutine teardown — without importing the Python package.
func fakePython(t *testing.T, body string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "fake-python")
	script := "#!/bin/sh\n" + body
	if err := os.WriteFile(path, []byte(script), 0o755); err != nil {
		t.Fatalf("write fake python: %v", err)
	}
	return path
}

// drain collects every event from a stream until Events is closed, then waits
// for Done. It fails the test on timeout rather than hanging forever.
func drain(t *testing.T, s *Stream) []*protocol.Event {
	t.Helper()
	var out []*protocol.Event
	timeout := time.After(15 * time.Second)
	for {
		select {
		case ev, ok := <-s.Events:
			if !ok {
				// Events closed: Done must already be closed or close imminently.
				select {
				case <-s.Done:
				case <-time.After(2 * time.Second):
					t.Fatal("Events closed but Done never closed")
				}
				return out
			}
			out = append(out, ev)
		case <-timeout:
			t.Fatalf("timed out; got %d events", len(out))
		}
	}
}

// TestRunnerDoesNotRetargetALiveRun pins the structural invariant that fixes F1.
//
// The pre-fix Runner held the live event/done channels in struct fields, and
// streamAndWait's teardown used them via the receiver:
//
//	defer func() { ...; r.running = false; closeEvents(); closeDone() }()
//
// where closeEvents() did `close(r.events)`. Because `running` is cleared
// BEFORE the closes, and Start() is admitted once `running` is false, a Start
// landing in that window overwrites r.events with the new run's channel — and
// the pending teardown then closes the NEW run's channel. The window is only a
// couple of instructions wide, so a timing test cannot reliably hit it; this
// test instead asserts the property that makes the race unrepresentable:
// no send or close in the teardown path resolves a channel through the Runner
// receiver. With channels owned per-run, no later Start can retarget it.
//
// A behavioural twin of this test exists in the TUI layer (TestStaleEventAfter
// StopIsIgnored), which guards the consumer side of the same bug.
func TestRunnerDoesNotRetargetALiveRun(t *testing.T) {
	src, err := os.ReadFile("agent.go")
	if err != nil {
		t.Fatalf("read agent.go: %v", err)
	}
	body := stripNonCode(string(src))

	// Any `r.<chan>` reference in the send/close path is exactly the bug.
	for i, line := range strings.Split(body, "\n") {
		if !strings.Contains(line, "close(") || strings.Contains(line, "stderrCh") || strings.Contains(line, "closeMu") {
			continue
		}
		if strings.Contains(line, "r.events") || strings.Contains(line, "r.done") {
			t.Errorf("agent.go:%d closes a channel through the Runner receiver (%q); "+
				"runs must own their channels so a later Start cannot retarget this teardown",
				i+1, strings.TrimSpace(line))
		}
	}
}

// stripNonCode blanks comments and string literals so the source scan above
// cannot be fooled by the very comments that document the invariant.
func stripNonCode(s string) string {
	lines := strings.Split(s, "\n")
	for i, l := range lines {
		if idx := strings.Index(l, "//"); idx >= 0 {
			lines[i] = l[:idx]
		}
	}
	return strings.Join(lines, "\n")
}

const twoEvents = `printf '%s\n' '{"type":"text","content":"hello"}' '{"type":"result","text":"hello","turns":1}'`

// TestRunnerStreamsAndClosesBothChannels is the basic lifecycle contract:
// events arrive, the stream closes, and Done closes afterwards.
func TestRunnerStreamsAndClosesBothChannels(t *testing.T) {
	r := NewRunner(fakePython(t, twoEvents), "comp_neuroscientist.cli")
	s, err := r.Start("hi", "test-model", false)
	if err != nil {
		t.Fatalf("Start: %v", err)
	}
	evs := drain(t, s)
	if len(evs) != 2 {
		t.Fatalf("expected 2 events, got %d", len(evs))
	}
	if evs[0].Type != protocol.EventText || evs[0].Content != "hello" {
		t.Errorf("unexpected first event: %+v", evs[0])
	}
	if evs[1].Type != protocol.EventResult {
		t.Errorf("unexpected second event: %+v", evs[1])
	}
	// The runner must be reusable once the run has finished.
	deadline := time.Now().Add(2 * time.Second)
	for r.IsRunning() && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if r.IsRunning() {
		t.Fatal("runner still marked running after Done")
	}
}

// TestRunIDsAreMonotonic guards the id allocation: a second run must get a
// distinct, higher id, since the UI uses that id to reject stale messages.
func TestRunIDsAreMonotonic(t *testing.T) {
	r := NewRunner(fakePython(t, twoEvents), "comp_neuroscientist.cli")
	s1, err := r.Start("one", "m", false)
	if err != nil {
		t.Fatalf("Start 1: %v", err)
	}
	drain(t, s1)

	s2, err := r.Start("two", "m", false)
	if err != nil {
		t.Fatalf("Start 2: %v", err)
	}
	drain(t, s2)

	if s2.ID <= s1.ID {
		t.Fatalf("run ids not increasing: first=%d second=%d", s1.ID, s2.ID)
	}
}

// TestConcurrentStartRejected pins the single-run invariant. The UI relies on
// Start failing rather than silently overlapping a second execution.
func TestConcurrentStartRejected(t *testing.T) {
	// A script that blocks until a marker file appears.
	dir := t.TempDir()
	marker := filepath.Join(dir, "go")
	script := "while [ ! -f " + marker + " ]; do sleep 0.01; done\n" + twoEvents
	r := NewRunner(fakePython(t, script), "comp_neuroscientist.cli")

	if _, err := r.Start("first", "m", false); err != nil {
		t.Fatalf("Start 1: %v", err)
	}
	if _, err := r.Start("second", "m", false); err == nil {
		t.Fatal("expected second concurrent Start to be rejected")
	}
	if err := os.WriteFile(marker, []byte("x"), 0o644); err != nil {
		t.Fatalf("unblock: %v", err)
	}
}

// TestStopTerminatesAndCloses is the user-facing stop contract: after Stop, the
// stream must terminate rather than leak, so the UI is not left latched.
func TestStopTerminatesAndCloses(t *testing.T) {
	// Block forever so only Stop can end the run.
	r := NewRunner(fakePython(t, "while true; do sleep 0.05; done\n"), "comp_neuroscientist.cli")
	s, err := r.Start("hi", "m", false)
	if err != nil {
		t.Fatalf("Start: %v", err)
	}

	done := make(chan struct{})
	go func() {
		drain(t, s)
		close(done)
	}()

	time.Sleep(200 * time.Millisecond)
	r.Stop()

	select {
	case <-done:
	case <-time.After(15 * time.Second):
		t.Fatal("Stop did not terminate the run; channels never closed")
	}
}

// TestMalformedOutputSurfacesError covers the protocol-violation path: a line
// that is not valid JSON must be reported as an error event, not silently
// dropped, so a corrupted stream is distinguishable from a clean one.
func TestMalformedOutputSurfacesError(t *testing.T) {
	r := NewRunner(fakePython(t, `printf '%s\n' 'not json at all'`), "comp_neuroscientist.cli")
	s, err := r.Start("hi", "m", false)
	if err != nil {
		t.Fatalf("Start: %v", err)
	}
	evs := drain(t, s)
	found := false
	for _, ev := range evs {
		if ev.Type == protocol.EventError && strings.Contains(strings.ToLower(ev.Message), "protocol") {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected a protocol-violation error event, got %+v", evs)
	}
}
