package ui

import (
	"errors"
	"strings"
	"sync"
	"testing"

	tea "charm.land/bubbletea/v2"
	"github.com/philosopher/comp-neuroscientist/tui/agent"
	"github.com/philosopher/comp-neuroscientist/tui/protocol"
)

func toModel(m tea.Model, _ tea.Cmd) Model {
	return m.(Model)
}

// newTestModel returns a Model whose Enter keypress does NOT spawn a real
// Python subprocess. The fake hands back a stream over channels that stay open
// for the life of the test; the tests drive delivery by sending AgentEventMsg /
// AgentDoneMsg directly, exactly as the real listener would.
func newTestModel(t *testing.T) Model {
	t.Helper()
	m := *NewModel("python3", "comp_neuroscientist.cli")

	var mu sync.Mutex
	next := 0
	m.startRun = func(prompt, modelName string, localMode bool) (*agent.Stream, error) {
		mu.Lock()
		next++
		id := next
		mu.Unlock()
		events := make(chan *protocol.Event, 64)
		done := make(chan struct{})
		return &agent.Stream{ID: id, Events: events, Done: done}, nil
	}
	return m
}

// runIDOf is the ID of the run the model is currently bound to. Tests assert
// against this rather than a counter on the model, because the run handle is
// the single authority for "which execution is this".
func runIDOf(t *testing.T, m Model) int {
	t.Helper()
	if m.currentRun == nil {
		t.Fatal("expected a currentRun handle, got nil")
	}
	return m.currentRun.ID
}

func TestTypingAfterResponse(t *testing.T) {
	m := newTestModel(t)

	// Simulate startup window size
	m = toModel(m.Update(tea.WindowSizeMsg{Width: 100, Height: 30}))

	// Simulate typing "hello" and pressing Enter
	for _, r := range "hello" {
		m = toModel(m.Update(tea.KeyPressMsg{Code: r, Text: string(r)}))
	}
	if m.input.Value() != "hello" {
		t.Fatalf("expected input value 'hello', got %q", m.input.Value())
	}

	// Submit
	m = toModel(m.Update(tea.KeyPressMsg{Code: tea.KeyEnter, Text: "enter"}))
	if !m.agentActive {
		t.Fatal("expected agent to be active after Enter")
	}

	// Simulate agent text event
	m = toModel(m.Update(AgentEventMsg{RunID: runIDOf(t, m), Event: &protocol.Event{Type: protocol.EventText, Content: "world"}}))

	// Simulate agent done
	m = toModel(m.Update(AgentDoneMsg{RunID: runIDOf(t, m)}))

	if m.agentActive {
		t.Fatal("expected agent to be inactive after done")
	}

	// Try typing again
	for _, r := range "test" {
		m = toModel(m.Update(tea.KeyPressMsg{Code: r, Text: string(r)}))
	}
	if m.input.Value() != "test" {
		t.Fatalf("expected input value 'test' after second response, got %q", m.input.Value())
	}
}

// runAgentStream drives the model through one complete agent run: the prompt
// submit, a stream of text deltas, and the terminal result event.
func runAgentStream(t *testing.T, m Model, prompt string, deltas []string, resultText string) Model {
	t.Helper()

	m = toModel(m.Update(tea.WindowSizeMsg{Width: 100, Height: 30}))
	for _, r := range prompt {
		m = toModel(m.Update(tea.KeyPressMsg{Code: r, Text: string(r)}))
	}
	m = toModel(m.Update(tea.KeyPressMsg{Code: tea.KeyEnter, Text: "enter"}))
	runID := runIDOf(t, m)

	for _, d := range deltas {
		m = toModel(m.Update(AgentEventMsg{
			RunID: runID,
			Event: &protocol.Event{Type: protocol.EventText, Content: d},
		}))
	}
	m = toModel(m.Update(AgentEventMsg{
		RunID: runID,
		Event: &protocol.Event{
			Type:       protocol.EventResult,
			Text:       resultText,
			Turns:      1,
			IsError:    false,
			DurationMs: 10,
		},
	}))
	return m
}

// visibleOutput is everything the user can see for a run. The result event
// finalizes the turn, which moves streamingOutput into outputHistory, so
// assertions must look at both.
func visibleOutput(m Model) string {
	return strings.Join(m.outputHistory, "\n") + m.streamingOutput
}

// TestFinalAnswerIsNotDuplicated is the regression guard for the wiring bug
// where result.text was appended to output that had already streamed in as
// text events — the user saw the agent's answer twice.
func TestFinalAnswerIsNotDuplicated(t *testing.T) {
	m := newTestModel(t)

	// A real run: the answer streams in three deltas, then repeats in full on
	// the result event.
	m = runAgentStream(t, m, "hi", []string{"alpha ", "bravo ", "charlie"}, "alpha bravo charlie")

	out := visibleOutput(m)
	if got := strings.Count(out, "charlie"); got != 1 {
		t.Fatalf("final answer duplicated: 'charlie' appears %d times in %q", got, out)
	}
	if !strings.Contains(out, "alpha bravo charlie") {
		t.Fatalf("expected the streamed answer to survive, got %q", out)
	}
}

// TestResultTextUsedWhenNoTextStreamed covers the fallback: an agent that emits
// only a result event (no text deltas) must still have its answer displayed.
func TestResultTextUsedWhenNoTextStreamed(t *testing.T) {
	m := newTestModel(t)

	m = runAgentStream(t, m, "hi", nil, "only in the result event")

	if !strings.Contains(visibleOutput(m), "only in the result event") {
		t.Fatalf("result.text fallback was dropped, output = %q", visibleOutput(m))
	}
}

// TestReceivedTextResetsBetweenRuns guards the cross-run leak: if the flag
// persisted, the second run's answer would vanish.
func TestReceivedTextResetsBetweenRuns(t *testing.T) {
	m := newTestModel(t)

	// Run 1 streams text.
	m = runAgentStream(t, m, "first", []string{"one"}, "one")
	if !m.receivedText {
		t.Fatal("expected receivedText set after streaming run")
	}
	historyBefore := len(m.outputHistory)

	// Run 2 streams no text, only a result event.
	m = runAgentStream(t, m, "second", nil, "second answer")

	if len(m.outputHistory) <= historyBefore {
		t.Fatalf("run 2 produced no new output block; history = %d", len(m.outputHistory))
	}
	last := m.outputHistory[len(m.outputHistory)-1]
	if strings.Contains(last, "one") {
		t.Fatalf("run 1 output leaked into run 2: %q", last)
	}
	if !strings.Contains(last, "second answer") {
		t.Fatalf("run 2 answer missing (stale receivedText suppressed it): %q", last)
	}
}

// ── F1 regression guards ──────────────────────────────────────

// TestStaleRunMessagesAreRejected is the guard for the cross-run contamination
// bug. A message from a superseded run must not touch the live run's state, and
// the guard must be symmetric across AgentEventMsg and AgentDoneMsg — the
// original code compared Done with `!=` on a bare counter while Event used a
// different rule, so a Done carrying a *future* id was accepted.
func TestStaleRunMessagesAreRejected(t *testing.T) {
	m := newTestModel(t)
	m = toModel(m.Update(tea.WindowSizeMsg{Width: 100, Height: 30}))
	for _, r := range "hello" {
		m = toModel(m.Update(tea.KeyPressMsg{Code: r, Text: string(r)}))
	}
	m = toModel(m.Update(tea.KeyPressMsg{Code: tea.KeyEnter, Text: "enter"}))
	live := runIDOf(t, m)

	// A stale event (older run) must be ignored entirely.
	before := m.streamingOutput
	m = toModel(m.Update(AgentEventMsg{
		RunID: live - 1,
		Event: &protocol.Event{Type: protocol.EventText, Content: "STALE"},
	}))
	if strings.Contains(visibleOutput(m), "STALE") {
		t.Fatal("stale run event leaked into visible output")
	}
	if m.streamingOutput != before {
		t.Fatalf("stale event mutated streamingOutput: %q != %q", m.streamingOutput, before)
	}

	// A stale Done must not tear down the live run.
	m = toModel(m.Update(AgentDoneMsg{RunID: live - 1}))
	if !m.agentActive {
		t.Fatal("stale AgentDoneMsg stopped the live run")
	}

	// A FUTURE id must be rejected too. The old Done guard was
	// `msg.RunID != 0 && msg.RunID < m.runID`, which accepts a future id and
	// would let a run that has not started yet end the current one.
	m = toModel(m.Update(AgentDoneMsg{RunID: live + 1}))
	if !m.agentActive {
		t.Fatal("future-id AgentDoneMsg stopped the live run (asymmetric guard)")
	}

	// The live Done still works.
	m = toModel(m.Update(AgentDoneMsg{RunID: live}))
	if m.agentActive {
		t.Fatal("live AgentDoneMsg did not stop the run")
	}
}

// TestDoneAfterStopIsInert covers the stop path: pressing Stop drops the handle,
// so a Done that was already in flight cannot resurrect or mutate state.
func TestDoneAfterStopIsInert(t *testing.T) {
	m := newTestModel(t)
	m = toModel(m.Update(tea.WindowSizeMsg{Width: 100, Height: 30}))
	for _, r := range "hello" {
		m = toModel(m.Update(tea.KeyPressMsg{Code: r, Text: string(r)}))
	}
	m = toModel(m.Update(tea.KeyPressMsg{Code: tea.KeyEnter, Text: "enter"}))
	live := runIDOf(t, m)

	// Stop key (^K) drops the handle.
	m = toModel(m.Update(tea.KeyPressMsg{Code: 'k', Mod: tea.ModCtrl}))
	if m.currentRun != nil {
		t.Fatal("Stop did not clear currentRun")
	}
	if m.agentActive {
		t.Fatal("Stop left agentActive true")
	}

	// A late event from the stopped run must be dropped, not displayed.
	m = toModel(m.Update(AgentEventMsg{
		RunID: live,
		Event: &protocol.Event{Type: protocol.EventText, Content: "ZOMBIE"},
	}))
	if strings.Contains(visibleOutput(m), "ZOMBIE") {
		t.Fatal("event from stopped run leaked into output after Stop")
	}
}

// TestFailedStartLeavesInputUsable guards the user-visible contract: if the
// agent cannot be spawned, the input must stay usable and the error must be
// shown. The mechanism is the error branch in the Enter handler, which resets
// agentActive and clears currentRun. Setting agentActive only after Start
// succeeds is defence-in-depth on top of that, not the primary guard — this
// test passes with either ordering, so it pins the behaviour, not the
// implementation.
func TestFailedStartLeavesInputUsable(t *testing.T) {
	m := newTestModel(t)
	m = toModel(m.Update(tea.WindowSizeMsg{Width: 100, Height: 30}))
	m.startRun = func(prompt, modelName string, localMode bool) (*agent.Stream, error) {
		return nil, errors.New("spawn failed")
	}
	for _, r := range "hello" {
		m = toModel(m.Update(tea.KeyPressMsg{Code: r, Text: string(r)}))
	}
	m = toModel(m.Update(tea.KeyPressMsg{Code: tea.KeyEnter, Text: "enter"}))

	if m.agentActive {
		t.Fatal("agentActive true after a failed start — input is wedged")
	}
	if m.agentStatus != "error" {
		t.Fatalf("expected status error, got %q", m.agentStatus)
	}
	// Typing must still work.
	for _, r := range "again" {
		m = toModel(m.Update(tea.KeyPressMsg{Code: r, Text: string(r)}))
	}
	if m.input.Value() != "again" {
		t.Fatalf("input unusable after failed start: %q", m.input.Value())
	}
}
