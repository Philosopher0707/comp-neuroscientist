package ui

import (
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
	"github.com/philosopher/comp-neuroscientist/tui/protocol"
)

func toModel(m tea.Model, _ tea.Cmd) Model {
	return m.(Model)
}

func TestTypingAfterResponse(t *testing.T) {
	m := *NewModel("python3", "comp_neuroscientist.cli")

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
	m = toModel(m.Update(AgentEventMsg{RunID: m.runID, Event: &protocol.Event{Type: protocol.EventText, Content: "world"}}))

	// Simulate agent done
	m = toModel(m.Update(AgentDoneMsg{RunID: m.runID}))

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

	for _, d := range deltas {
		m = toModel(m.Update(AgentEventMsg{
			RunID: m.runID,
			Event: &protocol.Event{Type: protocol.EventText, Content: d},
		}))
	}
	m = toModel(m.Update(AgentEventMsg{
		RunID: m.runID,
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
	m := *NewModel("python3", "comp_neuroscientist.cli")

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
	m := *NewModel("python3", "comp_neuroscientist.cli")

	m = runAgentStream(t, m, "hi", nil, "only in the result event")

	if !strings.Contains(visibleOutput(m), "only in the result event") {
		t.Fatalf("result.text fallback was dropped, output = %q", visibleOutput(m))
	}
}

// TestReceivedTextResetsBetweenRuns guards the cross-run leak: if the flag
// persisted, the second run's answer would vanish.
func TestReceivedTextResetsBetweenRuns(t *testing.T) {
	m := *NewModel("python3", "comp_neuroscientist.cli")

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
