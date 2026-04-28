package ui

import (
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
