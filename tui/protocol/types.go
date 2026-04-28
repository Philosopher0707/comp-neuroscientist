// Package protocol defines the JSON event protocol between the Python agent
// (running as a subprocess with --json flag) and the Go TUI.
//
// The Python agent writes one JSON object per line to stdout:
//
//   {"type":"text","content":"analyzing fMRI data..."}
//   {"type":"tool_call","name":"Bash","arguments":{"command":"python ..."}}
//   {"type":"tool_result","name":"Bash","result":"[exit 0] done"}
//   {"type":"status","status":"running","turns":1}
//   {"type":"result","text":"Final report...","turns":3,"duration_ms":12500,"is_error":false}
//   {"type":"error","message":"API connection failed"}
//
// The Go TUI reads these lines, parses them, and updates the UI accordingly.
package protocol

import (
	"encoding/json"
	"fmt"
)

// EventType labels each line of the agent's JSON output stream.
type EventType string

const (
	EventText     EventType = "text"
	EventToolCall EventType = "tool_call"
	EventToolResult EventType = "tool_result"
	EventStatus   EventType = "status"
	EventResult   EventType = "result"
	EventError    EventType = "error"
)

// Event is the top-level envelope for all agent output events.
type Event struct {
	Type EventType `json:"type"`

	// Text / content
	Content string `json:"content,omitempty"`

	// Tool call info
	Name      string                 `json:"name,omitempty"`
	Arguments map[string]interface{} `json:"arguments,omitempty"`

	// Tool result
	Result string `json:"result,omitempty"`

	// Status / result metadata
	Status     string `json:"status,omitempty"`
	Turns      int    `json:"turns,omitempty"`
	DurationMs int64  `json:"duration_ms,omitempty"`
	IsError    bool   `json:"is_error,omitempty"`

	// Error details
	Message string `json:"message,omitempty"`
}

// ParseEvent decodes a single JSON line into an Event.
func ParseEvent(line []byte) (*Event, error) {
	var ev Event
	if err := json.Unmarshal(line, &ev); err != nil {
		return nil, fmt.Errorf("parse event: %w", err)
	}
	return &ev, nil
}

// MustParseEvent parses or panics (for tests).
func MustParseEvent(line []byte) *Event {
	ev, err := ParseEvent(line)
	if err != nil {
		panic(err)
	}
	return ev
}
