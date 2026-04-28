package protocol

import (
	"testing"
)

func TestParseTextEvent(t *testing.T) {
	line := []byte(`{"type":"text","content":"hello world"}`)
	ev, err := ParseEvent(line)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ev.Type != EventText {
		t.Fatalf("expected EventText, got %s", ev.Type)
	}
	if ev.Content != "hello world" {
		t.Fatalf("expected 'hello world', got '%s'", ev.Content)
	}
}

func TestParseResultEvent(t *testing.T) {
	line := []byte(`{"type":"result","text":"done","turns":3,"duration_ms":12500,"is_error":false}`)
	ev, err := ParseEvent(line)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ev.Type != EventResult {
		t.Fatalf("expected EventResult, got %s", ev.Type)
	}
	if ev.Turns != 3 {
		t.Fatalf("expected 3 turns, got %d", ev.Turns)
	}
	if ev.DurationMs != 12500 {
		t.Fatalf("expected 12500ms, got %d", ev.DurationMs)
	}
	if ev.IsError {
		t.Fatal("expected is_error=false")
	}
}

func TestParseErrorEvent(t *testing.T) {
	line := []byte(`{"type":"error","message":"API connection failed"}`)
	ev, err := ParseEvent(line)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ev.Type != EventError {
		t.Fatalf("expected EventError, got %s", ev.Type)
	}
	if ev.Message != "API connection failed" {
		t.Fatalf("expected 'API connection failed', got '%s'", ev.Message)
	}
}

func TestParseToolCallEvent(t *testing.T) {
	line := []byte(`{"type":"tool_call","name":"Bash","arguments":{"command":"python test.py"}}`)
	ev, err := ParseEvent(line)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ev.Type != EventToolCall {
		t.Fatalf("expected EventToolCall, got %s", ev.Type)
	}
	if ev.Name != "Bash" {
		t.Fatalf("expected 'Bash', got '%s'", ev.Name)
	}
	if ev.Arguments["command"] != "python test.py" {
		t.Fatalf("expected 'python test.py', got '%v'", ev.Arguments["command"])
	}
}

func TestParseStatusEvent(t *testing.T) {
	line := []byte(`{"type":"status","status":"running","turns":1}`)
	ev, err := ParseEvent(line)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ev.Type != EventStatus {
		t.Fatalf("expected EventStatus, got %s", ev.Type)
	}
	if ev.Status != "running" {
		t.Fatalf("expected 'running', got '%s'", ev.Status)
	}
}

func TestParseInvalidJSON(t *testing.T) {
	line := []byte(`not json`)
	_, err := ParseEvent(line)
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
}

func TestMustParseEvent_PanicsOnInvalid(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic")
		}
	}()
	MustParseEvent([]byte(`invalid`))
}

func TestMustParseEvent_Ok(t *testing.T) {
	ev := MustParseEvent([]byte(`{"type":"text","content":"ok"}`))
	if ev.Content != "ok" {
		t.Fatalf("expected 'ok', got '%s'", ev.Content)
	}
}

func TestEventConstants(t *testing.T) {
	if string(EventText) != "text" {
		t.Fatalf("EventText should be 'text'")
	}
	if string(EventToolCall) != "tool_call" {
		t.Fatalf("EventToolCall should be 'tool_call'")
	}
	if string(EventToolResult) != "tool_result" {
		t.Fatalf("EventToolResult should be 'tool_result'")
	}
	if string(EventStatus) != "status" {
		t.Fatalf("EventStatus should be 'status'")
	}
	if string(EventResult) != "result" {
		t.Fatalf("EventResult should be 'result'")
	}
	if string(EventError) != "error" {
		t.Fatalf("EventError should be 'error'")
	}
}
