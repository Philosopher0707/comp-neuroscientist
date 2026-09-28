package protocol

import (
	"bufio"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// goldenPath resolves the shared NDJSON fixture. It lives outside this
// package (under tests/testdata) so it is a single artifact owned by neither
// language, decoded by both.
func goldenPath(t *testing.T) string {
	t.Helper()

	// Walk up from the package dir until tests/testdata/ appears.
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	for i := 0; i < 6; i++ {
		p := filepath.Join(dir, "tests", "testdata", "protocol_golden.ndjson")
		if _, err := os.Stat(p); err == nil {
			return p
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	t.Skip("shared golden fixture not found; run: python tests/generate_golden.py")
	return ""
}

// TestGoldenFixtureDecodes runs the real, machine-generated agent output
// through the real decoder.
//
// Every other test in this file feeds ParseEvent a hand-written string. That
// means nothing in the Go suite ever saw what the Python side actually prints.
// This test is the seam: the Python E2E suite (tests/test_protocol_e2e.py)
// asserts the agent still emits every event type recorded in this fixture, so
// if either side changes shape, one of the two suites fails.
func TestGoldenFixtureDecodes(t *testing.T) {
	path := goldenPath(t)

	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open golden: %v", err)
	}
	defer f.Close()

	seen := map[EventType]bool{}
	var streamedText, finalText string
	lineNo := 0

	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024) // tool_result lines can be long
	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		ev, err := ParseEvent([]byte(line))
		if err != nil {
			t.Fatalf("golden line %d failed to decode: %v\n%s", lineNo, err, line)
		}
		seen[ev.Type] = true

		switch ev.Type {
		case EventText:
			streamedText += ev.Content
		case EventResult:
			finalText = ev.Text
		case EventToolCall:
			if ev.Name == "" {
				t.Errorf("golden line %d: tool_call missing name", lineNo)
			}
			if ev.Arguments == nil {
				t.Errorf("golden line %d: tool_call missing arguments", lineNo)
			}
		case EventToolResult:
			if ev.Name == "" {
				t.Errorf("golden line %d: tool_result missing name", lineNo)
			}
		case EventError:
			if ev.Message == "" {
				t.Errorf("golden line %d: error missing message", lineNo)
			}
		case EventStatus:
			if ev.Status == "" {
				t.Errorf("golden line %d: status missing status", lineNo)
			}
			if ev.Turns < 1 {
				t.Errorf("golden line %d: status turns should be >= 1, got %d", lineNo, ev.Turns)
			}
		}
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("scan golden: %v", err)
	}

	// The decoder must be able to see every event type in the contract.
	for _, want := range []EventType{
		EventText, EventStatus, EventToolCall, EventToolResult, EventResult, EventError,
	} {
		if !seen[want] {
			t.Errorf("golden fixture never exercised event type %q", want)
		}
	}

	if streamedText == "" {
		t.Fatal("golden fixture produced no streamed text")
	}
	if finalText == "" {
		t.Fatal("golden result event carried no final answer — Text field regressed")
	}
	// The fixture is a real run: the final answer must match what streamed.
	if finalText != streamedText {
		t.Errorf("result.text %q does not match concatenated text events %q", finalText, streamedText)
	}
}

// TestGoldenFixtureIsMachineIndependent guards against the fixture embedding
// absolute paths, which would make it churn on every checkout and diff.
func TestGoldenFixtureIsMachineIndependent(t *testing.T) {
	path := goldenPath(t)
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	// Anything under /Users/<name>/ or /home/<name>/ is machine-specific.
	for _, marker := range []string{"/Users/", "/home/", "C:\\Users\\"} {
		if strings.Contains(string(b), marker) {
			t.Errorf("golden fixture embeds machine-specific path %q; "+
				"regenerate with normalization (python tests/generate_golden.py)", marker)
		}
	}
}
