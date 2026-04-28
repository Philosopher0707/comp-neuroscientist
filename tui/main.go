// Comp-Neuroscientist TUI — Go-based terminal UI using Charm's Bubble Tea
// and Lip Gloss. Spawns the Python agent as a subprocess with --json mode.
package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"

	tea "charm.land/bubbletea/v2"

	"github.com/philosopher/comp-neuroscientist/tui/ui"
)

func main() {
	// Find python3 binary
	pythonPath, err := findPython()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		fmt.Fprintf(os.Stderr, "Make sure Python 3.13+ is installed.\n")
		os.Exit(1)
	}

	// Determine agent module path
	agentModule := "comp_neuroscientist.cli"

	// Verify the Python package is importable
	if err := checkAgent(pythonPath, agentModule); err != nil {
		fmt.Fprintf(os.Stderr, "Warning: %v\n", err)
		fmt.Fprintf(os.Stderr, "The TUI will start but the agent may not work.\n")
	}

	// Create the Bubble Tea model
	model := ui.NewModel(pythonPath, agentModule)

	// Start the Bubble Tea program
	// AltScreen and mouse mode are declared in Model.View()
	p := tea.NewProgram(model)

	if _, err := p.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "Error running TUI: %v\n", err)
		os.Exit(1)
	}
}

// findPython locates a suitable python3 binary.
func findPython() (string, error) {
	// Check common paths
	candidates := []string{
		"/opt/homebrew/bin/python3",
		"/usr/local/bin/python3",
		"/usr/bin/python3",
		"python3.13",
		"python3.12",
		"python3",
	}

	// Also check PATH
	path, err := exec.LookPath("python3")
	if err == nil {
		return path, nil
	}

	for _, c := range candidates {
		// Check if absolute path exists
		if filepath.IsAbs(c) {
			if _, err := os.Stat(c); err == nil {
				// Verify it's actually Python 3.13+
				cmd := exec.Command(c, "--version")
				out, err := cmd.Output()
				if err == nil && len(out) > 0 {
					return c, nil
				}
			}
		} else {
			// PATH lookup
			p, err := exec.LookPath(c)
			if err == nil {
				return p, nil
			}
		}
	}

	return "", fmt.Errorf("python3 not found")
}

// checkAgent verifies the Python agent module is importable.
func checkAgent(pythonPath, agentModule string) error {
	cmd := exec.Command(pythonPath, "-c",
		fmt.Sprintf("import %s; print('ok')", agentModule),
	)
	cmd.Env = append(cmd.Env, "PYTHONPATH=src")
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("agent module '%s' not importable: %s", agentModule, string(out))
	}
	return nil
}
