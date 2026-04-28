// Package ui contains the Bubble Tea model, view rendering, and event handling
// for the Comp-Neuroscientist TUI.
package ui

import (
	"fmt"
	"math"
	"os/user"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/help"
	"github.com/charmbracelet/bubbles/key"
	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/philosopher/comp-neuroscientist/tui/agent"
	"github.com/philosopher/comp-neuroscientist/tui/protocol"
)

// ── Model ─────────────────────────────────────────────────────

// Model is the top-level Bubble Tea model for the Comp-Neuroscientist TUI.
type Model struct {
	// Agent
	agentRunner *agent.Runner
	agentActive bool
	agentStatus string // "ready", "running", "done", "error"

	// Output
	streamingOutput string    // accumulated text from current run (Pi-style)
	outputHistory   []string  // completed output blocks
	viewport        viewport.Model

	// Input
	input       textinput.Model
	inputFocused bool

	// Sidebar
	sidebarWidth int
	filesList    []string
	modelName    string

	// Timing / metrics
	startTime    time.Time
	turnCount    int
	durationMs   int64

	// Window size
	width  int
	height int

	// Help
	help     help.Model
	showHelp bool

	// Focus mode: "main", "input", "sidebar"
	focus string

	// Path sanitization — home directory to replace with ~
	homeDir string

	// Quit flag
	quitting bool
}

// NewModel creates the initial Bubble Tea model.
func NewModel(pythonPath, agentModule string) *Model {
	ti := textinput.New()
	ti.Placeholder = "Describe your neuroscience analysis..."
	ti.Focus()
	ti.CharLimit = 1000
	ti.Width = 60
	ti.Prompt = "┃ "

	// Start the agent runner (not running yet)
	r := agent.NewRunner(pythonPath, agentModule)

	// Get home directory for path sanitization
	homeDir := "/home/user"
	if u, err := user.Current(); err == nil {
		homeDir = u.HomeDir
	}

	return &Model{
		agentRunner:   r,
		agentStatus:   "ready",
		agentActive:   false,
		input:         ti,
		inputFocused:  true,
		sidebarWidth:  30,
		modelName:     "deepseek-v4-flash:cloud",
		help:          help.New(),
		focus:         "input",
		outputHistory: []string{},
		homeDir:       homeDir,
	}
}

// ── Messages ─────────────────────────────────────────────────

// AgentEventMsg is sent when a new protocol event arrives from the agent.
type AgentEventMsg struct {
	Event *protocol.Event
}

// AgentDoneMsg is sent when the agent process exits.
// SubmitMsg is sent when the user presses Enter to submit a prompt.
// TickMsg is sent periodically for real-time status updates.
type AgentDoneMsg struct{}
type SubmitMsg struct {
	Prompt string
}
type TickMsg time.Time

// ── Init ─────────────────────────────────────────────────────

func (m Model) Init() tea.Cmd {
	return tea.Batch(
		textinput.Blink,
		// waitForEvents is NOT registered here — it blocks forever on empty channels.
		// The Enter handler registers it when the agent starts.
		m.tick(),
	)
}

// waitForEvents returns a command that listens for agent events
// and sends them as tea.Msg on the main loop.
func (m Model) waitForEvents() tea.Cmd {
	return func() tea.Msg {
		select {
		case ev, ok := <-m.agentRunner.Events():
			if !ok {
				return AgentDoneMsg{}
			}
			return AgentEventMsg{Event: ev}
		case <-m.agentRunner.Done():
			return AgentDoneMsg{}
		}
	}
}

// tick returns a command for periodic ticks (status updates).
func (m Model) tick() tea.Cmd {
	return tea.Tick(time.Second, func(t time.Time) tea.Msg {
		return TickMsg(t)
	})
}

// ── Update ───────────────────────────────────────────────────

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd

	switch msg := msg.(type) {
	// ── Window resize ────────────────────────────────
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.help.Width = msg.Width
		m.viewport.Width = m.contentWidth()
		m.viewport.Height = m.contentHeight()
		m.input.Width = m.contentWidth() - 6

	// ── Key events ──────────────────────────────────

	case tea.KeyMsg:
		// If Enter is pressed while focused on input with text, submit the prompt
		if msg.Type == tea.KeyEnter && m.focus == "input" && m.input.Value() != "" && !m.agentActive {
			prompt := m.input.Value()
			m.input.SetValue("")
			m.agentStatus = "running"
			m.agentActive = true
			m.startTime = time.Now()
			m.turnCount = 0

			// Start new output with user's prompt visible
			m.streamingOutput = fmt.Sprintf("> %s\n\n", prompt)
			m.viewport.SetContent(RenderMarkdown(m.streamingOutput))

			localMode := !strings.Contains(m.modelName, ":cloud")
			if err := m.agentRunner.Start(prompt, m.modelName, localMode); err != nil {
				m.agentStatus = "error"
				m.streamingOutput += fmt.Sprintf("\n❌ Error: %v\n", err)
				return m, nil
			}
			// Register event listener for this run
			cmds = append(cmds, m.waitForEvents())
			return m, tea.Batch(cmds...)
		}

		// Global keys
		switch {
		case key.Matches(msg, keys.Help):
			m.showHelp = !m.showHelp
			return m, nil

		case key.Matches(msg, keys.Quit):
			if m.agentActive {
				m.agentRunner.Stop()
				m.agentActive = false
				m.agentStatus = "ready"
			}
			m.quitting = true
			return m, tea.Quit

		case key.Matches(msg, keys.Stop):
			if m.agentActive {
				m.agentRunner.Stop()
				m.agentActive = false
				m.agentStatus = "ready"
			}
			return m, nil

		case key.Matches(msg, keys.FocusInput):
			m.focus = "input"
			m.input.Focus()
			return m, nil

		case key.Matches(msg, keys.FocusOutput):
			m.focus = "main"
			m.input.Blur()
			return m, nil

		case key.Matches(msg, keys.Clear):
			m.streamingOutput = ""
			m.outputHistory = nil
			m.viewport.SetContent("")
			m.viewport.GotoTop()
			return m, nil
		}

		// If Enter was not handled above and textinput needs it, pass through
		// (textinput does not consume Enter, so this is a no-op fallback)

	// ── Agent events ────────────────────────────────

	case AgentEventMsg:
		ev := msg.Event
		switch ev.Type {
		case protocol.EventText:
			m.streamingOutput += ev.Content
			fullConv := m.buildConversation()
			displayText := sanitizePaths(RenderMarkdown(fullConv), m.homeDir)
			m.viewport.SetContent(displayText)
			m.viewport.GotoBottom()
			cmds = append(cmds, m.waitForEvents())

		case protocol.EventToolCall:
			callLine := fmt.Sprintf("  🔧 %s(%s)", ev.Name, formatArgs(ev.Arguments))
			m.streamingOutput += "\n" + callLine + "\n"
			fullConv := m.buildConversation()
			displayText := sanitizePaths(RenderMarkdown(fullConv), m.homeDir)
			m.viewport.SetContent(displayText)
			m.viewport.GotoBottom()
			cmds = append(cmds, m.waitForEvents())

		case protocol.EventToolResult:
			if len(ev.Result) > 80 {
				m.streamingOutput += fmt.Sprintf("  └─ [%d chars]\n", len(ev.Result))
			} else {
				m.streamingOutput += fmt.Sprintf("  └─ %s\n", ev.Result)
			}
			fullConv := m.buildConversation()
			displayText := sanitizePaths(RenderMarkdown(fullConv), m.homeDir)
			m.viewport.SetContent(displayText)
			cmds = append(cmds, m.waitForEvents())

		case protocol.EventStatus:
			m.agentStatus = ev.Status
			if ev.Turns > 0 {
				m.turnCount = ev.Turns
			}
			cmds = append(cmds, m.waitForEvents())

		case protocol.EventResult:
			m.agentActive = false
			m.agentStatus = "done"
			m.turnCount = ev.Turns
			m.durationMs = ev.DurationMs
			if ev.IsError {
				m.agentStatus = "error"
			}
			if m.streamingOutput != "" {
				m.outputHistory = append(m.outputHistory, m.streamingOutput)
				m.streamingOutput = ""
			}
			// Re-render conversation history
			fullConv := m.buildConversation()
			displayText := sanitizePaths(RenderMarkdown(fullConv), m.homeDir)
			m.viewport.SetContent(displayText)
			m.viewport.GotoBottom()
			cmds = append(cmds, m.waitForEvents())

		case protocol.EventError:
			m.agentActive = false
			m.agentStatus = "error"
			errorLine := fmt.Sprintf("\n❌ Error: %s\n", ev.Message)
			m.streamingOutput += errorLine
			fullConv := m.buildConversation()
			displayText := sanitizePaths(RenderMarkdown(fullConv), m.homeDir)
			m.viewport.SetContent(displayText)
			m.viewport.GotoBottom()
			cmds = append(cmds, m.waitForEvents())
		}

		// Re-register the event listener for next event
		// (also falls through to input/viewport updates below)

	case AgentDoneMsg:
		m.agentActive = false
		if m.agentStatus == "running" {
			m.agentStatus = "done"
		}
		// Don't re-register waitForEvents here — the runner's channels are closed.
		// The next Enter press will register a new one.

	case TickMsg:
		// Schedule next tick
		cmds = append(cmds, m.tick())
	}

	// ── Update input (only when focused) ────────────
	if m.focus == "input" {
		var inputCmd tea.Cmd
		m.input, inputCmd = m.input.Update(msg)
		cmds = append(cmds, inputCmd)
	}

	// ── Update viewport ────────────────────────────
	var vpCmd tea.Cmd
	m.viewport, vpCmd = m.viewport.Update(msg)
	cmds = append(cmds, vpCmd)

	return m, tea.Batch(cmds...)
}

// ── View ─────────────────────────────────────────────────────

func (m Model) View() string {
	if m.quitting {
		return "\n  Goodbye!\n\n"
	}

	if m.showHelp {
		return m.helpView()
	}

	// ── Layout ─────────────────────────────────────────
	// Sidebar | Main content
	//         | Output area (viewport)
	//         | Input area (bottom)

	sidebarView := m.sidebarView()
	mainView := m.mainView()
	inputView := m.inputView()
	statusBar := m.statusBarView()

	// Split: sidebar (fixed) | main (rest)
	sideWidth := m.sidebarWidth
	if m.width < 80 {
		sideWidth = 20 // narrower on small terminals
	}
	mainWidth := m.width - sideWidth - 1 // -1 for border

	sidebarRendered := SidebarStyle.
		Width(sideWidth).
		Render(sidebarView)

	mainRendered := lipgloss.NewStyle().
		Width(mainWidth).
		Render(lipgloss.JoinVertical(
			lipgloss.Top,
			mainView,
			inputView,
			statusBar,
		))

	return lipgloss.JoinHorizontal(
		lipgloss.Top,
		sidebarRendered,
		mainRendered,
	)
}

// ── Sidebar ─────────────────────────────────────────────────

func (m Model) sidebarView() string {
	var b strings.Builder

	// Status section
	b.WriteString(SidebarTitle.Render("📊 Status"))
	b.WriteString("\n")
	b.WriteString(StatusLabel.Render("Status:") + " " + m.statusStyle().Render(m.agentStatus) + "\n")
	b.WriteString(StatusLabel.Render("Model:") + " " + StatusValue.Render(m.modelName) + "\n")
	if m.agentActive {
		elapsed := time.Since(m.startTime).Round(time.Second)
		b.WriteString(StatusLabel.Render("Time:") + " " + StatusValue.Render(elapsed.String()) + "\n")
		b.WriteString(StatusLabel.Render("Turns:") + " " + StatusValue.Render(fmt.Sprintf("%d", m.turnCount)) + "\n")
	} else if m.agentStatus == "done" {
		b.WriteString(StatusLabel.Render("Time:") + " " + StatusValue.Render(fmt.Sprintf("%.1fs", float64(m.durationMs)/1000)) + "\n")
		b.WriteString(StatusLabel.Render("Turns:") + " " + StatusValue.Render(fmt.Sprintf("%d", m.turnCount)) + "\n")
	}

	// Quick commands section
	b.WriteString("\n")
	b.WriteString(QuickCommandTitle.Render("⚡ Quick"))
	b.WriteString("\n")
	b.WriteString(QuickBtn.Render("F5 EDA") + "\n")
	b.WriteString(QuickBtn.Render("F6 Pipeline") + "\n")
	b.WriteString(QuickBtn.Render("^N New") + "\n")
	b.WriteString(QuickBtn.Render("^C Stop") + "\n")

	// Keyboard shortcuts
	b.WriteString("\n")
	b.WriteString(QuickCommandTitle.Render("⌨️ Keys"))
	b.WriteString("\n")
	b.WriteString(FileItem.Render("Enter submit") + "\n")
	b.WriteString(FileItem.Render("↑↓ history") + "\n")
	b.WriteString(FileItem.Render("^E focus input") + "\n")
	b.WriteString(FileItem.Render("^L clear") + "\n")
	b.WriteString(FileItem.Render("? help") + "\n")

	// Files section (placeholder)
	b.WriteString("\n")
	b.WriteString(FilesTitle.Render("📁 Results"))
	b.WriteString("\n")
	if len(m.filesList) == 0 {
		b.WriteString(FileItem.Render("(run analysis to see files)") + "\n")
	} else {
		for _, f := range m.filesList {
			b.WriteString(FileItem.Render("· "+f) + "\n")
		}
	}

	return b.String()
}

// ── Main content ─────────────────────────────────────────────

// buildConversation assembles all history + current streaming output into one string.
func (m Model) buildConversation() string {
	contentWidth := m.contentWidth()
	var b strings.Builder
	for i, entry := range m.outputHistory {
		if i > 0 {
			b.WriteString("\n")
			b.WriteString(strings.Repeat("─", min(contentWidth, 20)))
			b.WriteString("\n\n")
		}
		b.WriteString(entry)
	}
	if m.streamingOutput != "" {
		if len(m.outputHistory) > 0 {
			b.WriteString("\n")
			b.WriteString(strings.Repeat("─", min(contentWidth, 20)))
			b.WriteString("\n\n")
		}
		b.WriteString(m.streamingOutput)
	}
	return b.String()
}

func (m Model) mainView() string {
	contentWidth := m.contentWidth()
	contentHeight := m.contentHeight()

	title := OutputTitle.Render("🧠 Comp-Neuroscientist")

	displayContent := m.buildConversation()
	if displayContent == "" {
		displayContent = "Ready. Describe your neuroscience analysis task below.\n\n" +
			"Examples:\n" +
			"  · \"Load BOLD data and compute functional connectivity\"\n" +
			"  · \"Run spike sorting on neuropixels recording\"\n" +
			"  · \"EEG time-frequency analysis on face vs house\"\n" +
			"  · \"Simulate LIF network with STDP\"\n" +
			"  · \"Permutation test with cluster correction\""
	}

	m.viewport.Width = contentWidth
	m.viewport.Height = contentHeight
	m.viewport.SetContent(displayContent)

	return lipgloss.JoinVertical(
		lipgloss.Top,
		title,
		m.viewport.View(),
	)
}

// ── Input area ─────────────────────────────────────────────

func (m Model) inputView() string {
	prompt := InputPrompt.Render("┃ ")
	help := InputHelp.Render("Enter to submit · Shift+Enter newline · Esc focus main · ? help")

	return InputContainer.Render(
		prompt + m.input.View() + "\n" + help,
	)
}

// ── Status bar ─────────────────────────────────────────────

func (m Model) statusBarView() string {
	statusText := m.statusStyle().Render(strings.ToUpper(m.agentStatus))
	mode := "NORMAL"
	if m.focus == "input" {
		mode = "INPUT"
	} else if m.focus == "main" {
		mode = "OUTPUT"
	}
	if m.showHelp {
		mode = "HELP"
	}

	modelLabel := m.modelName
	if len(modelLabel) > 20 {
		modelLabel = modelLabel[:17] + "..."
	}

	return StatusBar.Render(
		fmt.Sprintf("%s │ %s │ %s",
			mode,
			statusText,
			"comp-neuroscientist v0.1.0",
		),
	)
}

// ── Help view ─────────────────────────────────────────────

func (m Model) helpView() string {
	return help.New().View(keys)
}

// ── Helpers ───────────────────────────────────────────────

func (m Model) contentWidth() int {
	return m.width - m.sidebarWidth - 6 // -6 for padding
}

func (m Model) contentHeight() int {
	return m.height - 8 // -8 for title, input, status bar, padding
}

// formatArgs formats tool call arguments for display.
func formatArgs(args map[string]interface{}) string {
	if args == nil {
		return ""
	}
	// Show the most relevant field
	if cmd, ok := args["command"]; ok {
		s := fmt.Sprintf("%v", cmd)
		if len(s) > 60 {
			s = s[:57] + "..."
		}
		return s
	}
	if path, ok := args["file_path"]; ok {
		return fmt.Sprintf("%v", path)
	}
	return fmt.Sprintf("%v", args)
}

// statusStyle returns the appropriate style for the current status.
func (m Model) statusStyle() lipgloss.Style {
	switch m.agentStatus {
	case "running":
		return StatusRunning
	case "done":
		return StatusDone
	case "error":
		return StatusError
	default:
		return StatusReady
	}
}

// sanitizePaths replaces absolute home directory paths with ~ for privacy.
func sanitizePaths(text, homeDir string) string {
	if homeDir == "" {
		return text
	}
	return strings.ReplaceAll(text, homeDir, "~")
}

// ── Key bindings ───────────────────────────────────────────

type keyMap struct {
	Help       key.Binding
	Quit       key.Binding
	Stop       key.Binding
	FocusInput key.Binding
	FocusOutput key.Binding
	Clear      key.Binding
	Up         key.Binding
	Down       key.Binding
}

var keys = keyMap{
	Help: key.NewBinding(
		key.WithKeys("?"),
		key.WithHelp("?", "help"),
	),
	Quit: key.NewBinding(
		key.WithKeys("ctrl+c", "q"),
		key.WithHelp("q/^C", "quit"),
	),
	Stop: key.NewBinding(
		key.WithKeys("ctrl+z"),
		key.WithHelp("^Z", "stop agent"),
	),
	FocusInput: key.NewBinding(
		key.WithKeys("ctrl+e"),
		key.WithHelp("^E", "focus input"),
	),
	FocusOutput: key.NewBinding(
		key.WithKeys("esc"),
		key.WithHelp("esc", "focus output"),
	),
	Clear: key.NewBinding(
		key.WithKeys("ctrl+l"),
		key.WithHelp("^L", "clear output"),
	),
}

// ShortHelp returns key bindings for the help view.
func (k keyMap) ShortHelp() []key.Binding {
	return []key.Binding{k.Help, k.Quit, k.Stop, k.FocusInput, k.Clear}
}

// FullHelp returns all key bindings for the help view.
func (k keyMap) FullHelp() [][]key.Binding {
	return [][]key.Binding{
		{k.FocusInput, k.FocusOutput, k.Clear},
		{k.Stop, k.Help, k.Quit},
	}
}

// max returns the larger of two ints.
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// min returns the smaller of two ints.
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// round rounds a float64 to the nearest integer.
func round(f float64) int {
	return int(math.Round(f))
}

// Ensure keyMap implements help.KeyMap
var _ help.KeyMap = keys

// Key binding for Enter (used in KeyMsg handler)
const KeyEnter = "enter"
