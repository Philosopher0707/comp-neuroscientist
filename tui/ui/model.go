// Package ui contains the Bubble Tea model, view rendering, and event handling
// for the Comp-Neuroscientist TUI.
package ui

import (
	"fmt"
	"math"
	"os"
	"os/user"
	"strings"
	"time"

	"charm.land/bubbles/v2/help"
	"charm.land/bubbles/v2/key"
	"charm.land/bubbles/v2/textinput"
	"charm.land/bubbles/v2/viewport"
	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"

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
	runID      int    // incremented each Start() to detect stale AgentDoneMsg

	// Output
	streamingOutput string    // accumulated text from current run (Pi-style)
	outputHistory   []string  // completed output blocks
	viewport        viewport.Model

	// Input
	input textinput.Model

	// Sidebar
	sidebarWidth int
	filesList    []string
	modelName    string

	// Pre-rendered history (rendered once when a turn completes, not on every chunk)
	renderedHistory string

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
	ti.SetWidth(60)
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
		input: ti,
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
	RunID int
	Event *protocol.Event
}

// AgentDoneMsg is sent when the agent process exits.
type AgentDoneMsg struct {
	RunID int
}
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
	runID := m.runID // capture current run ID — reject if stale
	return func() tea.Msg {
		// Non-blocking drain: consume any buffered event before waiting.
		// This prevents data loss when Done() closes simultaneously with
		// the last buffered events still in the channel.
		select {
		case ev, ok := <-m.agentRunner.Events():
			if ok {
				return AgentEventMsg{RunID: runID, Event: ev}
			}
			// Channel closed — agent is done
			return AgentDoneMsg{RunID: runID}
		default:
		}

		// Blocking wait for the next event or done signal
		select {
		case ev, ok := <-m.agentRunner.Events():
			if !ok {
				return AgentDoneMsg{RunID: runID}
			}
			return AgentEventMsg{RunID: runID, Event: ev}
		case <-m.agentRunner.Done():
			return AgentDoneMsg{RunID: runID}
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
		m.help.SetWidth(msg.Width)
		m.viewport.SetWidth(m.contentWidth())
		m.viewport.SetHeight(m.contentHeight())
		m.input.SetWidth(m.contentWidth() - 6)
		if m.viewport.GetContent() == "" {
			m.viewport.SetContent("Ready. Describe your neuroscience analysis task below.\n\nExamples:\n  · \"Load BOLD data and compute functional connectivity\"\n  · \"Run spike sorting on neuropixels recording\"\n  · \"EEG time-frequency analysis on face vs house\"\n  · \"Simulate LIF network with STDP\"\n  · \"Permutation test with cluster correction\"")
		}

	// ── Key events ──────────────────────────────────

	case tea.KeyPressMsg:
		// Enter always submits if there's text and agent is idle — regardless of focus
		if msg.String() == "enter" && m.input.Value() != "" && !m.agentActive {
			prompt := m.input.Value()
			m.input.SetValue("")
			m.agentStatus = "running"
			m.agentActive = true
			m.runID++
			m.startTime = time.Now()
			m.turnCount = 0

			// Start new output with user's prompt visible, keeping history in view
			m.streamingOutput = fmt.Sprintf("> %s\n\n", prompt)
			displayText := m.renderedHistory
			if m.renderedHistory != "" {
				displayText += fmt.Sprintf("\n%s\n\n", strings.Repeat("─", min(m.contentWidth(), 20)))
			}
			displayText += sanitizePaths(RenderMarkdown(m.streamingOutput), m.homeDir)
			m.viewport.SetWidth(m.contentWidth())
			m.viewport.SetHeight(m.contentHeight())
			m.viewport.SetContent(displayText)
			m.viewport.GotoBottom()

			localMode := !strings.Contains(m.modelName, ":cloud")
			if err := m.agentRunner.Start(prompt, m.modelName, localMode); err != nil {
				m.agentActive = false // FIX: reset so next Enter works
				m.agentStatus = "error"
				m.streamingOutput += fmt.Sprintf("\n❌ Error: %v\n", err)
				// Refresh viewport so user sees the error
				fullConv := m.buildConversation()
				displayText = sanitizePaths(RenderMarkdown(fullConv), m.homeDir)
				m.viewport.SetContent(displayText)
				m.viewport.GotoBottom()
				return m, nil
			}
			// Re-focus input for next prompt
			m.focus = "input"
			m.input.Focus()
			// Register event listener for this run
			cmds = append(cmds, m.waitForEvents())
			return m, tea.Batch(cmds...)
		}

		// Global keys
		switch {
		case key.Matches(msg, keys.Help):
			if m.focus == "input" && msg.String() == "?" {
				break
			}
			m.showHelp = !m.showHelp
			return m, nil
		case key.Matches(msg, keys.Quit):
			if m.focus == "input" && msg.String() == "q" {
				break
			}
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
			m.renderedHistory = ""
			m.viewport.SetContent("")
			m.viewport.GotoTop()
			return m, nil
		}

		// If Enter was not handled above and textinput needs it, pass through
		// (textinput does not consume Enter, so this is a no-op fallback)

	// ── Agent events ────────────────────────────────

	case AgentEventMsg:
		// Reject events from a previous run (stale goroutine)
		if msg.RunID != m.runID {
			break
		}
		ev := msg.Event
		switch ev.Type {
		case protocol.EventText:
			m.streamingOutput += ev.Content
			// Show: pre-rendered history + current streaming (rendered fresh)
			displayText := m.renderedHistory
			if m.renderedHistory != "" && m.streamingOutput != "" {
				displayText += fmt.Sprintf("\n%s\n\n", strings.Repeat("─", min(m.contentWidth(), 20)))
			}
			displayText += sanitizePaths(RenderMarkdown(m.streamingOutput), m.homeDir)
			m.viewport.SetContent(displayText)
			m.viewport.GotoBottom()
			cmds = append(cmds, m.waitForEvents())

		case protocol.EventToolCall:
			callLine := fmt.Sprintf("  🔧 %s(%s)", ev.Name, formatArgs(ev.Arguments))
			m.streamingOutput += "\n" + callLine + "\n"
			displayText := m.renderedHistory
			if m.renderedHistory != "" {
				displayText += fmt.Sprintf("\n%s\n\n", strings.Repeat("─", min(m.contentWidth(), 20)))
			}
			displayText += sanitizePaths(RenderMarkdown(m.streamingOutput), m.homeDir)
			m.viewport.SetContent(displayText)
			m.viewport.GotoBottom()
			cmds = append(cmds, m.waitForEvents())

		case protocol.EventToolResult:
			if len(ev.Result) > 80 {
				m.streamingOutput += fmt.Sprintf("  └─ [%d chars]\n", len(ev.Result))
			} else {
				m.streamingOutput += fmt.Sprintf("  └─ %s\n", ev.Result)
			}
			displayText := m.renderedHistory
			if m.renderedHistory != "" {
				displayText += fmt.Sprintf("\n%s\n\n", strings.Repeat("─", min(m.contentWidth(), 20)))
			}
			displayText += sanitizePaths(RenderMarkdown(m.streamingOutput), m.homeDir)
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
			m = m.addRenderedHistory()
			m.refreshFilesList()
			// Re-render conversation
			fullConv := m.buildConversation()
			displayText := sanitizePaths(RenderMarkdown(fullConv), m.homeDir)
			m.viewport.SetContent(displayText)
			m.viewport.GotoBottom()
			// Don't re-register waitForEvents — the agent process is done
			// and AgentDoneMsg will arrive as channels close

		case protocol.EventError:
			m.agentActive = false
			m.agentStatus = "error"
			errorLine := fmt.Sprintf("\n❌ Error: %s\n", ev.Message)
			m.streamingOutput += errorLine
			m = m.addRenderedHistory()
			// Show error in context of conversation history
			fullConv := m.buildConversation()
			displayText := sanitizePaths(RenderMarkdown(fullConv), m.homeDir)
			m.viewport.SetContent(displayText)
			m.viewport.GotoBottom()
			// Don't re-register waitForEvents — the agent process is done
		}



	case AgentDoneMsg:
		// Ignore stale DoneMsg from a previous run (only current or newer runIDs are valid)
		if msg.RunID != 0 && msg.RunID < m.runID {
			break
		}
		m.agentActive = false
		// Don't overwrite a more specific status set by EventResult or EventError
		if m.agentStatus == "running" {
			m.agentStatus = "done"
		}
		// If nothing was ever rendered (agent died silently), show something
		if m.streamingOutput == "" && len(m.renderedHistory) == 0 {
			m.streamingOutput = "\n⚠️ Agent exited without producing output.\n"
		}
		// Persist partial output so it survives into the next turn
		m = m.addRenderedHistory()
		fullConv := m.buildConversation()
		displayText := sanitizePaths(RenderMarkdown(fullConv), m.homeDir)
		m.viewport.SetContent(displayText)
		m.viewport.GotoBottom()
		// Runner's channels are closed — next Enter press registers new listener

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
	// Skip WindowSizeMsg — we handle dimensions manually in the case above
	// to prevent the viewport's internal resize handler from duplicating content.
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		// handled above
	default:
		var vpCmd tea.Cmd
		if _, isKey := msg.(tea.KeyMsg); isKey {
			if m.focus == "main" {
				m.viewport, vpCmd = m.viewport.Update(msg)
			}
		} else {
			m.viewport, vpCmd = m.viewport.Update(msg)
		}
		cmds = append(cmds, vpCmd)
	}

	return m, tea.Batch(cmds...)
}

// ── View ─────────────────────────────────────────────────────

func (m Model) View() tea.View {
	if m.quitting {
		return tea.NewView("\n  Goodbye!\n\n")
	}

	if m.showHelp {
		return tea.NewView(m.helpView())
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
	if mainWidth < 1 {
		mainWidth = 1
	}

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

	v := tea.NewView(lipgloss.JoinHorizontal(
		lipgloss.Top,
		sidebarRendered,
		mainRendered,
	))
	v.AltScreen = true
	v.MouseMode = tea.MouseModeCellMotion
	return v
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
	b.WriteString(QuickBtn.Render("^K Stop") + "\n")

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

// buildConversation assembles history blocks above the current streaming output.
// History is static text (rendered once on completion).
// Streaming output is the active turn being built.
func (m Model) buildConversation() string {
	sepWidth := min(m.contentWidth(), 20)
	var b strings.Builder
	for i, entry := range m.outputHistory {
		if i > 0 {
			b.WriteString("\n")
			b.WriteString(strings.Repeat("─", sepWidth))
			b.WriteString("\n\n")
		}
		b.WriteString(entry)
	}
	if m.streamingOutput != "" {
		if len(m.outputHistory) > 0 {
			b.WriteString("\n")
			b.WriteString(strings.Repeat("─", sepWidth))
			b.WriteString("\n\n")
		}
		b.WriteString(m.streamingOutput)
	}
	return b.String()
}

// maxHistoryEntries caps the number of conversation turns kept in memory.
const maxHistoryEntries = 10

// addRenderedHistory appends the final streaming output to history as pre-rendered text,
// then caps history to maxHistoryEntries and rebuilds the renderedHistory cache.
func (m Model) addRenderedHistory() Model {
	if m.streamingOutput == "" {
		return m
	}
	// Save raw text to history for future access
	m.outputHistory = append(m.outputHistory, m.streamingOutput)
	m.streamingOutput = ""

	// Cap history to prevent unbounded memory growth
	if len(m.outputHistory) > maxHistoryEntries {
		excess := len(m.outputHistory) - maxHistoryEntries
		// Copy to new slice so GC can collect evicted entries
		trimmed := make([]string, len(m.outputHistory)-excess)
		copy(trimmed, m.outputHistory[excess:])
		m.outputHistory = trimmed
	}

	// Rebuild pre-rendered history (render all entries once, not per chunk)
	return m.rebuildRenderedHistory()
}

// rebuildRenderedHistory pre-renders all history entries into renderedHistory.
// Called once when a turn completes, not on every streaming chunk.
func (m Model) rebuildRenderedHistory() Model {
	sepWidth := min(m.contentWidth(), 20)
	var b strings.Builder
	for i, entry := range m.outputHistory {
		if i > 0 {
			b.WriteString("\n")
			b.WriteString(strings.Repeat("─", sepWidth))
			b.WriteString("\n\n")
		}
		b.WriteString(entry)
	}
	m.renderedHistory = sanitizePaths(RenderMarkdown(b.String()), m.homeDir)
	return m
}

func (m Model) mainView() string {
	contentWidth := m.contentWidth()

	title := OutputTitle.Render("🧠 Comp-Neuroscientist")

	// Use pre-rendered history + fresh render of current streaming output
	// This avoids re-rendering the full conversation on every View() call
	displayContent := m.renderedHistory
	if m.streamingOutput != "" {
		if m.renderedHistory != "" {
			displayContent += fmt.Sprintf("\n%s\n\n", strings.Repeat("─", min(contentWidth, 20)))
		}
		displayContent += sanitizePaths(RenderMarkdown(m.streamingOutput), m.homeDir)
	}
	if displayContent == "" {
		displayContent = "Ready. Describe your neuroscience analysis task below.\n\n" +
			"Examples:\n" +
			"  · \"Load BOLD data and compute functional connectivity\"\n" +
			"  · \"Run spike sorting on neuropixels recording\"\n" +
			"  · \"EEG time-frequency analysis on face vs house\"\n" +
			"  · \"Simulate LIF network with STDP\"\n" +
			"  · \"Permutation test with cluster correction\""
	}


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
	runes := []rune(modelLabel)
	if len(runes) > 20 {
		modelLabel = string(runes[:17]) + "..."
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

// ── Layout helpers ──────────────────────────────────────────

// contentWidthPadding is the horizontal space consumed by layout chrome
// between the outer window edge and the inner content area:
//   - Sidebar right border: 1
//   - Sidebar right padding: 1
//   - Visual gap: 4
const contentWidthPadding = 6

// contentHeightPadding is the vertical space consumed by title, input
// area, status bar, and gaps between them:
//   - OutputTitle + border: 2
//   - InputContainer + border: 2
//   - StatusBar: 1
//   - Gaps: 3
const contentHeightPadding = 8

func (m Model) contentWidth() int {
	w := m.width - m.sidebarWidth - contentWidthPadding
	if w < 1 {
		w = 1
	}
	return w
}

func (m Model) contentHeight() int {
	h := m.height - contentHeightPadding
	if h < 1 {
		h = 1
	}
	return h
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

// refreshFilesList scans the results directory for output files and updates the sidebar.
func (m *Model) refreshFilesList() {
	entries, err := os.ReadDir("results")
	if err != nil {
		m.filesList = nil
		return
	}
	var files []string
	for _, e := range entries {
		if !e.IsDir() {
			files = append(files, e.Name())
		}
	}
	// Also scan subdirectories
	for _, sub := range []string{"plots", "models", "processed"} {
		subEntries, err := os.ReadDir("results/" + sub)
		if err != nil {
			continue
		}
		for _, e := range subEntries {
			if !e.IsDir() {
				files = append(files, sub+"/"+e.Name())
			}
		}
	}
	if len(files) > 20 {
		files = files[:20]
	}
	if files == nil {
		files = []string{}
	}
	m.filesList = files
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
		key.WithKeys("ctrl+k"),
		key.WithHelp("^K", "stop agent"),
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
