// Package ui contains the Bubble Tea model, views, and Lip Gloss styles
// for the Comp-Neuroscientist TUI.
package ui

import (
	"github.com/charmbracelet/lipgloss"
)

// ── Color palette (Tokyo Night inspired) ──────────────────────────

var (
	ColorBG          = lipgloss.Color("#1a1b26")
	ColorSidebarBG   = lipgloss.Color("#24253a")
	ColorBorder      = lipgloss.Color("#3b3d5e")
	ColorBlue        = lipgloss.Color("#7aa2f7")
	ColorCyan        = lipgloss.Color("#7dcfff")
	ColorPurple      = lipgloss.Color("#bb9af7")
	ColorGreen       = lipgloss.Color("#73daca")
	ColorOrange      = lipgloss.Color("#e0af68")
	ColorRed         = lipgloss.Color("#f7768e")
	ColorText        = lipgloss.Color("#c0caf5")
	ColorSubtext     = lipgloss.Color("#a9b1d6")
	ColorMuted       = lipgloss.Color("#565f89")
	ColorInputBG     = lipgloss.Color("#1a1b26")
	ColorInputBorder = lipgloss.Color("#7aa2f7")
)

// ── Component styles ─────────────────────────────────────────────

var (
	// App background
	AppStyle = lipgloss.NewStyle().
			Background(ColorBG)

	// Sidebar
	SidebarStyle = lipgloss.NewStyle().
			Width(30).
			Background(ColorSidebarBG).
			BorderStyle(lipgloss.NormalBorder()).
			BorderRight(true).
			BorderForeground(ColorBorder).
			Padding(0, 1)

	SidebarTitle = lipgloss.NewStyle().
			Foreground(ColorBlue).
			Bold(true).
			Padding(1, 0)

	StatusLabel = lipgloss.NewStyle().
			Foreground(ColorMuted).
			Width(9)

	StatusValue = lipgloss.NewStyle().
			Foreground(ColorText).
			MaxWidth(18)

	QuickCommandTitle = lipgloss.NewStyle().
				Foreground(ColorCyan).
				Bold(true).
				Padding(1, 0, 0, 0)

	QuickBtn = lipgloss.NewStyle().
			Background(ColorBorder).
			Foreground(ColorText).
			Padding(0, 1).
			Margin(0, 0, 1, 0)

	QuickBtnFocused = lipgloss.NewStyle().
			Background(ColorBlue).
			Foreground(ColorBG).
			Padding(0, 1).
			Margin(0, 0, 1, 0)

	FilesTitle = lipgloss.NewStyle().
			Foreground(ColorPurple).
			Bold(true).
			Padding(1, 0, 0, 0)

	FileItem = lipgloss.NewStyle().
			Foreground(ColorSubtext)

	// Main content area
	MainContentStyle = lipgloss.NewStyle().
				Padding(0, 2)

	OutputTitle = lipgloss.NewStyle().
			Foreground(ColorBlue).
			Bold(true).
			Padding(1, 0).
			BorderStyle(lipgloss.NormalBorder()).
			BorderBottom(true).
			BorderForeground(ColorBorder)

	// Streaming output
	StreamingText = lipgloss.NewStyle().
			Foreground(ColorText)

	ToolCallText = lipgloss.NewStyle().
			Foreground(ColorOrange).
			Italic(true)

	ToolResultText = lipgloss.NewStyle().
			Foreground(ColorMuted)

	ErrorText = lipgloss.NewStyle().
			Foreground(ColorRed).
			Bold(true)

	DoneText = lipgloss.NewStyle().
			Foreground(ColorGreen).
			Bold(true)

	// Input area
	InputContainer = lipgloss.NewStyle().
			Background(ColorSidebarBG).
			BorderStyle(lipgloss.NormalBorder()).
			BorderTop(true).
			BorderForeground(ColorBorder)

	InputPrompt = lipgloss.NewStyle().
			Foreground(ColorBlue).
			Bold(true)

	InputHelp = lipgloss.NewStyle().
			Foreground(ColorMuted).
			Italic(true).
			Padding(0, 2)

	// Status bar
	StatusBar = lipgloss.NewStyle().
			Foreground(ColorMuted).
			Padding(0, 2)

	// Status indicators
	StatusRunning = lipgloss.NewStyle().
			Foreground(ColorGreen)

	StatusDone = lipgloss.NewStyle().
			Foreground(ColorGreen)

	StatusError = lipgloss.NewStyle().
			Foreground(ColorRed)

	StatusReady = lipgloss.NewStyle().
			Foreground(ColorMuted)
)

// QuickCmdStyle returns the appropriate style based on focus state.
func QuickCmdStyle(focused bool) lipgloss.Style {
	if focused {
		return QuickBtnFocused
	}
	return QuickBtn
}
