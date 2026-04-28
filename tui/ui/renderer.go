// Package ui — markdown-to-ANSI renderer for agent output.
//
// Converts common markdown patterns to terminal-formatted text:
//   **bold** → ANSI bold
//   *italic* → ANSI italic
//   `code`   → colored inline code
//   # headers → bold + underline
//   - lists  → indented with bullet
//   ``` ... ``` → fenced code blocks
//
// This is a lightweight renderer — no AST, just regex. Good enough for
// LLM streaming output that uses these patterns pervasively.

package ui

import (
	"regexp"
	"strings"

	"github.com/charmbracelet/lipgloss"
)

var (
	// Bold: **text** or __text__
	boldRe = regexp.MustCompile(`\*\*(.+?)\*\*|__(.+?)__`)

	// Italic: *text* (but not ** which is handled by boldRe first)
	// Since boldRe runs first, remaining *...* must be italic.
	// No lookarounds — Go's RE2 doesn't support them.
	italicRe = regexp.MustCompile(`\*(.+?)\*`)

	// Inline code: `text`
	codeRe = regexp.MustCompile("`([^`]+)`")

	// Headers: # text, ## text, etc.
	headerRe = regexp.MustCompile(`^(#{1,6})\s+(.+)$`)

	// Fenced code block markers: ``` and ```
	fenceOpenRe  = regexp.MustCompile("^```")
	fenceCloseRe = regexp.MustCompile("```$")

	// List items: - text, * text, 1. text
	listRe = regexp.MustCompile(`^(\s*)[\-\*]\s+(.*)$|^(\s*)\d+\.\s+(.*)$`)

	// Horizontal rules
	hrRe = regexp.MustCompile(`^---+\s*$|^\*\*\*+\s*$`)
)

// RenderMarkdown converts a markdown string to ANSI-styled terminal output.
func RenderMarkdown(text string) string {
	lines := strings.Split(text, "\n")
	var result []string
	inFence := false

	for _, line := range lines {
		if fenceOpenRe.MatchString(line) && !inFence {
			inFence = true
			result = append(result, CodeFenceStyle.Render(line))
			continue
		}
		if fenceCloseRe.MatchString(line) && inFence {
			inFence = false
			result = append(result, CodeFenceStyle.Render(line))
			continue
		}
		if inFence {
			result = append(result, CodeBlockStyle.Render(line))
			continue
		}

		rendered := renderLine(line)
		result = append(result, rendered)
	}

	return strings.Join(result, "\n")
}

// renderLine applies inline markdown styles to a single line.
func renderLine(line string) string {
	// Check for headers
	if m := headerRe.FindStringSubmatch(line); m != nil {
		level := len(m[1])
		text := m[2]
		style := HeaderStyle
		if level == 1 {
			style = Header1Style
		} else if level == 2 {
			style = Header2Style
		}
		return style.Render(text)
	}

	// Check for HR
	if hrRe.MatchString(line) {
		return strings.Repeat("─", 40)
	}

	// Check for list items
	if m := listRe.FindStringSubmatch(line); m != nil {
		indent := ""
		itemText := ""
		if m[1] != "" {
			indent = m[1]
			itemText = m[2]
		} else {
			indent = m[3]
			itemText = m[4]
		}
		bullet := ListBulletStyle.Render("• ")
		line = indent + bullet + renderInline(itemText)
		return line
	}

	return renderInline(line)
}

// renderInline applies inline markdown styles (bold, italic, code).
func renderInline(text string) string {
	// Bold first (so ** doesn't get caught by *)
	text = boldRe.ReplaceAllStringFunc(text, func(m string) string {
		inner := boldRe.ReplaceAllString(m, "$1$2")
		return BoldStyle.Render(inner)
	})

	// Italic
	text = italicRe.ReplaceAllStringFunc(text, func(m string) string {
		inner := italicRe.ReplaceAllString(m, "$1$2")
		return ItalicStyle.Render(inner)
	})

	// Code
	text = codeRe.ReplaceAllStringFunc(text, func(m string) string {
		inner := codeRe.ReplaceAllString(m, "$1")
		return InlineCodeStyle.Render(inner)
	})

	return text
}

// ── ANSI styles for markdown elements ─────────────────────────

var (
	BoldStyle = lipgloss.NewStyle().Bold(true)

	ItalicStyle = lipgloss.NewStyle().Italic(true)

	InlineCodeStyle = lipgloss.NewStyle().
			Foreground(ColorGreen).
			Background(ColorSidebarBG).
			Padding(0, 1)

	HeaderStyle = lipgloss.NewStyle().Bold(true)

	Header1Style = lipgloss.NewStyle().
			Bold(true).
			Underline(true).
			Foreground(ColorBlue).
			Padding(0, 0, 1, 0)

	Header2Style = lipgloss.NewStyle().
			Bold(true).
			Foreground(ColorCyan).
			Padding(0, 0, 0, 0)

	ListBulletStyle = lipgloss.NewStyle().Foreground(ColorPurple)

	CodeFenceStyle = lipgloss.NewStyle().Foreground(ColorMuted)

	CodeBlockStyle = lipgloss.NewStyle().
			Foreground(ColorGreen).
			Background(ColorSidebarBG).
			Padding(0, 2)
)
