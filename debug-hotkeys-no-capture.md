# Debug Session: hotkeys-no-capture
- **Status**: [OPEN]
- **Issue**: Live capture/process hotkeys are not triggering screenshot capture or processing during interactive use. ESC still exits the listener.
- **Debug Server**: pending startup
- **Log File**: .dbg/trae-debug-log-hotkeys-no-capture.ndjson

## Reproduction Steps
1. Run the application in live listener mode.
2. Press the expected capture and process keys during an active session.
3. Observe whether screenshots are captured, processing starts, or only ESC is handled.

## Hypotheses & Verification
| ID | Hypothesis | Likelihood | Effort | Evidence |
|----|------------|------------|--------|----------|
| A | The user is pressing `F1`/`F2`, but the application now expects modifier combos, so no capture/process action should fire. | High | Low | Pending |
| B | The listener is receiving ESC but not regular/modifier keys because macOS Input Monitoring / Accessibility permissions are incomplete. | High | Low | Pending |
| C | The new combo-tracking logic is not normalizing `Option`/number keys correctly on this keyboard layout, so the combo never becomes active. | Medium | Medium | Pending |
| D | The terminal or IDE is intercepting some keys before `pynput` sees them, causing partial or no events for capture/process keys. | Medium | Medium | Pending |
| E | The listener thread is alive, but the callback path into `handle_capture` / `handle_process` is failing silently before user-visible effects. | Low | Medium | Pending |

## Log Evidence
- `.dbg/trae-debug-log-hotkeys-no-capture.ndjson`
- `F1` and `F2` were received by the listener as `Key.f1` and `Key.f2`.
- The combo-only implementation ignored those events because it only acted on normalized modifier tokens, not direct function keys.
- `handle_capture` / `handle_process` did not run in the failing build, so the break was in hotkey matching rather than in screenshot capture itself.
- The screenshot also shows TRAE opening its command palette on `F1`, which means the IDE still binds that key in the integrated terminal.
- After restoring direct `F1` / `F2`, user verification indicates: capture works again, but TRAE still pops its command UI when `F1` is pressed inside the integrated terminal.

## Verification Conclusion
- Root cause confirmed: the recent hotkey refactor changed live capture/process from direct `F1` / `F2` to a combo-only matcher, which broke the legacy flow even though `F1` / `F2` were still arriving.
- Secondary environmental factor: when running inside the TRAE integrated terminal, `F1` is also handled by the IDE, so the command palette may still appear even while the app receives the key.
- Minimal fix applied: restore direct `F1` / `F2` handling in the listener while keeping instrumentation for post-fix verification.
- Current status: application behavior is corrected; remaining interference is IDE-level keybinding behavior, not a failure to capture inside the app.
