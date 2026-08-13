# Homeostat Agent

A cybernetic problem-solving system for macOS: an autonomous agent that perceives a problem, constructs a solution, verifies it by execution, and iterates on its own failures until it delivers a proven result. The human does not guide the work — the human supervises it, acting as a governor that authorizes the effort (time, attempts, API calls) the agent may spend on a solution.

The project is named after W. Ross Ashby's **Homeostat** (1948), the first machine built to exhibit ultrastability: when its behavior failed, it reconfigured its own internal parameters until it found a working configuration. This agent applies the same principle to problem-solving — and it is deliberately **not wired to any specific kind of challenge**, following Ashby's Law of Requisite Variety: only internal variety can absorb the variety of arbitrary problems.

## Scientific Background

The design draws on cybernetics (Wiener; Ashby; Beer), the science of self-regulating, goal-directed systems:

- **Closed-loop regulation (feedback).** The core of the agent is an error-controlled loop: solve, execute, compare against expectations, feed the failure back as context, and solve again. A verified solution is a fixed point of this solve-verify dynamic — the loop converges or the attempt budget expires.
- **Ultrastability (Ashby).** Adaptation operates at two levels. The inner loop adjusts the *solution* when tests fail. The outer loop adjusts the *approach*: the human can inject a directive (F9) that changes the parameters of the search itself, and rejected answers captured from the environment re-enter as feedback for the next attempt.
- **Homeostasis.** The system restores its own functions after disturbance without human repair: the global hotkey monitor is a self-healing Quartz event tap that re-enables itself when macOS disables it, and the notification backend detects machine capabilities at runtime (pync where available, native AppleScript otherwise).
- **Human-on-the-loop (supervisory control).** In classical AI assistants the human is *in* the loop, steering each step. Here the human is *on* the loop: the agent runs autonomously and the human acts as a governor, triggering work and authorizing how much effort may be expended — a viability constraint over the agent's dynamics, not a navigator inside them.

The longer-term research direction is complexity science proper: multiple interacting solving strategies whose selection emerges from measured feedback dynamics (convergence rates, failure basins, stability of the iteration), moving the system from a cybernetic regulator toward a complex adaptive system.

## How the Concepts Map to the System

| Cybernetic concept | Implementation |
|---|---|
| Perception (sensory channel) | Full-screen capture (Pillow) + OCR (Tesseract) in a parallel background worker |
| Distillation | OCR text is parsed; irrelevant windows and UI chrome are filtered out |
| Effector (action channel) | Solutions generated via the DeepSeek API; answers delivered to the macOS clipboard |
| Error-controlled feedback | Solutions are executed and tested in a sandboxed temp directory; failures loop back as context (up to 4 attempts, extendable) |
| Environmental feedback | Rejected answers / grader messages recaptured with F6 re-enter as task context |
| Ultrastable parameter shift | F9 user directives steer approach, debugging, or emphasis on the next attempt |
| Homeostatic recovery | Self-healing hotkey tap; runtime-adaptive notification backend |
| Governor (viability constraint) | The human triggers capture/process and controls attempt and retry budgets |
| Memory / provenance | SQLite database plus append-only provenance log of every question, answer, code version, and event |

## Features

- Global hotkeys (F6/F7/F8/F9) monitored via a self-healing Quartz event tap — they keep working from any app and any desktop/Space, even after macOS disables the tap
- Full-screen capture with Pillow; OCR via Tesseract runs in a background worker and is parallelized across CPU cores
- Question parsing, answering, and solution generation via the DeepSeek API (screenshots are distilled to OCR text; irrelevant windows/UI chrome are filtered out at parse time)
- Generated solutions are executed and tested in a sandboxed temp directory; failed attempts loop with failure feedback (up to 4 attempts) until tests pass
- User directives (F9): free-form guidance notes that are attached to the next processing attempt, steering the solver without editing code
- Answers are copied straight to the macOS clipboard for exact pasting (no transcription typos)
- Every question, answer, code version, and event is stored in a SQLite database plus an append-only provenance log

## Requirements

- macOS (the global hotkey monitor and clipboard integration are macOS-only)
- Python 3.13+ (3.13 requires pynput >= 1.8.1, already pinned)
- [Tesseract](https://formulae.brew.sh/formula/tesseract) OCR binary: `brew install tesseract`
- A DeepSeek API key

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/albertoHdzE/homeostat-agent.git
   cd homeostat-agent
   ```

2. Create a virtual environment and install dependencies:
   ```
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Provide your DeepSeek API key in a `.env` file (see `.env.example`):
   ```
   cp .env.example .env
   # edit .env and set DEEPSEEK_API_KEY=...
   ```

4. Install the OCR binary if you haven't:
   ```
   brew install tesseract
   ```

## Usage

1. Run the script:
   ```
   python main.py
   ```

2. On first run, macOS will ask for permissions for the terminal app — grant both, then fully quit and relaunch the terminal:
   - **Accessibility** (System Settings → Privacy & Security → Accessibility) — required for global hotkeys
   - **Screen Recording** (same panel) — required for screenshots to contain window content

3. When you encounter a problem:
   - Press `F6` to capture a screenshot from anywhere on your Mac
   - Press `F7` to process the current screenshot batch (waits for background OCR, then parses, solves, and self-tests)
   - Press `F9` to add a directive: the terminal prompts for a free-form note (e.g. "prefer an O(n log n) approach" or "the expected output needs exactly two decimal places"). Pending directives are injected into the next `F7` processing attempt as guidance for the solver, then marked consumed. Directives steer approach, debugging, or emphasis; they do not override the problem statement.
   - Press `F8` to quit the listener

   Function keys are used so the hotkeys never collide with text you type into the target interface. If your keyboard maps the F-keys to media functions, either press them together with `fn` or enable "Use F1, F2, etc. keys as standard function keys" in System Settings → Keyboard.

4. The final answer is placed on your clipboard automatically — paste it with Cmd+V instead of retyping it.

   Notifications adapt to the machine automatically: on Intel Macs and Apple Silicon with Rosetta 2 they use pync/terminal-notifier; on Apple Silicon without Rosetta (where pync's Intel-only binary cannot run) the app falls back to native AppleScript notifications. The selected backend is printed at startup and requires no configuration.

5. If the environment rejects an answer, capture the feedback screen with `F6` and press `F7` again: visible feedback (failed tests, grader messages) is parsed as task context for the next attempt. Combine this with an `F9` directive when you want to redirect the approach explicitly.

## Maintenance

- `./clean-env` resets the agent to a pristine state (deletes the database, provenance log, screenshots, sandbox runs, and caches; asks for confirmation, `-y` skips it). State is recreated automatically on the next run.
- Runtime artifacts live in `screenshots/`, `data/`, and `temp_runs/` — all git-ignored.
- Verification helper scripts (require the venv): `temp_runs/hotkey_smoke_test.py`, `temp_runs/async_capture_test.py`, `temp_runs/iterative_loop_test.py`.

## Important Notes

- This tool is for research, educational, and demonstration purposes.
- Use responsibly and ethically.
- Be aware of the terms of service of any platform you interact with.
- This tool is designed specifically for macOS and may not work on other operating systems.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This project explores autonomous, self-regulating problem-solving as a systems-design question. It should not be used to gain unfair advantages in assessments or to violate the rules of any platform. It is specifically designed for macOS users and may not function on other operating systems.
