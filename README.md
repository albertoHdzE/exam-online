# Gorilla Test Assistant

An automated assistant for TestGorilla/CodeSignal-style assessments, demonstrating the potential limitations of traditional testing in the age of AI. This tool is specifically designed for macOS users.

## Features

- Global hotkeys (F6/F7/F8/F9) monitored via a self-healing Quartz event tap — they keep working from any app and any desktop/Space, even after macOS disables the tap
- Full-screen capture with Pillow; OCR via Tesseract runs in a background worker and is parallelized across CPU cores
- Question parsing, answering, and solution generation via the DeepSeek API (screenshots are distilled to OCR text; irrelevant windows/UI chrome are filtered out at parse time)
- Generated solutions are executed and tested in a sandboxed temp directory; failed attempts loop with failure feedback (up to 4 attempts) until tests pass
- User directives (F9): free-form guidance notes that are attached to the next processing attempt, so you can steer the solver without editing code
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
   git clone https://github.com/albertoHdzE/exam-online.git
   cd exam-online
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

3. When you encounter a question:
   - Press `F6` to capture a screenshot from anywhere on your Mac
   - Press `F7` to process the current screenshot batch (waits for background OCR, then parses, solves, and self-tests)
   - Press `F9` to add a directive: the terminal prompts for a free-form note (e.g. "prefer an O(n log n) approach" or "the grader expects exactly two decimal places"). Pending directives are injected into the next `F7` processing attempt as guidance for the solver, then marked consumed. Directives steer approach, debugging, or emphasis; they do not override the problem statement.
   - Press `F8` to quit the listener

   Function keys are used so the hotkeys never collide with text you type into the exam interface. If your keyboard maps the F-keys to media functions, either press them together with `fn` or enable "Use F1, F2, etc. keys as standard function keys" in System Settings → Keyboard.

4. The final answer is placed on your clipboard automatically — paste it into the exam with Cmd+V instead of retyping it.

   Notifications adapt to the machine automatically: on Intel Macs and Apple Silicon with Rosetta 2 they use pync/terminal-notifier; on Apple Silicon without Rosetta (where pync's Intel-only binary cannot run) the app falls back to native AppleScript notifications. The selected backend is printed at startup and requires no configuration.

5. If the platform rejects an answer, capture the feedback screen with `F6` and press `F7` again: visible feedback (failed tests, grader messages) is parsed as task context for the next attempt. Combine this with an `F9` directive when you want to redirect the approach explicitly.

## Maintenance

- `./clean-env` resets the app to a pristine state (deletes the database, provenance log, screenshots, sandbox runs, and caches; asks for confirmation, `-y` skips it). State is recreated automatically on the next run.
- Runtime artifacts live in `screenshots/`, `data/`, and `temp_runs/` — all git-ignored.
- Verification helper scripts (require the venv): `temp_runs/hotkey_smoke_test.py`, `temp_runs/async_capture_test.py`, `temp_runs/iterative_loop_test.py`.

## Important Notes

- This tool is for educational and demonstration purposes only.
- Use responsibly and ethically.
- Be aware of the terms of service for any platforms you're using.
- This tool is designed specifically for macOS and may not work on other operating systems.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This project is meant to highlight the need for more advanced and relevant assessment methods in tech hiring. It should not be used to gain unfair advantages in actual assessments. It is specifically designed for macOS users and may not function on other operating systems.
