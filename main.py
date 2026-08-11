import os
import argparse
import json
import sys
import sqlite3
import base64
import threading
import termios
import shutil
import random
import tempfile
import resource
import queue
import time
import tty
import venv
import subprocess
import textwrap
import hashlib
import urllib.request
import platform
from concurrent.futures import ThreadPoolExecutor
from importlib import metadata as importlib_metadata
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Callable
from dataclasses import dataclass

import pynput
from PIL import Image, ImageGrab, ImageOps, ImageFilter
import pydantic
import pync
import pytesseract
import imagehash
from ollama import Client
from dotenv import load_dotenv
import requests
from pygments import highlight
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.formatters import TerminalFormatter
from pygments.style import Style
from pygments.token import Keyword, Name, Comment, String, Number, Operator, Generic

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PATH = (
    PROJECT_ROOT / ".venv"
    if (PROJECT_ROOT / ".venv").exists()
    else PROJECT_ROOT / "venv"
)
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "exam_online.db"
PROVENANCE_LOG = DATA_DIR / "provenance.jsonl"
TEMP_DIR = PROJECT_ROOT / "temp_runs"

SCREENSHOTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# Global hotkeys are function keys so they never collide with typing answers
# in the exam interface (plain letters were suppressed system-wide before).
# macOS virtual keycodes: F6=97, F7=98, F8=100.
HOTKEY_ACTION_KEYCODES = {97: "capture", 98: "process", 100: "quit"}
CAPTURE_LABEL = "F6"
PROCESS_LABEL = "F7"
QUIT_LABEL = "F8"

# Single-key controls used only in terminal control mode (no global monitor).
KEY_CAPTURE = "c"
KEY_PROCESS = "p"
KEY_QUIT = "q"

PLAY_LOCAL = False
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
USE_JSON_RESPONSE_FORMAT = True

CODE_TIMEOUT_SECONDS = 30
MAX_RETRIES_LLM = 2
MAX_SOLUTION_ATTEMPTS = 4
HOTKEY_DEBOUNCE_SECONDS = 0.35


def _debug_report(hypothesis_id: str, location: str, msg: str,
                  data: Optional[Dict[str, Any]] = None,
                  run_id: str = "pre-fix") -> None:
    # #region debug-point shared:reporter
    payload = {
        "sessionId": "screenshots-batch-parse",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "msg": f"[DEBUG] {msg}",
        "data": data or {},
        "ts": int(datetime.now(timezone.utc).timestamp() * 1000),
    }
    env_path = PROJECT_ROOT / ".dbg" / "screenshots-batch-parse.env"
    debug_server_url = "http://127.0.0.1:7777/event"
    try:
        if env_path.exists():
            env_lines = env_path.read_text(encoding="utf-8").splitlines()
            for line in env_lines:
                if line.startswith("DEBUG_SERVER_URL="):
                    debug_server_url = line.split("=", 1)[1].strip()
                    break
        request = urllib.request.Request(
            debug_server_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(request, timeout=2).read()
    except Exception:
        pass
    # #endregion


def _debug_hotkey_report(hypothesis_id: str, location: str, msg: str,
                         data: Optional[Dict[str, Any]] = None,
                         run_id: str = "pre-fix") -> None:
    # #region debug-point shared:hotkeys-reporter
    payload = {
        "sessionId": "hotkeys-no-capture",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "msg": f"[DEBUG] {msg}",
        "data": data or {},
        "ts": int(datetime.now(timezone.utc).timestamp() * 1000),
    }
    env_path = PROJECT_ROOT / ".dbg" / "hotkeys-no-capture.env"
    debug_server_url = "http://127.0.0.1:7777/event"
    try:
        if env_path.exists():
            env_lines = env_path.read_text(encoding="utf-8").splitlines()
            for line in env_lines:
                if line.startswith("DEBUG_SERVER_URL="):
                    debug_server_url = line.split("=", 1)[1].strip()
                    break
        request = urllib.request.Request(
            debug_server_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(request, timeout=2).read()
    except Exception:
        pass
    # #endregion


def should_use_terminal_control_mode() -> bool:
    requested = (os.getenv("FULLTEST_TERMINAL_CONTROL") or "").strip().lower()
    return requested in {"1", "true", "yes"}


class MacOSHotkeyMonitor:
    """Global hotkey monitor backed by a Quartz CGEventTap.

    Hotkeys are matched by macOS virtual keycode (F6/F7/F8), not by character,
    so they fire regardless of keyboard layout and never collide with typed
    text. Unlike pynput's macOS listener, this monitor re-enables the event
    tap whenever macOS disables it (kCGEventTapDisabledByTimeout /
    kCGEventTapDisabledByUserInput). macOS disables taps around Space/desktop
    switches and after slow callbacks, which silently killed global hotkeys
    until the app was restarted. The tap is also re-created if creation
    fails, so monitoring resumes as soon as the system allows it again.
    """

    def __init__(self, on_hotkey: Callable[[str], None]):
        self._on_hotkey = on_hotkey
        self._running = threading.Event()
        self._stopped = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._tap_lock = threading.Lock()
        self._tap: Optional[Any] = None

    def start(self) -> None:
        self._running.set()
        self._thread = threading.Thread(
            target=self._run, name="fulltest-hotkey-tap", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        self._stopped.wait(timeout=3)

    def _run(self) -> None:
        import Quartz  # type: ignore
        try:
            while self._running.is_set():
                tap = self._create_tap(Quartz)
                if tap is None:
                    # Tap creation can fail transiently (e.g. right after the
                    # trust prompt). Keep retrying so monitoring self-heals.
                    time.sleep(1.0)
                    continue
                with self._tap_lock:
                    self._tap = tap
                loop_source = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
                run_loop = Quartz.CFRunLoopGetCurrent()
                Quartz.CFRunLoopAddSource(
                    run_loop, loop_source, Quartz.kCFRunLoopDefaultMode)
                Quartz.CGEventTapEnable(tap, True)
                while self._running.is_set():
                    Quartz.CFRunLoopRunInMode(
                        Quartz.kCFRunLoopDefaultMode, 0.5, False)
                Quartz.CGEventTapEnable(tap, False)
                Quartz.CFRunLoopRemoveSource(
                    run_loop, loop_source, Quartz.kCFRunLoopDefaultMode)
                with self._tap_lock:
                    self._tap = None
        finally:
            self._stopped.set()

    def _create_tap(self, Quartz: Any) -> Optional[Any]:
        def tap_callback(proxy: Any, event_type: Any, event: Any, refcon: Any) -> Any:
            if event_type in (
                Quartz.kCGEventTapDisabledByTimeout,
                Quartz.kCGEventTapDisabledByUserInput,
            ):
                # macOS disabled the tap (Space switch, slow callback, secure
                # input, ...). Re-enable it so hotkeys keep working.
                with self._tap_lock:
                    tap = self._tap
                if tap is not None:
                    Quartz.CGEventTapEnable(tap, True)
                return None
            if event_type != Quartz.kCGEventKeyDown:
                return event
            action = self._event_action(Quartz, event)
            if action is not None:
                try:
                    self._on_hotkey(action)
                except Exception:
                    pass
                return None  # suppress hotkeys so focused apps never see them
            return event

        return Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap,
            Quartz.kCGHeadInsertEventTap,
            Quartz.kCGEventTapOptionDefault,
            Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown),
            tap_callback,
            None,
        )

    @staticmethod
    def _event_action(Quartz: Any, event: Any) -> Optional[str]:
        try:
            keycode = Quartz.CGEventGetIntegerValueField(
                event, Quartz.kCGKeyboardEventKeycode)
        except Exception:
            return None
        return HOTKEY_ACTION_KEYCODES.get(int(keycode))


def ensure_macos_input_monitoring_trust() -> None:
    if sys.platform != "darwin":
        return

    try:
        import HIServices  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "macOS global hotkeys require the PyObjC Quartz bridge. Install dependencies with "
            "`pip install -r requirements.txt` inside the project virtual environment."
        ) from exc

    try:
        is_trusted = bool(
            HIServices.AXIsProcessTrustedWithOptions(
                {HIServices.kAXTrustedCheckOptionPrompt: True}
            )
        )
    except Exception:
        is_trusted = bool(HIServices.AXIsProcessTrusted())

    if is_trusted:
        return

    terminal_program = os.getenv("TERM_PROGRAM") or "your terminal app"
    python_path = sys.executable
    raise RuntimeError(
        "macOS blocked global keyboard monitoring for this process. "
        "Approve the app in System Settings -> Privacy & Security -> Accessibility "
        f"(and Input Monitoring if shown), then relaunch the terminal and rerun the app.\n"
        f"Current terminal: {terminal_program}\n"
        f"Python executable: {python_path}"
    )


class QuestionState(Enum):
    CAPTURED = auto()
    PARSED = auto()
    ANSWERING = auto()
    TESTING = auto()
    VERIFIED = auto()
    PASSED = auto()
    FAILED = auto()
    ERROR_CAPTURE = auto()
    ERROR_PARSE = auto()
    ERROR_ANSWER = auto()
    ERROR_TEST = auto()

    @classmethod
    def terminal_states(cls):
        return {cls.VERIFIED, cls.PASSED, cls.FAILED,
                cls.ERROR_CAPTURE, cls.ERROR_PARSE,
                cls.ERROR_ANSWER, cls.ERROR_TEST}


class ProblemType(str, Enum):
    MULTIPLE_CHOICE = "multiple-choice"
    PROGRAMMING = "programming"
    MATH = "math"
    LOGIC = "logic"
    UNKNOWN = "unknown"


@dataclass
class CapturedImage:
    index: int
    file_path: Path
    timestamp: str
    phash: str
    ocr_text: str = ""
    is_duplicate: bool = False


@dataclass
class CodeVersion:
    version_number: int
    code: str
    bug_type: Optional[str] = None
    bug_description: Optional[str] = None
    tests_passed: Optional[bool] = None
    test_output: Optional[str] = None


class ExamStyle(Style):
    background_color = "#000000"
    styles = {
        Keyword: "bold #00aa00",
        Name.Builtin: "#00aaaa",
        Name.Function: "bold #0000aa",
        Name.Class: "bold #aa0000",
        String: "#aa5500",
        Number: "#009999",
        Operator: "bold #555555",
        Comment: "italic #555555",
        Generic.Error: "bold #ff0000",
        Generic.Emph: "italic",
        Generic.Strong: "bold",
    }


# =========================================================================
# Section 1: Virtual Environment Manager
# =========================================================================

class VenvManager:
    @staticmethod
    def ensure_venv() -> Path:
        python_exe = VENV_PATH / "bin" / "python"
        if not python_exe.exists():
            print("[VENV] Creating virtual environment...")
            venv.create(VENV_PATH, with_pip=True, clear=False)
            print("[VENV] Installing dependencies...")
            pip_exe = VENV_PATH / "bin" / "pip"
            req_file = PROJECT_ROOT / "requirements.txt"
            subprocess.run(
                [str(pip_exe), "install", "-r", str(req_file)],
                capture_output=True, text=True, check=False,
            )
        return python_exe

    @staticmethod
    def get_python_path() -> Path:
        return VENV_PATH / "bin" / "python"


def _parse_version_parts(version_text: str) -> Tuple[int, ...]:
    parts: List[int] = []
    for token in version_text.split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def validate_runtime_compatibility() -> None:
    try:
        pynput_version = importlib_metadata.version("pynput")
    except importlib_metadata.PackageNotFoundError:
        raise RuntimeError(
            "pynput is not installed. Run `pip install -r requirements.txt` in the project virtual environment."
        )

    if sys.version_info >= (3, 13) and _parse_version_parts(pynput_version) < (1, 8, 1):
        raise RuntimeError(
            "Installed pynput version is incompatible with Python 3.13 on macOS. "
            f"Detected pynput {pynput_version}. Upgrade dependencies with "
            "`pip install -r requirements.txt` in this project's virtual environment."
        )


def configure_ocr_runtime() -> str:
    configured_binary = shutil.which("tesseract")
    if configured_binary:
        pytesseract.pytesseract.tesseract_cmd = configured_binary
        return configured_binary

    candidate_paths = [
        "/opt/homebrew/bin/tesseract",
        "/usr/local/bin/tesseract",
    ]
    for candidate in candidate_paths:
        if Path(candidate).exists():
            pytesseract.pytesseract.tesseract_cmd = candidate
            return candidate

    raise RuntimeError(
        "Tesseract OCR binary is not installed. Install it with `brew install tesseract` "
        "and rerun the application."
    )


# =========================================================================
# Section 2: Provenance Ledger (Append-Only JSONL)
# =========================================================================

class ProvenanceLedger:
    @staticmethod
    def record(event_type: str, payload: Dict[str, Any]) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "payload": payload,
        }
        with open(PROVENANCE_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")


def copy_to_clipboard(text: str) -> bool:
    """Copy text to the macOS clipboard so answers can be pasted (Cmd+V)
    instead of retyped — hand-copying introduces typos that fail tests."""
    if sys.platform != "darwin" or not text:
        return False
    try:
        subprocess.run(["pbcopy"], input=text, text=True, check=True)
        ProvenanceLedger.record("clipboard_copy", {"chars": len(text)})
        return True
    except Exception as e:
        ProvenanceLedger.record("clipboard_copy_error", {"error": str(e)})
        return False


# -------------------------------------------------------------------------
# Notifications: adaptive backend selection per machine.
# pync's vendored terminal-notifier is an x86_64-only binary: it runs
# natively on Intel Macs and on Apple Silicon only when Rosetta 2 is
# installed. Everywhere else we fall back to AppleScript (osascript),
# which is native on every macOS.
# -------------------------------------------------------------------------
_NOTIFICATION_BACKEND: Optional[str] = None


def _detect_notification_backend() -> str:
    if sys.platform != "darwin":
        return "osascript"
    machine = platform.machine()
    if machine == "x86_64":
        return "pync"  # Intel Mac: vendored binary runs natively
    if machine == "arm64":
        try:
            probe = subprocess.run(
                ["/usr/bin/arch", "-x86_64", "/usr/bin/true"],
                capture_output=True, timeout=5)
            if probe.returncode == 0:
                return "pync"  # Apple Silicon with Rosetta 2 installed
        except Exception:
            pass
    return "osascript"


def init_notifications() -> str:
    """Select and remember the notification backend for this machine."""
    global _NOTIFICATION_BACKEND
    if _NOTIFICATION_BACKEND is None:
        _NOTIFICATION_BACKEND = _detect_notification_backend()
        ProvenanceLedger.record("notification_backend", {
            "backend": _NOTIFICATION_BACKEND,
            "machine": platform.machine(),
        })
    return _NOTIFICATION_BACKEND


def _osascript_notify(message: str, title: str) -> None:
    def esc(text: str) -> str:
        return text.replace("\\", "\\\\").replace('"', '\\"')

    script = f'display notification "{esc(message)}" with title "{esc(title)}"'
    try:
        subprocess.run(["osascript", "-e", script],
                       capture_output=True, timeout=5, check=False)
    except Exception as e:
        ProvenanceLedger.record("notification_error", {"error": str(e)})


def notify_user(message: str, title: str) -> None:
    global _NOTIFICATION_BACKEND
    backend = init_notifications()
    if backend == "pync":
        try:
            pync.notify(message, title=title)
            return
        except Exception as e:
            # e.g. Rosetta removed mid-session or a broken vendored binary
            ProvenanceLedger.record("notification_fallback", {
                "from": "pync", "to": "osascript", "error": str(e)})
            _NOTIFICATION_BACKEND = "osascript"
    _osascript_notify(message, title)


# =========================================================================
# Section 3: SQLite Database Layer
# =========================================================================

class Database:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                problem_type TEXT,
                raw_question_text TEXT,
                full_question TEXT,
                state TEXT NOT NULL,
                proposed_answer TEXT,
                correctness TEXT,
                status_note TEXT,
                screenshots_json TEXT,
                metadata_json TEXT
            );

            CREATE TABLE IF NOT EXISTS code_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id INTEGER NOT NULL,
                version_number INTEGER NOT NULL,
                code TEXT NOT NULL,
                bug_type TEXT,
                bug_description TEXT,
                tests_passed INTEGER,
                test_output TEXT,
                FOREIGN KEY (question_id) REFERENCES questions(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS extracted_tests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id INTEGER NOT NULL,
                test_source TEXT NOT NULL,
                test_code TEXT NOT NULL,
                FOREIGN KEY (question_id) REFERENCES questions(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS generated_artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id INTEGER NOT NULL,
                artifact_name TEXT NOT NULL,
                artifact_type TEXT NOT NULL,
                artifact_origin TEXT NOT NULL,
                content TEXT NOT NULL,
                FOREIGN KEY (question_id) REFERENCES questions(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS clarification_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id INTEGER NOT NULL,
                question_text TEXT NOT NULL,
                answer_text TEXT,
                status TEXT NOT NULL,
                FOREIGN KEY (question_id) REFERENCES questions(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_questions_session ON questions(session_id);
            CREATE INDEX IF NOT EXISTS idx_questions_state ON questions(state);
            """)

    def create_question(self, session_id: str, screenshots_json: str) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO questions
                (session_id, created_at, state, screenshots_json, correctness)
                VALUES (?, ?, ?, ?, ?)""",
                (session_id, datetime.now(timezone.utc).isoformat(),
                 QuestionState.CAPTURED.name, screenshots_json, "UNCHECKED"),
            )
            qid = cur.lastrowid
            ProvenanceLedger.record("question_created",
                                    {"question_id": qid, "session_id": session_id})
            return qid

    def update_question(self, question_id: int, **fields) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k} = ?" for k in fields.keys())
        vals = list(fields.values()) + [question_id]
        with self._connect() as conn:
            conn.execute(f"UPDATE questions SET {cols} WHERE id = ?", vals)
        ProvenanceLedger.record("question_updated",
                                {"question_id": question_id, "fields": list(fields.keys())})

    def set_state(self, question_id: int, new_state: QuestionState) -> None:
        self.update_question(question_id, state=new_state.name)
        ProvenanceLedger.record("state_transition",
                                {"question_id": question_id, "new_state": new_state.name})

    def add_code_version(self, question_id: int, cv: CodeVersion) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO code_versions
                (question_id, version_number, code, bug_type,
                 bug_description, tests_passed, test_output)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (question_id, cv.version_number, cv.code, cv.bug_type,
                 cv.bug_description,
                 1 if cv.tests_passed else 0 if cv.tests_passed is not None else None,
                 cv.test_output),
            )
            return cur.lastrowid

    def add_extracted_test(self, question_id: int, source: str, code: str) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO extracted_tests (question_id, test_source, test_code) VALUES (?, ?, ?)",
                (question_id, source, code),
            )
            return cur.lastrowid

    def add_generated_artifact(self, question_id: int, artifact_name: str,
                               artifact_type: str, artifact_origin: str,
                               content: str) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO generated_artifacts
                (question_id, artifact_name, artifact_type, artifact_origin, content)
                VALUES (?, ?, ?, ?, ?)""",
                (question_id, artifact_name, artifact_type, artifact_origin, content),
            )
            return cur.lastrowid

    def add_clarification_item(self, question_id: int, question_text: str,
                               answer_text: Optional[str], status: str) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO clarification_items
                (question_id, question_text, answer_text, status)
                VALUES (?, ?, ?, ?)""",
                (question_id, question_text, answer_text, status),
            )
            return cur.lastrowid

    def get_question(self, question_id: int) -> Optional[sqlite3.Row]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM questions WHERE id = ?", (question_id,)
            ).fetchone()
            return row

    def get_code_versions(self, question_id: int) -> List[sqlite3.Row]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM code_versions WHERE question_id = ? ORDER BY version_number",
                (question_id,),
            ).fetchall()
            return list(rows)


# =========================================================================
# Section 4: Pydantic Response Models
# =========================================================================

class ParsedQuestion(pydantic.BaseModel):
    problem_type: ProblemType
    full_question: str
    is_single_answer: Optional[bool] = None
    is_multiple_answer: Optional[bool] = None
    mc_options: Optional[List[str]] = None
    extracted_tests: Optional[List[str]] = None
    programming_language: Optional[str] = None
    required_function_name: Optional[str] = None
    required_output_format: Optional[str] = None
    likely_assessment_family: Optional[str] = None
    likely_progressive_assessment: Optional[bool] = None
    progressive_requirements: Optional[List[str]] = None
    design_requirements: Optional[List[str]] = None
    requires_stateful_solution: Optional[bool] = None
    standard_library_only: Optional[bool] = None
    likely_domain_entities: Optional[List[str]] = None
    question_contract_known: Optional[bool] = None
    requires_clarification: Optional[bool] = None
    clarification_questions: Optional[List[str]] = None
    visible_interfaces: Optional[List[str]] = None
    required_runtime_artifacts: Optional[List[str]] = None
    artifact_generation_strategy: Optional[str] = None
    conservative_assumptions: Optional[List[str]] = None


class MultipleChoiceAnswer(pydantic.BaseModel):
    explanation_of_question: str
    reasoning: str
    is_single_answer: bool
    is_multiple_answer: bool
    answer: List[int]


class SingleSolutionCode(pydantic.BaseModel):
    solution_code: str
    explanation: str
    programming_language: str
    suggested_test_code: Optional[str] = None
    architecture_notes: Optional[str] = None


class ArtifactFileSpec(pydantic.BaseModel):
    file_name: str
    content: str
    purpose: str


class GeneratedArtifacts(pydantic.BaseModel):
    suggested_test_code: Optional[str] = None
    artifact_files: Optional[List[ArtifactFileSpec]] = None
    notes: Optional[str] = None


class AdaptiveArtifactBuilder:
    def __init__(self, llm: "LLMClient", db: Database):
        self.llm = llm
        self.db = db

    @staticmethod
    def build_test_module_from_extracted(parsed: ParsedQuestion) -> str:
        lines = [
            "import sys",
            "import os",
            "sys.path.insert(0, os.path.dirname(__file__))",
            "from solution import *",
            "",
            "# Auto-generated from tests/examples embedded in question",
        ]
        for i, test_line in enumerate(parsed.extracted_tests or []):
            clean = test_line.strip()
            if clean.startswith("assert"):
                lines.append(f"def test_extracted_{i}():")
                lines.append(f"    {clean}")
            else:
                lines.append(f"# from question: {clean[:200]}")
        lines.append("")
        return "\n".join(lines)

    def build(self, question_id: int, parsed: ParsedQuestion,
              images: List[CapturedImage], solution_code: str) -> Tuple[Optional[str], Dict[str, str], str]:
        if parsed.extracted_tests:
            test_code = self.build_test_module_from_extracted(parsed)
            self.db.add_generated_artifact(
                question_id, "test_solution.py", "pytest-module",
                "extracted-tests", test_code
            )
            return test_code, {}, "Built tests from extracted visible assertions/examples"

        should_generate_adaptive_artifacts = bool(
            parsed.required_runtime_artifacts
            or parsed.visible_interfaces
            or parsed.requires_stateful_solution
            or parsed.requires_clarification
            or parsed.likely_assessment_family in {"banking-ledger", "key-value-store"}
        )

        if should_generate_adaptive_artifacts:
            generated = self.llm.generate_adaptive_artifacts(parsed, images, solution_code)
            auxiliary_files: Dict[str, str] = {}
            if generated.suggested_test_code:
                self.db.add_generated_artifact(
                    question_id, "test_solution.py", "pytest-module",
                    "adaptive-llm", generated.suggested_test_code
                )
            for artifact in generated.artifact_files or []:
                auxiliary_files[artifact.file_name] = artifact.content
                self.db.add_generated_artifact(
                    question_id, artifact.file_name, "runtime-artifact",
                    "adaptive-llm", artifact.content
                )
            return generated.suggested_test_code, auxiliary_files, (
                generated.notes or "Generated adaptive runtime artifacts"
            )

        test_code = self.llm.generate_supplementary_tests(parsed, images, solution_code)
        self.db.add_generated_artifact(
            question_id, "test_solution.py", "pytest-module",
            "supplementary-llm", test_code
        )
        return test_code, {}, "Generated supplementary pytest verification only"


# =========================================================================
# Section 5: Screenshot Capture & Storage
# =========================================================================

class ScreenshotManager:
    def __init__(self):
        self._session_counter: Dict[str, int] = {}

    @staticmethod
    def _score_ocr_text(text: str) -> int:
        alnum_count = sum(1 for char in text if char.isalnum())
        line_count = len([line for line in text.splitlines() if line.strip()])
        return alnum_count + (10 * line_count)

    @staticmethod
    def _extract_ocr_text(image) -> str:
        width, height = image.size
        prompt_panel_left = image.crop((220, 90, min(width, 1260), height - 40))
        prompt_text_column = image.crop((260, 110, min(width, 900), height - 60))
        crops = [image, prompt_panel_left, prompt_text_column]
        configs = [
            "--psm 6",
            "--psm 11",
            "--psm 4",
            "--psm 3",
        ]
        jobs: List[Tuple[Any, str]] = []
        for crop in crops:
            grayscale = ImageOps.autocontrast(ImageOps.grayscale(crop))
            enlarged = grayscale.resize(
                (max(1, grayscale.width * 2), max(1, grayscale.height * 2))
            )
            variants = [
                crop,
                grayscale,
                grayscale.filter(ImageFilter.SHARPEN),
                ImageOps.invert(grayscale),
                enlarged,
                ImageOps.invert(enlarged),
                ImageOps.posterize(grayscale.convert("RGB"), 2),
            ]
            for variant in variants:
                for config in configs:
                    jobs.append((variant, config))

        def run_ocr_job(job: Tuple[Any, str]) -> str:
            variant, config = job
            try:
                return pytesseract.image_to_string(variant, config=config) or ""
            except Exception:
                return ""

        # Tesseract runs as a subprocess, so parallel workers scale across CPU
        # cores. map() preserves job order, keeping candidate selection stable.
        max_workers = min(8, (os.cpu_count() or 4))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            ocr_candidates = [text for text in pool.map(run_ocr_job, jobs) if text]

        if not ocr_candidates:
            # #region debug-point A:ocr-empty
            _debug_report(
                "A",
                "main.py:541",
                "OCR produced no candidates",
                {"image_size": getattr(image, "size", None)},
            )
            # #endregion
            return ""
        best_text = max(ocr_candidates, key=ScreenshotManager._score_ocr_text).strip()
        # #region debug-point A:ocr-best
        _debug_report(
            "A",
            "main.py:549",
            "OCR selected best candidate",
            {
                "candidate_count": len(ocr_candidates),
                "best_score": ScreenshotManager._score_ocr_text(best_text),
                "preview": best_text[:200],
            },
        )
        # #endregion
        return best_text

    @staticmethod
    def _normalize_ocr_text(text: str) -> str:
        return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in text).split())

    def _get_today_prefix(self) -> str:
        return datetime.now().strftime("%Y-%m-%d")

    def _next_index_for_today(self) -> int:
        prefix = self._get_today_prefix()
        existing = list(SCREENSHOTS_DIR.glob(f"{prefix}-*.png"))
        indices = []
        for p in existing:
            try:
                idx = int(p.stem.split("-")[-1])
                indices.append(idx)
            except (ValueError, IndexError):
                continue
        return max(indices) + 1 if indices else 1

    def capture(self, extract_text: bool = True) -> CapturedImage:
        idx = self._next_index_for_today()
        prefix = self._get_today_prefix()
        file_name = f"{prefix}-{idx}.png"
        file_path = SCREENSHOTS_DIR / file_name
        image = ImageGrab.grab()
        image.save(file_path, format="PNG")
        phash = str(imagehash.phash(image))
        # OCR is expensive (dozens of tesseract passes). Callers can defer it
        # to the background OCR worker so the capture hotkey stays responsive.
        ocr_text = self._extract_ocr_text(image) if extract_text else ""
        captured = CapturedImage(
            index=idx,
            file_path=file_path,
            timestamp=datetime.now(timezone.utc).isoformat(),
            phash=phash,
            ocr_text=ocr_text,
        )
        ProvenanceLedger.record("screenshot_captured", {
            "file": str(file_path),
            "index": idx,
            "phash": phash,
            "ocr_len": len(ocr_text),
            "ocr_preview": ocr_text[:200],
            "ocr_deferred": not extract_text,
        })
        return captured

    @staticmethod
    def load_existing_images(directory: Path) -> List[CapturedImage]:
        files = sorted(directory.glob("*.png"))
        loaded_images: List[CapturedImage] = []
        for idx, file_path in enumerate(files, start=1):
            image = Image.open(file_path)
            loaded_images.append(CapturedImage(
                index=idx,
                file_path=file_path,
                timestamp=datetime.now(timezone.utc).isoformat(),
                phash=str(imagehash.phash(image)),
                ocr_text=ScreenshotManager._extract_ocr_text(image),
            ))
        return loaded_images

    @staticmethod
    def mark_duplicates(images: List[CapturedImage],
                        hamming_threshold: int = 1) -> List[CapturedImage]:
        seen: List[Tuple[imagehash.ImageHash, str]] = []
        for img in images:
            try:
                current = imagehash.hex_to_hash(img.phash)
            except Exception:
                img.is_duplicate = False
                continue
            current_text = ScreenshotManager._normalize_ocr_text(img.ocr_text)
            is_dup = False
            for prev_hash, prev_text in seen:
                hash_distance = current - prev_hash
                if hash_distance == 0:
                    is_dup = True
                    break
                if current_text and prev_text and hash_distance <= hamming_threshold and current_text == prev_text:
                    is_dup = True
                    break
            img.is_duplicate = is_dup
            # #region debug-point D:duplicate-decision
            _debug_report(
                "D",
                "main.py:587",
                "Duplicate decision computed",
                {
                    "file": str(img.file_path),
                    "is_duplicate": is_dup,
                    "ocr_preview": img.ocr_text[:160],
                },
            )
            # #endregion
            if not is_dup:
                seen.append((current, current_text))
        return images


# =========================================================================
# Section 6: LLM Integration (Multi-Image, Dual Backend)
# =========================================================================

class LLMClient:
    def __init__(self):
        self.local_client = Client(host="http://localhost:11434")

    @staticmethod
    def _programming_context_block(parsed: ParsedQuestion) -> str:
        context_lines = [
            f"Likely assessment family: {parsed.likely_assessment_family or 'general'}",
            f"Progressive assessment: {parsed.likely_progressive_assessment}",
            f"Requires stateful solution: {parsed.requires_stateful_solution}",
            f"Standard library only: {parsed.standard_library_only}",
        ]
        if parsed.progressive_requirements:
            context_lines.append(
                "Observed or inferred progressive requirements: "
                + "; ".join(parsed.progressive_requirements)
            )
        if parsed.design_requirements:
            context_lines.append(
                "Design requirements: " + "; ".join(parsed.design_requirements)
            )
        if parsed.likely_domain_entities:
            context_lines.append(
                "Likely domain entities: " + ", ".join(parsed.likely_domain_entities)
            )
        return "\n".join(context_lines)

    @staticmethod
    def image_to_base64(path: Path) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def _merge_ocr_texts(images: List[CapturedImage]) -> str:
        parts = []
        for img in images:
            if not img.is_duplicate:
                header = f"[IMAGE-{img.index} OCR]"
                parts.append(header)
                parts.append(img.ocr_text.strip() or "(no text detected)")
        return "\n".join(parts)

    def _llm_local_multi(self, prompt: str,
                         images: List[CapturedImage],
                         expect_json: bool = True) -> str:
        b64_list = [
            self.image_to_base64(img.file_path)
            for img in images if not img.is_duplicate
        ]
        response = self.local_client.generate(
            model="deepseek-coder-v2",
            prompt=prompt,
            images=b64_list if b64_list else None,
            format="json" if expect_json else "",
            options={"temperature": 0.2, "num_ctx": 16384},
        )
        ProvenanceLedger.record("llm_local_call", {
            "prompt_len": len(prompt),
            "num_images": len(b64_list),
            "output_len": len(response.response),
        })
        return response.response

    def _llm_online(self, prompt: str, extracted_text: str,
                    expect_json: bool = True) -> str:
        if not DEEPSEEK_API_KEY:
            raise ValueError("DEEPSEEK_API_KEY not set in env/.env")
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }
        full_prompt = f"{prompt}\n\n=== EXTRACTED TEXT FROM SCREENSHOTS ===\n{extracted_text}"
        payload: Dict[str, Any] = {
            "model": DEEPSEEK_MODEL,
            "messages": [{"role": "user", "content": full_prompt}],
            "temperature": 0.2,
        }
        if expect_json and USE_JSON_RESPONSE_FORMAT:
            payload["response_format"] = {"type": "json_object"}
        try:
            resp = requests.post(DEEPSEEK_API_URL, headers=headers,
                                 json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            if isinstance(content, str) and expect_json:
                try:
                    content = json.loads(content)
                    content = json.dumps(content)
                except json.JSONDecodeError:
                    pass
            ProvenanceLedger.record("llm_online_call", {
                "model": DEEPSEEK_MODEL,
                "prompt_len": len(full_prompt),
                "output_len": len(str(content)),
            })
            return content if isinstance(content, str) else json.dumps(content)
        except Exception as e:
            ProvenanceLedger.record("llm_online_error", {"error": str(e)})
            raise

    def parse_question(self, images: List[CapturedImage]) -> ParsedQuestion:
        merged_ocr = self._merge_ocr_texts(images)
        # #region debug-point C:merged-ocr
        _debug_report(
            "C",
            "main.py:786",
            "Prepared merged OCR for parse",
            {
                "unique_images": len([img for img in images if not img.is_duplicate]),
                "ocr_length": len(merged_ocr),
                "ocr_preview": merged_ocr[:300],
            },
        )
        # #endregion
        schema_str = ParsedQuestion.model_json_schema()
        prompt = f"""You are a precise exam question parser. The user has captured multiple screenshots of a single question.
Each screenshot is a FULL macOS desktop capture: the OCR text mixes the exam/assessment window with unrelated
content (browser chrome and tabs, terminal or IDE windows, menu bar items, notifications, messaging apps, bookmarks).
Extract the following fields into valid JSON matching this JSON Schema:
{json.dumps(schema_str, indent=2)}

Critical rules:
0. FOCUS FIRST: locate the exam/assessment content (e.g. a TestGorilla/CodeSignal-style question, a coding task,
   an answer form, or FEEDBACK about a previous submission such as failed tests or grader messages). Everything
   else — window titles, menus, toolbars, URLs, tab names, sidebar text, chat messages — is NOISE: ignore it
   completely and never include it in any field. If submission feedback is visible, treat it as part of the task
   context and include it in full_question (the user is retrying after a failed attempt).
1. problem_type: must be one of [multiple-choice, programming, math, logic, unknown]
2. full_question: reassemble the entire question text in correct reading order across images, containing ONLY the
   assessment content (plus visible feedback, if any). Use [IMAGE-N] markers only if content is explicitly split.
3. For programming: extract programming_language, required_function_name (if stated), and any embedded test assertions or example input/output pairs as extracted_tests (list of strings).
3a. OCR VALIDATION (mandatory): the OCR text may contain recognition errors, especially in expected-output lists
   (dropped or merged tokens like "4" -> ""). For EVERY example input/output pair, mentally simulate the described
   operations step by step. First COUNT: the expected-output list MUST contain exactly one element per query —
   if the counts differ, the OCR text is corrupted; reconstruct the list from your simulation. Then verify each
   expected value against your simulation; if one contradicts it, trust the simulation, correct the value, and
   note the correction in full_question. Likewise, if an operation's return value is not clearly described,
   INFER it from the example outputs (e.g. an example showing "true"/"false" for a REMOVE operation means REMOVE
   returns "true"/"false") and state the inferred contract explicitly in full_question. Never copy an expected
   output that violates the operation semantics.
4. For multiple-choice: list all mc_options verbatim, and set is_single_answer / is_multiple_answer accordingly.
5. For programming questions, infer whether this resembles a progressive coding assessment where later levels extend the initial design. Common families include:
   - banking ledger / accounts / scheduled transactions / interest / audits
   - in-memory key-value database / scans / prefix search / range query / TTL / backup / rollback
6. Fill the architecture-related fields carefully:
   - likely_assessment_family: e.g. "banking-ledger", "key-value-store", or "general-programming"
   - likely_progressive_assessment: true when the exercise appears multi-level or explicitly staged
   - progressive_requirements: visible or strongly implied later extensions such as transfers, audits, scheduling, TTL, rollback
   - design_requirements: phrases like modular, maintainable, helper methods, dataclass-friendly, class-based, maps/lists
   - requires_stateful_solution: true when the problem maintains evolving state across operations
   - standard_library_only: true unless the prompt explicitly permits extra packages
   - likely_domain_entities: likely classes or records such as Account, Transaction, ScheduledPayment, Record, Entry
7. Also evaluate ambiguity and contract visibility:
   - question_contract_known: true only if the visible prompt defines enough behavior to implement and test safely
   - requires_clarification: true when the task refers to hidden state, hidden data, unclear APIs, ambiguous semantics, or missing examples
   - clarification_questions: 1-4 concrete questions that would materially reduce ambiguity
   - visible_interfaces: visible functions, methods, commands, or contracts explicitly shown in the prompt
   - required_runtime_artifacts: artifacts needed to test or simulate the visible contract, e.g. helper fixtures, fake API module, operation-sequence tests, temporary state records
   - artifact_generation_strategy: one short sentence describing what should be generated on the fly
   - conservative_assumptions: safe fallback assumptions if clarifications remain unanswered
8. Return ONLY valid JSON, no markdown fences, no commentary.

Images: the user provided {len([i for i in images if not i.is_duplicate])} unique screenshots.
"""
        raw = ""
        last_error = None
        for attempt in range(MAX_RETRIES_LLM + 1):
            try:
                if PLAY_LOCAL:
                    raw = self._llm_local_multi(prompt, images, expect_json=True)
                else:
                    raw = self._llm_online(prompt, merged_ocr, expect_json=True)
                if not isinstance(raw, str):
                    raw = json.dumps(raw)
                obj = json.loads(raw)
                parsed = ParsedQuestion.model_validate(obj)
                # #region debug-point E:parse-result
                _debug_report(
                    "E",
                    "main.py:834",
                    "Parse stage returned structured question",
                    {
                        "problem_type": parsed.problem_type.value,
                        "language": parsed.programming_language,
                        "required_function_name": parsed.required_function_name,
                        "full_question_preview": (parsed.full_question or "")[:300],
                    },
                )
                # #endregion
                return parsed
            except Exception as e:
                last_error = e
                # #region debug-point C:parse-retry
                _debug_report(
                    "C",
                    "main.py:847",
                    "Parse attempt failed",
                    {
                        "attempt": attempt,
                        "error": str(e),
                        "raw_preview": str(raw)[:300],
                    },
                )
                # #endregion
                ProvenanceLedger.record("parse_retry", {
                    "attempt": attempt,
                    "error": str(e),
                    "raw_preview": str(raw)[:500],
                })
                continue
        raise RuntimeError(f"Failed to parse question after retries: {last_error}")

    def answer_multiple_choice(self, parsed: ParsedQuestion,
                               images: List[CapturedImage]) -> MultipleChoiceAnswer:
        merged_ocr = self._merge_ocr_texts(images)
        prompt = f"""Solve the following multiple-choice exam question.
Return ONLY valid JSON matching this shape:
{{"explanation_of_question": str, "reasoning": str, "is_single_answer": bool, "is_multiple_answer": bool, "answer": [int, ...]}}

Rules:
- answer indices MUST be 1-based integers (the first option is 1, second is 2, etc.)
- answer must be a sorted list
- if single answer, list has length 1 and is_single_answer=true, is_multiple_answer=false
- if multiple answers allowed, is_single_answer=false, is_multiple_answer=true

Full question:
{parsed.full_question}

Options:
{json.dumps(parsed.mc_options or [])}
"""
        raw = ""
        for attempt in range(MAX_RETRIES_LLM + 1):
            try:
                if PLAY_LOCAL:
                    raw = self._llm_local_multi(prompt, images, expect_json=True)
                else:
                    raw = self._llm_online(prompt, merged_ocr, expect_json=True)
                obj = json.loads(raw) if isinstance(raw, str) else raw
                return MultipleChoiceAnswer.model_validate(obj)
            except Exception as e:
                ProvenanceLedger.record("mc_answer_retry", {
                    "attempt": attempt, "error": str(e)})
                continue
        raise RuntimeError(f"Failed to solve multiple-choice: {raw}")

    def generate_solution(self, parsed: ParsedQuestion,
                          images: List[CapturedImage]) -> SingleSolutionCode:
        merged_ocr = self._merge_ocr_texts(images)
        architecture_context = self._programming_context_block(parsed)
        extracted_tests_block = ""
        if parsed.extracted_tests:
            extracted_tests_block = (
                "The question contains these embedded tests or example assertions. "
                "The solution_code MUST pass ALL of them:\n"
                + "\n".join(f"  - {t}" for t in parsed.extracted_tests)
            )
        lang = parsed.programming_language or "Python"
        prompt = f"""You are a careful code-writing assistant for an exam system. The user asks you to produce
ONE correct solution to a programming problem. STRICTLY follow these instructions:

LANGUAGE: {lang}
REQUIRED FUNCTION NAME (if any): {parsed.required_function_name or "(any valid name)"}
REQUIRED OUTPUT FORMAT (if any): {parsed.required_output_format or "(standard)"}

ASSESSMENT CONTEXT:
{architecture_context}

{extracted_tests_block}

PROBLEM DESCRIPTION:
{parsed.full_question}

Return ONLY valid JSON matching this exact schema with no other text:
{{
  "solution_code": "the fully correct solution",
  "explanation": "explanation of approach",
  "programming_language": "{lang}",
  "suggested_test_code": "optional string containing a pytest-compatible test module exercising the solution thoroughly",
  "architecture_notes": "short note about the design choices that keep the solution extensible"
}}

Important:
- variable names must be DESCRIPTIVE but not excessively long (readable on a printed page).
- split logic into several small steps rather than a single one-liner.
- NEVER optimize prematurely: prefer clarity over cleverness.
- Assume this may be a CodeSignal-style progressive assessment. Even if only Level 1 is visible, structure solution_code so later extensions can be added with minimal refactoring.
- Prefer standard-library Python only unless the question explicitly permits dependencies.
- For stateful problems, prefer a small class-based design with clear helper methods and, when natural, dataclasses for records.
- Prefer dictionaries, lists, and straightforward control flow over clever abstractions.
- If the domain resembles a banking ledger or in-memory database, design for future operations such as transfers, scans, TTL, audits, scheduling, or rollback even if they are not yet implemented.
- suggested_test_code should normally be present. It must be pytest-compatible, self-contained, and verify correctness plus a few edge cases using only the visible requirements.
- Make absolutely sure solution_code is correct and compiles/runs first try.
"""
        raw = ""
        for attempt in range(MAX_RETRIES_LLM + 1):
            try:
                if PLAY_LOCAL:
                    raw = self._llm_local_multi(prompt, images, expect_json=True)
                else:
                    raw = self._llm_online(prompt, merged_ocr, expect_json=True)
                obj = json.loads(raw) if isinstance(raw, str) else raw
                return SingleSolutionCode.model_validate(obj)
            except Exception as e:
                ProvenanceLedger.record("solution_gen_retry", {
                    "attempt": attempt, "error": str(e),
                    "raw_preview": str(raw)[:800]})
                continue
        raise RuntimeError(f"Failed to generate solution after retries. Last preview: {str(raw)[:1200]}")

    def generate_supplementary_tests(self, parsed: ParsedQuestion,
                                     images: List[CapturedImage],
                                     solution_code: str) -> str:
        merged_ocr = self._merge_ocr_texts(images)
        architecture_context = self._programming_context_block(parsed)
        prompt = f"""Generate a pytest-compatible test module for the following programming question and solution.
Return ONLY valid JSON in the form:
{{"suggested_test_code": "full pytest module as a string"}}

Requirements:
- Use only Python standard library plus pytest.
- Prefer visible requirements and examples from the prompt over invented behavior.
- Cover baseline behavior and a few edge cases.
- If the task is stateful, test the main lifecycle operations in sequence.
- If the task resembles a banking ledger or key-value store, favor operation-sequence tests that verify evolving state.

ASSESSMENT CONTEXT:
{architecture_context}

QUESTION:
{parsed.full_question}

SOLUTION CODE:
{solution_code}
"""
        raw = ""
        for attempt in range(MAX_RETRIES_LLM + 1):
            try:
                if PLAY_LOCAL:
                    raw = self._llm_local_multi(prompt, images, expect_json=True)
                else:
                    raw = self._llm_online(prompt, merged_ocr, expect_json=True)
                obj = json.loads(raw) if isinstance(raw, str) else raw
                test_code = obj.get("suggested_test_code")
                if not test_code:
                    raise ValueError("LLM did not return suggested_test_code")
                return test_code
            except Exception as e:
                ProvenanceLedger.record("supplementary_test_retry", {
                    "attempt": attempt,
                    "error": str(e),
                    "raw_preview": str(raw)[:800],
                })
                continue
        raise RuntimeError("Failed to generate supplementary tests")

    def generate_adaptive_artifacts(self, parsed: ParsedQuestion,
                                    images: List[CapturedImage],
                                    solution_code: str) -> GeneratedArtifacts:
        merged_ocr = self._merge_ocr_texts(images)
        architecture_context = self._programming_context_block(parsed)
        prompt = f"""Generate adaptive runtime artifacts for verifying a programming solution.
Return ONLY valid JSON matching this schema:
{{
  "suggested_test_code": "pytest-compatible test module string, optional but usually present",
  "artifact_files": [
    {{
      "file_name": "relative/path/to/file.py",
      "content": "file contents",
      "purpose": "what this artifact does"
    }}
  ],
  "notes": "short explanation of generated artifacts"
}}

Rules:
- Generate artifacts only when justified by the visible contract.
- Do not invent hidden behavior that the prompt does not support.
- If the prompt references hidden state or invisible data, build a minimal contract-respecting stub or fixture only around what is explicitly exposed.
- Prefer operation-sequence tests for stateful domains like banking or key-value stores.
- Use only Python standard library plus pytest.
- Keep artifacts small, readable, and temporary.

ASSESSMENT CONTEXT:
{architecture_context}

VISIBLE INTERFACES:
{json.dumps(parsed.visible_interfaces or [])}

REQUIRED RUNTIME ARTIFACTS:
{json.dumps(parsed.required_runtime_artifacts or [])}

CLARIFICATION QUESTIONS:
{json.dumps(parsed.clarification_questions or [])}

CONSERVATIVE ASSUMPTIONS:
{json.dumps(parsed.conservative_assumptions or [])}

QUESTION:
{parsed.full_question}

SOLUTION CODE:
{solution_code}
"""
        raw = ""
        for attempt in range(MAX_RETRIES_LLM + 1):
            try:
                if PLAY_LOCAL:
                    raw = self._llm_local_multi(prompt, images, expect_json=True)
                else:
                    raw = self._llm_online(prompt, merged_ocr, expect_json=True)
                obj = json.loads(raw) if isinstance(raw, str) else raw
                return GeneratedArtifacts.model_validate(obj)
            except Exception as e:
                ProvenanceLedger.record("adaptive_artifact_retry", {
                    "attempt": attempt,
                    "error": str(e),
                    "raw_preview": str(raw)[:800],
                })
                continue
        raise RuntimeError("Failed to generate adaptive artifacts")


# =========================================================================
# Section 7: Isolated Code Executor (Sandbox)
# =========================================================================

class CodeExecutor:
    @staticmethod
    def _normalize_python_test_code(test_code: str) -> str:
        stripped = test_code.lstrip()
        has_solution_import = (
            "from solution import" in test_code
            or "import solution" in test_code
        )
        if has_solution_import:
            return test_code

        preamble = "\n".join([
            "import os",
            "import sys",
            "sys.path.insert(0, os.path.dirname(__file__))",
            "from solution import *",
            "",
        ])
        return preamble + stripped

    @staticmethod
    def _prepare_script(code: str, test_code: Optional[str],
                        auxiliary_files: Optional[Dict[str, str]] = None) -> Tuple[Path, Path]:
        tmpdir = Path(tempfile.mkdtemp(prefix="exam_run_", dir=str(TEMP_DIR)))
        script = tmpdir / "solution.py"
        script.write_text(code, encoding="utf-8")
        for file_name, content in (auxiliary_files or {}).items():
            relative_path = Path(file_name)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                continue
            destination = tmpdir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")
        if test_code:
            test_file = tmpdir / "test_solution.py"
            test_file.write_text(
                CodeExecutor._normalize_python_test_code(test_code),
                encoding="utf-8",
            )
            return script, test_file
        return script, tmpdir / "_no_test_"

    @staticmethod
    def _set_limits() -> None:
        try:
            resource.setrlimit(resource.RLIMIT_CPU,
                               (CODE_TIMEOUT_SECONDS, CODE_TIMEOUT_SECONDS * 2))
            resource.setrlimit(resource.RLIMIT_AS,
                               (2 * 1024 * 1024 * 1024, 4 * 1024 * 1024 * 1024))
        except Exception:
            pass

    @classmethod
    def run_python(cls, code: str,
                   test_code: Optional[str] = None,
                   auxiliary_files: Optional[Dict[str, str]] = None) -> Tuple[bool, str]:
        script_path, test_path = cls._prepare_script(code, test_code, auxiliary_files)
        tmpdir = script_path.parent
        python_exe = str(VenvManager.get_python_path())
        run_ok = True
        combined_output_parts: List[str] = []

        try:
            compile(code, str(script_path), "exec")
            combined_output_parts.append("[compile-check] syntax OK")
        except SyntaxError as se:
            return False, f"[compile-check] SYNTAX ERROR at line {se.lineno}: {se.msg}\n{se.text}"

        try:
            run = subprocess.run(
                [python_exe, str(script_path)],
                cwd=str(tmpdir),
                capture_output=True, text=True,
                timeout=CODE_TIMEOUT_SECONDS,
                preexec_fn=cls._set_limits,
            )
            combined_output_parts.append(f"[run] exit={run.returncode}")
            if run.stdout.strip():
                combined_output_parts.append(f"[run stdout]\n{run.stdout}")
            if run.stderr.strip():
                combined_output_parts.append(f"[run stderr]\n{run.stderr}")
            if run.returncode != 0:
                run_ok = False
        except subprocess.TimeoutExpired:
            return False, "\n".join(combined_output_parts + [f"[run] TIMEOUT after {CODE_TIMEOUT_SECONDS}s"])
        except Exception as e:
            return False, "\n".join(combined_output_parts + [f"[run] EXCEPTION: {e}"])

        test_ok = True
        if test_code and test_path.exists():
            try:
                tr = subprocess.run(
                    [python_exe, "-m", "pytest", str(test_path), "-v", "--tb=short"],
                    cwd=str(tmpdir),
                    capture_output=True, text=True,
                    timeout=CODE_TIMEOUT_SECONDS,
                    preexec_fn=cls._set_limits,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                )
                combined_output_parts.append(f"[pytest] exit={tr.returncode}")
                if tr.stdout.strip():
                    combined_output_parts.append(f"[pytest stdout]\n{tr.stdout}")
                if tr.stderr.strip():
                    combined_output_parts.append(f"[pytest stderr]\n{tr.stderr}")
                test_ok = (tr.returncode == 0)
            except subprocess.TimeoutExpired:
                test_ok = False
                combined_output_parts.append("[pytest] TIMEOUT")
            except Exception as e:
                test_ok = False
                combined_output_parts.append(f"[pytest] EXCEPTION: {e}")

        overall_ok = run_ok and test_ok
        ProvenanceLedger.record("code_execution", {
            "ok": overall_ok,
            "has_tests": bool(test_code),
            "output_chars": sum(len(s) for s in combined_output_parts),
        })
        return overall_ok, "\n".join(combined_output_parts)


# =========================================================================
# Section 8: Console Formatter (ASCII Boxes, Syntax Highlighting)
# =========================================================================

class ConsoleFormatter:
    @staticmethod
    def _width() -> int:
        try:
            return min(120, shutil.get_terminal_size((100, 40)).columns)
        except Exception:
            return 100

    @classmethod
    def _box(cls, lines: List[str], title: str = "",
             color_title: str = "\033[1;36m",
             color_border: str = "\033[1;30m",
             color_content: str = "\033[0m") -> str:
        W = cls._width()
        inner = W - 4
        reset = "\033[0m"
        top = f"{color_border}╔{'═' * (W - 2)}╗{reset}"
        if title:
            title_line = f"{color_border}║ {color_title}{title.center(inner)}{reset}{color_border} ║{reset}"
        else:
            title_line = f"{color_border}║{' ' * (W - 2)}║{reset}"
        sep = f"{color_border}╠{'─' * (W - 2)}╣{reset}"
        bot = f"{color_border}╚{'═' * (W - 2)}╝{reset}"
        wrapped_lines: List[str] = []
        for line in lines:
            if not line:
                wrapped_lines.append("")
                continue
            for part in line.split("\n"):
                sub = textwrap.wrap(part, width=inner, replace_whitespace=False,
                                    drop_whitespace=False, break_long_words=True)
                if not sub:
                    wrapped_lines.append("")
                else:
                    wrapped_lines.extend(sub)
        body_lines = []
        for wl in wrapped_lines:
            padded = wl.ljust(inner)[:inner]
            body_lines.append(f"{color_border}║ {color_content}{padded}{reset}{color_border} ║{reset}")
        parts = [top, title_line, sep] + body_lines + [bot]
        return "\n".join(parts)

    @staticmethod
    def highlight_code(code: str, language: str = "Python") -> str:
        try:
            lexer = get_lexer_by_name(language.lower(), stripall=False)
        except Exception:
            try:
                lexer = guess_lexer(code)
            except Exception:
                return code
        formatter = TerminalFormatter(style=ExamStyle)
        return highlight(code, lexer, formatter)

    @classmethod
    def header(cls, text: str) -> None:
        W = cls._width()
        print("\n" + "=" * W)
        print(f"  {text}")
        print("=" * W + "\n")

    @classmethod
    def print_state_transition(cls, qid: int, src: str, dst: str,
                               note: str = "") -> None:
        arrow = "\033[1;33m-->\033[0m"
        print(f"\033[1;35m[Q{qid}]\033[0m state: "
              f"\033[2m{src}\033[0m {arrow} \033[1;32m{dst}\033[0m"
              + (f"  \033[2m({note})\033[0m" if note else ""))

    @classmethod
    def print_session_status(cls, num_images: int, mode_desc: str) -> None:
        W = cls._width()
        ready = "\033[1;32mREADY\033[0m" if num_images == 0 else \
            f"\033[1;33m{num_images} IMAGE(S)\033[0m"
        bar = f"[{ready}] {mode_desc}  |  " \
              f"\033[1m{CAPTURE_LABEL}\033[0m=Capture  \033[1m{PROCESS_LABEL}\033[0m=Process  \033[1m{QUIT_LABEL}\033[0m=Quit"
        print("\r" + bar.ljust(W), end="", flush=True)

    @classmethod
    def print_question_header(cls, qid: int, ptype: ProblemType,
                              state: QuestionState) -> None:
        cls.header(f"QUESTION #{qid}  TYPE={ptype.value.upper()}  STATE={state.name}")

    @classmethod
    def print_full_question(cls, text: str) -> None:
        lines = text.splitlines() or ["(empty)"]
        print(cls._box(lines, title=" QUESTION TEXT "))

    @classmethod
    def print_mc_answer(cls, ans: MultipleChoiceAnswer) -> None:
        lines = [
            f"EXPLANATION: {ans.explanation_of_question}",
            "",
            f"REASONING:   {ans.reasoning}",
            "",
            f"SINGLE={ans.is_single_answer}  MULTIPLE={ans.is_multiple_answer}",
            f"ANSWER = {ans.answer}",
        ]
        print(cls._box(lines, title=" MULTIPLE-CHOICE ANSWER ",
                       color_title="\033[1;33m"))

    @classmethod
    def print_clarification_box(cls, questions: List[str],
                                assumptions: Optional[List[str]] = None) -> None:
        lines = ["The visible prompt appears underspecified."]
        for idx, question in enumerate(questions, start=1):
            lines.append(f"{idx}. {question}")
        if assumptions:
            lines.append("")
            lines.append("Conservative assumptions if unanswered:")
            for assumption in assumptions:
                lines.append(f"- {assumption}")
        print(cls._box(lines, title=" CLARIFICATION QUESTIONS ",
                       color_title="\033[1;35m"))

    @classmethod
    def print_code_version(cls, cv: CodeVersion, language: str,
                           header_title: str) -> None:
        status_color = "\033[1;31m" if cv.tests_passed is False else \
            ("\033[1;32m" if cv.tests_passed else "\033[1;33m")
        status = ("PASS" if cv.tests_passed else "FAIL") \
            if cv.tests_passed is not None else "NOT TESTED"
        title = f" {header_title}  [{status_color}{status}\033[0m] "
        meta_lines = []
        if cv.bug_type:
            meta_lines.append(f"BUG TYPE:        {cv.bug_type}")
        if cv.bug_description:
            meta_lines.append(f"BUG DESCRIPTION: {cv.bug_description}")
        if meta_lines:
            print(cls._box(meta_lines, title=" METADATA ",
                           color_title="\033[1;35m"))
        print(cls._box([title.strip()], title="",
                       color_title="\033[1m\033[36m"))
        highlighted = cls.highlight_code(cv.code, language)
        for line in highlighted.splitlines():
            print(f"  {line}")
        print()
        if cv.test_output:
            print(cls._box(cv.test_output.splitlines(),
                           title=" TEST / RUN OUTPUT ",
                           color_title="\033[1;34m"))

    @classmethod
    def print_summary(cls, qid: int, ptype: ProblemType,
                      final_state: QuestionState,
                      correctness: str, note: str = "") -> None:
        color_state = "\033[1;32m" if final_state in {
            QuestionState.VERIFIED, QuestionState.PASSED
        } else "\033[1;31m"
        lines = [
            f"QUESTION ID:    {qid}",
            f"TYPE:           {ptype.value}",
            f"FINAL STATE:    {color_state}{final_state.name}\033[0m",
            f"CORRECTNESS:    {correctness}",
            "",
            f"NOTE: {note}" if note else "",
            "",
            f"Database:       {DB_PATH}",
            f"Provenance:     {PROVENANCE_LOG}",
        ]
        print(cls._box([l for l in lines if l is not None and l != ""],
                       title=" SESSION SUMMARY ",
                       color_title="\033[1;36m"))


# =========================================================================
# Section 9: Main Pipeline Orchestrator
# =========================================================================

class ExamPipeline:
    def __init__(self):
        self.db = Database()
        self.llm = LLMClient()
        self.artifact_builder = AdaptiveArtifactBuilder(self.llm, self.db)
        self.shots = ScreenshotManager()
        self.session_images: List[CapturedImage] = []
        self.session_id: str = datetime.now().strftime("%Y%m%dT%H%M%S") + \
            "-" + hashlib.md5(str(random.random()).encode()).hexdigest()[:6]
        # OCR runs on a dedicated worker so captures return instantly and the
        # hotkey loop stays responsive; handle_process drains it before use.
        self._ocr_queue: "queue.Queue[Optional[CapturedImage]]" = queue.Queue()
        self._ocr_worker = threading.Thread(
            target=self._ocr_worker_loop,
            name="fulltest-ocr-worker",
            daemon=True,
        )
        self._ocr_worker.start()
        ProvenanceLedger.record("session_started",
                                {"session_id": self.session_id,
                                 "mode": "local" if PLAY_LOCAL else "online"})

    def _ocr_worker_loop(self) -> None:
        while True:
            img = self._ocr_queue.get()
            try:
                if img is None:
                    return
                with Image.open(img.file_path) as image:
                    img.ocr_text = ScreenshotManager._extract_ocr_text(image)
                ProvenanceLedger.record("ocr_completed", {
                    "file": str(img.file_path),
                    "ocr_len": len(img.ocr_text),
                    "ocr_preview": img.ocr_text[:200],
                })
            except Exception as e:
                ProvenanceLedger.record("ocr_error", {
                    "file": str(getattr(img, "file_path", "?")),
                    "error": str(e),
                })
            finally:
                self._ocr_queue.task_done()

    @staticmethod
    def _parsed_metadata(parsed: ParsedQuestion) -> Dict[str, Any]:
        return {
            "assessment_family": parsed.likely_assessment_family,
            "progressive_assessment": parsed.likely_progressive_assessment,
            "progressive_requirements": parsed.progressive_requirements,
            "design_requirements": parsed.design_requirements,
            "requires_stateful_solution": parsed.requires_stateful_solution,
            "question_contract_known": parsed.question_contract_known,
            "requires_clarification": parsed.requires_clarification,
            "visible_interfaces": parsed.visible_interfaces,
            "required_runtime_artifacts": parsed.required_runtime_artifacts,
            "artifact_generation_strategy": parsed.artifact_generation_strategy,
            "conservative_assumptions": parsed.conservative_assumptions,
        }

    def _handle_clarifications(self, qid: int, parsed: ParsedQuestion) -> ParsedQuestion:
        questions = parsed.clarification_questions or []
        if not questions:
            return parsed

        ConsoleFormatter.print_clarification_box(
            questions, assumptions=parsed.conservative_assumptions
        )

        answered_pairs: List[Tuple[str, str]] = []
        interactive_terminal = sys.stdin.isatty()
        if interactive_terminal:
            print("Provide optional clarification answers. Press Enter to skip any question.\n")
            for idx, question_text in enumerate(questions, start=1):
                try:
                    answer_text = input(f"Clarification {idx}: {question_text}\n> ").strip()
                except EOFError:
                    answer_text = ""
                status = "answered" if answer_text else "unanswered"
                self.db.add_clarification_item(qid, question_text, answer_text or None, status)
                if answer_text:
                    answered_pairs.append((question_text, answer_text))
        else:
            for question_text in questions:
                self.db.add_clarification_item(qid, question_text, None, "unanswered")

        augmented = ParsedQuestion(**parsed.model_dump())
        if answered_pairs:
            clarification_block = ["[CLARIFICATION ANSWERS]"]
            for idx, (question_text, answer_text) in enumerate(answered_pairs, start=1):
                clarification_block.append(f"Q{idx}: {question_text}")
                clarification_block.append(f"A{idx}: {answer_text}")
            augmented.full_question = augmented.full_question + "\n\n" + "\n".join(clarification_block)
            augmented.requires_clarification = False
        elif parsed.conservative_assumptions:
            assumption_block = ["[UNANSWERED CLARIFICATIONS - USE ONLY THESE CONSERVATIVE ASSUMPTIONS]"]
            assumption_block.extend(f"- {item}" for item in parsed.conservative_assumptions)
            augmented.full_question = augmented.full_question + "\n\n" + "\n".join(assumption_block)

        return augmented

    # ------------------------------------------------------------------
    # Keyboard handlers
    # ------------------------------------------------------------------
    def handle_capture(self) -> None:
        try:
            # #region debug-point E:handle-capture-entry
            _debug_hotkey_report(
                "E",
                "main.py:1513",
                "Entered handle_capture",
                {"session_image_count_before": len(self.session_images)},
            )
            # #endregion
            img = self.shots.capture(extract_text=False)
            self.session_images.append(img)
            self._ocr_queue.put(img)
            notify_user(
                f"Captured #{len(self.session_images)} ({img.file_path.name})",
                title="FullTest Capture",
            )
            ConsoleFormatter.print_session_status(
                len(self.session_images),
                f"session={self.session_id[:12]}")
        except Exception as e:
            print(f"\n\033[1;31m[CAPTURE ERROR]\033[0m {e}")
            ProvenanceLedger.record("capture_error", {"error": str(e)})

    def handle_process(self) -> None:
        # #region debug-point E:handle-process-entry
        _debug_hotkey_report(
            "E",
            "main.py:1533",
            "Entered handle_process",
            {"session_image_count_before": len(self.session_images)},
        )
        # #endregion
        if not self.session_images:
            notify_user("No screenshots captured yet.", title="FullTest Warning")
            print(f"\n\033[1;33m[WARN]\033[0m Press {CAPTURE_LABEL} first to capture at least one screenshot.")
            return
        try:
            # Always join: the OCR worker may have already dequeued the last
            # image, so empty()/qsize() are unreliable — unfinished_tasks is
            # the only safe signal, and join() waits for it.
            if self._ocr_queue.unfinished_tasks:
                print("\n\033[2m[INFO] Waiting for background OCR to finish...\033[0m")
            self._ocr_queue.join()
            self.session_images = ScreenshotManager.mark_duplicates(self.session_images)
            num_unique = sum(1 for i in self.session_images if not i.is_duplicate)
            unique_images = [i for i in self.session_images if not i.is_duplicate]
            if not any(i.ocr_text.strip() for i in unique_images):
                warning = (
                    "No readable text found in the captured screenshot(s). "
                    "Make sure the exam window is visible on the current desktop "
                    "and that your terminal has Screen Recording permission "
                    "(System Settings -> Privacy & Security -> Screen Recording)."
                )
                notify_user(warning, title="FullTest Warning")
                print(f"\n\033[1;33m[WARN]\033[0m {warning}")
                ProvenanceLedger.record("process_aborted_empty_ocr", {
                    "files": [str(i.file_path) for i in unique_images],
                })
                return
            # #region debug-point B:process-start
            _debug_report(
                "B",
                "main.py:1411",
                "Started processing current screenshot session",
                {
                    "total_images": len(self.session_images),
                    "unique_images": num_unique,
                    "files": [str(img.file_path) for img in self.session_images],
                },
            )
            # #endregion
            notify_user(
                f"Processing {len(self.session_images)} shot(s) ({num_unique} unique)...",
                title="FullTest AI Processing",
            )
            qid = self._run_pipeline()
            self.session_images = []
            notify_user(f"Question #{qid} complete.",
                        title="FullTest Complete")
            ConsoleFormatter.print_session_status(
                0, f"session={self.session_id[:12]} | last Q=#{qid}")
        except Exception as e:
            print(f"\n\033[1;31m[PROCESS ERROR]\033[0m {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            ProvenanceLedger.record("process_fatal",
                                    {"error": str(e),
                                     "trace": traceback.format_exc()})

    def process_existing_directory(self, directory: Path) -> int:
        if not directory.exists():
            raise RuntimeError(f"Screenshots directory does not exist: {directory}")
        self.session_images = ScreenshotManager.load_existing_images(directory)
        if not self.session_images:
            raise RuntimeError(f"No PNG screenshots found in {directory}")
        self.session_images = ScreenshotManager.mark_duplicates(self.session_images)
        num_unique = sum(1 for image in self.session_images if not image.is_duplicate)
        _debug_report(
            "B",
            "main.py:1464",
            "Loaded screenshots from existing directory",
            {
                "directory": str(directory),
                "total_images": len(self.session_images),
                "unique_images": num_unique,
                "files": [str(image.file_path) for image in self.session_images],
            },
        )
        print(f"\n[INFO] Processing screenshots from {directory} ({len(self.session_images)} files, {num_unique} unique)")
        qid = self._run_pipeline()
        ConsoleFormatter.print_session_status(
            0, f"session={self.session_id[:12]} | last Q=#{qid}")
        return qid

    def run_terminal_control_mode(self) -> None:
        if not sys.stdin.isatty():
            raise RuntimeError("Terminal control mode requires an interactive TTY.")

        print(
            "\n[INFO] Terminal control mode enabled. "
            "Use single-key controls: c to capture, p to process, q to quit."
        )
        ConsoleFormatter.print_session_status(
            len(self.session_images),
            f"session={self.session_id[:12]} | terminal-controls"
        )

        fd = sys.stdin.fileno()
        previous_terminal_settings = termios.tcgetattr(fd)

        try:
            tty.setraw(fd)
            while True:
                command = sys.stdin.read(1).lower()
                if command == "\x03":
                    raise KeyboardInterrupt
                if command == KEY_CAPTURE:
                    self.handle_capture()
                elif command == KEY_PROCESS:
                    self.handle_process()
                elif command == KEY_QUIT:
                    print("\n\n\033[1;31m[QUIT]\033[0m Terminal control mode exited.")
                    ProvenanceLedger.record("user_quit", {"mode": "terminal-controls"})
                    return
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, previous_terminal_settings)

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------
    def _run_pipeline(self) -> int:
        images_json = json.dumps([
            {"index": im.index, "path": str(im.file_path),
             "timestamp": im.timestamp, "phash": im.phash,
             "is_duplicate": im.is_duplicate,
             "ocr_len": len(im.ocr_text)}
            for im in self.session_images
        ])
        qid = self.db.create_question(self.session_id, images_json)
        # #region debug-point E:pipeline-start
        _debug_report(
            "E",
            "main.py:1438",
            "Created question row and entered pipeline",
            {
                "question_id": qid,
                "image_count": len(self.session_images),
                "unique_count": sum(1 for img in self.session_images if not img.is_duplicate),
            },
        )
        # #endregion

        ConsoleFormatter.print_question_header(
            qid, ProblemType.UNKNOWN, QuestionState.CAPTURED)
        ConsoleFormatter.print_state_transition(
            qid, "NEW", QuestionState.CAPTURED.name,
            f"{len(self.session_images)} images, "
            f"{sum(1 for i in self.session_images if not i.is_duplicate)} unique")

        # Stage: PARSE
        try:
            parsed = self.llm.parse_question(self.session_images)
        except Exception as e:
            self.db.set_state(qid, QuestionState.ERROR_PARSE)
            self.db.update_question(qid, status_note=f"parse failed: {e}")
            ConsoleFormatter.print_state_transition(
                qid, QuestionState.CAPTURED.name,
                QuestionState.ERROR_PARSE.name, str(e))
            ConsoleFormatter.print_summary(
                qid, ProblemType.UNKNOWN,
                QuestionState.ERROR_PARSE, "UNCHECKED", str(e))
            return qid

        raw_text = "\n".join(i.ocr_text for i in self.session_images)
        self.db.update_question(
            qid,
            problem_type=parsed.problem_type.value,
            full_question=parsed.full_question,
            raw_question_text=raw_text,
            metadata_json=json.dumps(self._parsed_metadata(parsed)),
        )
        self.db.set_state(qid, QuestionState.PARSED)
        ConsoleFormatter.print_state_transition(
            qid, QuestionState.CAPTURED.name, QuestionState.PARSED.name,
            f"type={parsed.problem_type.value}")
        ConsoleFormatter.print_question_header(
            qid, parsed.problem_type, QuestionState.PARSED)
        ConsoleFormatter.print_full_question(parsed.full_question)

        if parsed.extracted_tests:
            for idx, t in enumerate(parsed.extracted_tests):
                self.db.add_extracted_test(qid, f"extracted_{idx}", t)

        if parsed.problem_type == ProblemType.PROGRAMMING and parsed.requires_clarification:
            parsed = self._handle_clarifications(qid, parsed)
            self.db.update_question(
                qid,
                full_question=parsed.full_question,
                metadata_json=json.dumps(self._parsed_metadata(parsed)),
            )

        # Dispatch by type
        if parsed.problem_type == ProblemType.MULTIPLE_CHOICE:
            final_state, correctness, note = self._stage_mc(qid, parsed)
        elif parsed.problem_type == ProblemType.PROGRAMMING:
            final_state, correctness, note = self._stage_programming(qid, parsed)
        else:
            final_state, correctness, note = self._stage_general_text(qid, parsed)

        ConsoleFormatter.print_summary(
            qid, parsed.problem_type, final_state, correctness, note)
        return qid

    # ------------------------------------------------------------------
    @staticmethod
    def _deliver_answer(text: str, label: str) -> None:
        if copy_to_clipboard(text):
            print(f"\n\033[1;32m[CLIPBOARD]\033[0m {label} copied to clipboard — "
                  "paste it with Cmd+V (do NOT retype it; typos fail the tests).")
            notify_user(f"{label} copied to clipboard — paste with Cmd+V",
                        title="FullTest Answer Ready")

    # ------------------------------------------------------------------
    def _stage_mc(self, qid: int,
                  parsed: ParsedQuestion) -> Tuple[QuestionState, str, str]:
        self.db.set_state(qid, QuestionState.ANSWERING)
        ConsoleFormatter.print_state_transition(
            qid, QuestionState.PARSED.name, QuestionState.ANSWERING.name)
        try:
            ans = self.llm.answer_multiple_choice(parsed, self.session_images)
        except Exception as e:
            self.db.set_state(qid, QuestionState.ERROR_ANSWER)
            self.db.update_question(qid, status_note=f"mc answer failed: {e}")
            return QuestionState.ERROR_ANSWER, "UNCHECKED", str(e)

        answer_str = json.dumps({
            "answer": ans.answer,
            "reasoning": ans.reasoning,
            "single": ans.is_single_answer,
            "multiple": ans.is_multiple_answer,
            "explanation": ans.explanation_of_question,
        }, indent=2)
        self.db.update_question(qid, proposed_answer=answer_str,
                                correctness="PROPOSED",
                                status_note="multiple-choice answered via LLM")
        self.db.set_state(qid, QuestionState.PASSED)
        ConsoleFormatter.print_state_transition(
            qid, QuestionState.ANSWERING.name, QuestionState.PASSED.name,
            f"answer={ans.answer}")
        ConsoleFormatter.print_mc_answer(ans)
        self._deliver_answer(
            "Correct option(s): " + ", ".join(str(i) for i in ans.answer),
            "Multiple-choice answer")
        return QuestionState.PASSED, "PROPOSED", \
            f"MC answer: {ans.answer} (verification against ground truth required)"

    # ------------------------------------------------------------------
    def _stage_programming(self, qid: int,
                           parsed: ParsedQuestion) -> Tuple[QuestionState, str, str]:
        self.db.set_state(qid, QuestionState.ANSWERING)
        ConsoleFormatter.print_state_transition(
            qid, QuestionState.PARSED.name, QuestionState.ANSWERING.name)

        # Ensure venv is ready before the first run
        try:
            VenvManager.ensure_venv()
        except Exception as e:
            ProvenanceLedger.record("venv_warning", {"error": str(e)})

        attempts: List[CodeVersion] = []
        artifact_test_code: Optional[str] = None
        auxiliary_files: Dict[str, str] = {}
        artifact_notes = ""
        lang = parsed.programming_language or "Python"
        parsed_for_attempt = parsed

        for attempt in range(1, MAX_SOLUTION_ATTEMPTS + 1):
            if attempt > 1:
                ConsoleFormatter.header(
                    f"ATTEMPT {attempt}/{MAX_SOLUTION_ATTEMPTS} — "
                    "RE-GENERATING WITH FAILURE FEEDBACK")
            try:
                solution = self.llm.generate_solution(
                    parsed_for_attempt, self.session_images)
            except Exception as e:
                ProvenanceLedger.record("solution_generation_error", {
                    "question_id": qid, "attempt": attempt, "error": str(e)})
                if attempt == 1:
                    self.db.set_state(qid, QuestionState.ERROR_ANSWER)
                    self.db.update_question(
                        qid, status_note=f"solution generation failed: {e}")
                    return QuestionState.ERROR_ANSWER, "UNCHECKED", str(e)
                break

            lang = solution.programming_language or parsed.programming_language or "Python"

            # Prefer the freshest test code: each attempt may correct
            # OCR-corrupted expectations from the previous round.
            if solution.suggested_test_code:
                artifact_test_code = solution.suggested_test_code
                self.db.add_generated_artifact(
                    qid, "test_solution.py", "pytest-module",
                    f"attempt-{attempt}", artifact_test_code
                )
            elif not artifact_test_code:
                try:
                    artifact_test_code, auxiliary_files, artifact_notes = self.artifact_builder.build(
                        qid, parsed, self.session_images, solution.solution_code
                    )
                except Exception as e:
                    ProvenanceLedger.record("artifact_builder_warning", {
                        "question_id": qid, "attempt": attempt, "error": str(e)})

            candidate = CodeVersion(attempt, solution.solution_code)

            if attempt == 1:
                self.db.set_state(qid, QuestionState.TESTING)
                ConsoleFormatter.print_state_transition(
                    qid, QuestionState.ANSWERING.name, QuestionState.TESTING.name)

            try:
                ok, output = CodeExecutor.run_python(
                    candidate.code, artifact_test_code, auxiliary_files)
            except Exception as e:
                ok, output = False, f"[runner-exception] {e}"
            candidate.tests_passed = ok
            candidate.test_output = output
            self.db.add_code_version(qid, candidate)
            ConsoleFormatter.print_code_version(
                candidate, language=lang,
                header_title=f"SOLUTION (ATTEMPT {attempt}/{MAX_SOLUTION_ATTEMPTS})")
            self._deliver_answer(candidate.code,
                                 f"Solution code (attempt {attempt})")
            attempts.append(candidate)

            if ok:
                note = ("solution passes embedded tests and clean run"
                        if attempt == 1 else
                        f"solution passed on attempt {attempt} with failure feedback")
                self.db.update_question(qid,
                    proposed_answer=candidate.code,
                    correctness="VERIFIED",
                    status_note=note,
                    metadata_json=json.dumps({
                        "explanation": solution.explanation,
                        "language": lang,
                        "attempts": attempt,
                        "assessment_family": parsed.likely_assessment_family,
                        "progressive_assessment": parsed.likely_progressive_assessment,
                        "design_requirements": parsed.design_requirements,
                        "architecture_notes": solution.architecture_notes,
                        "artifact_notes": artifact_notes,
                        "required_runtime_artifacts": parsed.required_runtime_artifacts,
                        "visible_interfaces": parsed.visible_interfaces,
                    }))
                self.db.set_state(qid, QuestionState.VERIFIED)
                ConsoleFormatter.print_state_transition(
                    qid, QuestionState.TESTING.name, QuestionState.VERIFIED.name,
                    f"solution passed tests on attempt {attempt}")
                explanation = solution.explanation if attempt == 1 else \
                    solution.explanation + f" [after {attempt} attempts]"
                return QuestionState.VERIFIED, "VERIFIED", explanation

            ProvenanceLedger.record("solution_attempt_failed", {
                "question_id": qid,
                "attempt": attempt,
                "output_preview": (candidate.test_output or "")[:2000],
            })
            parsed_for_attempt = self._augment_parsed_with_failure(parsed, attempts)

        self.db.update_question(qid,
            proposed_answer=attempts[-1].code if attempts else None,
            correctness="FAILED_VERIFICATION",
            status_note=f"solution did not pass tests after {len(attempts)} attempt(s)")
        self.db.set_state(qid, QuestionState.FAILED)
        ConsoleFormatter.print_state_transition(
            qid, QuestionState.TESTING.name, QuestionState.FAILED.name,
            f"solution did not pass tests after {len(attempts)} attempt(s)")
        return QuestionState.FAILED, "FAILED_VERIFICATION", \
            "See test output above or DB code_versions table"

    @staticmethod
    def _augment_parsed_with_failure(parsed: ParsedQuestion,
                                     attempts: List[CodeVersion]) -> ParsedQuestion:
        aug = ParsedQuestion(**parsed.model_dump())
        parts = [
            "",
            f"[{len(attempts)} PREVIOUS ATTEMPT(S) FAILED VERIFICATION]",
            "Ground truth is the problem description above — NOT the previously generated code, and NOT",
            "necessarily the test expectations (they were extracted via OCR and may be corrupted).",
            "For each failing test, simulate the operations BY HAND against the description, step by step,",
            "writing out the intermediate state, then decide:",
            "- If the CODE violates the description: fix all defects and produce a new correct solution_code.",
            "- If the CODE follows the description but a TEST expectation contradicts it: the test is",
            "  OCR-corrupted. Keep the spec-conformant solution_code (improve it if needed) and return a",
            "  corrected suggested_test_code whose expectations match the description and your simulation.",
            "- NEVER make a failing test pass by weakening its expectation to match the code.",
            "- Sanity check: an example's expected-output list MUST contain exactly one element per query.",
            "  If the visible list has a different length, it is OCR-corrupted — reconstruct it by simulation.",
        ]
        for cv in attempts:
            parts.append(f"---- attempt {cv.version_number} solution code ----")
            parts.append(cv.code)
            parts.append(f"---- attempt {cv.version_number} test output ----")
            parts.append((cv.test_output or "")[:2000])
        aug.full_question = aug.full_question + "\n" + "\n".join(parts)
        return aug

    # ------------------------------------------------------------------
    def _stage_general_text(self, qid: int,
                            parsed: ParsedQuestion) -> Tuple[QuestionState, str, str]:
        self.db.set_state(qid, QuestionState.ANSWERING)
        ConsoleFormatter.print_state_transition(
            qid, QuestionState.PARSED.name, QuestionState.ANSWERING.name,
            f"type={parsed.problem_type.value} -> text answer")
        merged_ocr = LLMClient._merge_ocr_texts(self.session_images)
        prompt = f"""Answer the following {parsed.problem_type.value} question clearly, step by step.
Produce a concise but complete answer string.

Question:
{parsed.full_question}

Return ONLY valid JSON of the form:
{{"solution_text": "...your full answer here, as plain text with line breaks as \\n...",
  "explanation": "one sentence describing the approach"}}
"""
        try:
            if PLAY_LOCAL:
                raw = self.llm._llm_local_multi(prompt, self.session_images, True)
            else:
                raw = self.llm._llm_online(prompt, merged_ocr, True)
            obj = json.loads(raw) if isinstance(raw, str) else raw
            solution = obj.get("solution_text", str(obj))
            explanation = obj.get("explanation", "")
        except Exception as e:
            self.db.set_state(qid, QuestionState.ERROR_ANSWER)
            self.db.update_question(qid, status_note=f"text answer failed: {e}")
            return QuestionState.ERROR_ANSWER, "UNCHECKED", str(e)

        boxed_answer_lines = solution.splitlines() or ["(no text)"]
        print(ConsoleFormatter._box(boxed_answer_lines,
                                    title=f" {parsed.problem_type.value.upper()} ANSWER ",
                                    color_title="\033[1;33m"))
        self.db.update_question(qid, proposed_answer=solution,
                                correctness="PROPOSED",
                                status_note=explanation or "text answer")
        self.db.set_state(qid, QuestionState.PASSED)
        ConsoleFormatter.print_state_transition(
            qid, QuestionState.ANSWERING.name, QuestionState.PASSED.name,
            f"{parsed.problem_type.value} answer delivered")
        self._deliver_answer(solution, "Answer text")
        return QuestionState.PASSED, "PROPOSED", \
            (explanation or "Text answer: ground-truth verification required")


# =========================================================================
# Section 10: Entry Point & Listener Loop
# =========================================================================

def main() -> None:
    try:
        parser = argparse.ArgumentParser(description="FullTest exam processing pipeline")
        parser.add_argument(
            "--process-existing",
            type=str,
            help="Process PNG screenshots from an existing directory and exit",
        )
        args = parser.parse_args()

        validate_runtime_compatibility()
        ocr_binary = configure_ocr_runtime()
        banner = [
            "FULLTEST BRANCH  —  multi-shot capture · provenance DB · adaptive artifacts · sandboxed solution verification",
            "",
            f"Screenshots dir: {SCREENSHOTS_DIR}",
            f"Database:        {DB_PATH}",
            f"Provenance log:  {PROVENANCE_LOG}",
            f"Backend:         {'LOCAL (Ollama)' if PLAY_LOCAL else 'ONLINE (DeepSeek API)'}",
            f"OCR binary:      {ocr_binary}",
            f"Capture key:     {CAPTURE_LABEL}  |  Process key:  {PROCESS_LABEL}  |  Quit: {QUIT_LABEL}",
        ]
        print(ConsoleFormatter._box(banner,
            title=" FULLTEST PIPELINE INITIALIZED ", color_title="\033[1;36m"))

        pipeline = ExamPipeline()

        if args.process_existing:
            pipeline.process_existing_directory(Path(args.process_existing).expanduser().resolve())
            return

        if should_use_terminal_control_mode():
            print(
                "\n[INFO] Terminal control mode requested via FULLTEST_TERMINAL_CONTROL. "
                "Single-key terminal controls are active in this window only."
            )
            pipeline.run_terminal_control_mode()
            return

        ensure_macos_input_monitoring_trust()

        if sys.platform == "darwin":
            notification_backend = init_notifications()
            backend_note = (
                "pync/terminal-notifier (Intel binary via Rosetta 2)"
                if notification_backend == "pync" else
                "osascript (native AppleScript)")
            print(
                "\n[INFO] macOS global hotkey monitor active (Quartz event tap with "
                f"auto-recovery). {CAPTURE_LABEL}=Capture  {PROCESS_LABEL}=Process  "
                f"{QUIT_LABEL}=Quit — detected from any app and any desktop/Space, "
                "and suppressed so other apps never receive them."
            )
            print(
                f"[INFO] Machine: {platform.machine()} | "
                f"Notifications: {backend_note}"
            )
        else:
            print(
                "\n[INFO] Global hotkey listener active. "
                f"{CAPTURE_LABEL}=Capture  {PROCESS_LABEL}=Process  {QUIT_LABEL}=Quit."
            )

        ConsoleFormatter.print_session_status(
            0, f"session={pipeline.session_id[:12]}")

        action_queue: "queue.Queue[Optional[str]]" = queue.Queue()
        hotkey_state_lock = threading.Lock()
        last_hotkey_at: Dict[str, float] = {}

        def should_accept_hotkey(action: str) -> bool:
            now = time.monotonic()
            with hotkey_state_lock:
                last_seen = last_hotkey_at.get(action, 0.0)
                if now - last_seen < HOTKEY_DEBOUNCE_SECONDS:
                    return False
                last_hotkey_at[action] = now
                return True

        def enqueue_action(action: str) -> None:
            if not should_accept_hotkey(action):
                return
            action_queue.put(action)

        def action_worker() -> None:
            while True:
                action = action_queue.get()
                try:
                    if action is None:
                        return
                    if action == "capture":
                        pipeline.handle_capture()
                    elif action == "process":
                        pipeline.handle_process()
                finally:
                    action_queue.task_done()

        action_thread = threading.Thread(
            target=action_worker,
            name="fulltest-hotkey-worker",
            daemon=True,
        )
        action_thread.start()

        shutdown_event = threading.Event()

        def handle_hotkey_action(action: str, raw_key: str = "") -> None:
            if action == "quit":
                # #region debug-point B:esc-received
                _debug_hotkey_report(
                    "B",
                    "main.py:1987",
                    "Received quit key in listener",
                    {"raw_key": raw_key, "action": action},
                )
                # #endregion
                print(f"\n\n\033[1;31m[QUIT]\033[0m {QUIT_LABEL} pressed — shutting down listener.")
                ProvenanceLedger.record("user_quit", {})
                action_queue.put(None)
                shutdown_event.set()
                return
            # #region debug-point A:key-press
            _debug_hotkey_report(
                "A",
                "main.py:1996",
                "Received key press",
                {"raw_key": raw_key, "action": action},
            )
            # #endregion
            if action == "capture":
                # #region debug-point C:capture-combo
                _debug_hotkey_report(
                    "C",
                    "main.py:2011",
                    "Capture hotkey matched",
                    {"raw_key": raw_key, "action": action},
                )
                # #endregion
                enqueue_action("capture")
            elif action == "process":
                # #region debug-point C:process-combo
                _debug_hotkey_report(
                    "C",
                    "main.py:2019",
                    "Process hotkey matched",
                    {"raw_key": raw_key, "action": action},
                )
                # #endregion
                enqueue_action("process")

        if sys.platform == "darwin":
            monitor = MacOSHotkeyMonitor(handle_hotkey_action)
            monitor.start()
            try:
                while not shutdown_event.wait(timeout=0.5):
                    pass
            finally:
                monitor.stop()
        else:
            fkey_actions = {
                pynput.keyboard.Key.f6: "capture",
                pynput.keyboard.Key.f7: "process",
                pynput.keyboard.Key.f8: "quit",
            }

            def on_press(key):
                action = fkey_actions.get(key)
                if action is None:
                    return None
                handle_hotkey_action(action, str(key))
                if action == "quit":
                    return False
                return None

            def on_release(key):
                # #region debug-point D:key-release
                _debug_hotkey_report(
                    "D",
                    "main.py:2033",
                    "Processed key release",
                    {
                        "raw_key": str(key),
                    },
                )
                # #endregion
                return None

            with pynput.keyboard.Listener(
                on_press=on_press,
                on_release=on_release,
            ) as listener:
                listener.join()
        action_thread.join(timeout=1)
    except KeyboardInterrupt:
        print("\n\n\033[1;33m[INFO]\033[0m Keyboard interrupt received. Goodbye.")
    except RuntimeError as e:
        print(f"\n\033[1;31m[RUNTIME ERROR]\033[0m {e}")


if __name__ == "__main__":
    main()
