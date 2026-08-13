"""Smoke test for MacOSHotkeyMonitor (F6/F7/F8 keycode hotkeys).

Part A (no permissions needed): drives the tap callback directly with real
synthetic CGEvents and a fake tap, verifying hotkey detection by keycode,
suppression, pass-through of regular typing keys, and disabled-tap
auto-recovery.

Part B (requires Accessibility trust): posts synthetic key events into the
real HID event stream and verifies end-to-end delivery.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import Quartz  # noqa: E402
import HIServices  # noqa: E402
from main import MacOSHotkeyMonitor  # noqa: E402

F6, F7, F8 = 97, 98, 100  # macOS virtual keycodes


def make_key_event(keycode: int):
    return Quartz.CGEventCreateKeyboardEvent(None, keycode, True)


# ---------- Part A: callback logic with a fake tap ----------
received = []
enable_calls = []
created = {}
fake_tap = object()

real_create = Quartz.CGEventTapCreate
real_enable = Quartz.CGEventTapEnable


def fake_create(location, placement, options, mask, callback, userinfo):
    created["args"] = (location, placement, options, mask)
    created["callback"] = callback
    return fake_tap


def fake_enable(tap, enable):
    enable_calls.append((tap, enable))


Quartz.CGEventTapCreate = fake_create
Quartz.CGEventTapEnable = fake_enable
try:
    monitor = MacOSHotkeyMonitor(received.append)
    assert monitor._create_tap(Quartz) is fake_tap
    callback = created["callback"]
    monitor._tap = fake_tap

    location, placement, options, mask = created["args"]
    assert location == Quartz.kCGSessionEventTap, "tap must be session-wide"
    assert options == Quartz.kCGEventTapOptionDefault, "tap must be active (allows suppression)"
    assert mask == Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown)
    print("[ok] tap configured session-wide, active, key-down only")

    # F6/F7/F8 -> actions, event suppressed (None)
    assert callback(None, Quartz.kCGEventKeyDown, make_key_event(F6), None) is None
    assert callback(None, Quartz.kCGEventKeyDown, make_key_event(F7), None) is None
    assert callback(None, Quartz.kCGEventKeyDown, make_key_event(F8), None) is None
    assert received == ["capture", "process", "quit"], f"got {received!r}"
    print("[ok] F6/F7/F8 map to capture/process/quit and are suppressed")

    # typing keys (c, p, q and others) -> pass through untouched
    for keycode, name in [(8, "c"), (35, "p"), (12, "q"), (0, "a")]:
        event = make_key_event(keycode)
        assert callback(None, Quartz.kCGEventKeyDown, event, None) is event, \
            f"'{name}' should pass through"
    assert received == ["capture", "process", "quit"], \
        f"typing keys leaked into handler: {received!r}"
    print("[ok] typing 'c'/'p'/'q' (and other keys) passes through to apps — "
          "no more collision with exam answers")

    # macOS disables the tap (Space switch / timeout) -> auto re-enable
    assert callback(None, Quartz.kCGEventTapDisabledByTimeout, None, None) is None
    assert callback(None, Quartz.kCGEventTapDisabledByUserInput, None, None) is None
    assert enable_calls == [(fake_tap, True), (fake_tap, True)], enable_calls
    print("[ok] tap auto re-enabled on disabled-by-timeout and disabled-by-user-input")
finally:
    Quartz.CGEventTapCreate = real_create
    Quartz.CGEventTapEnable = real_enable

# ---------- Part B: live event stream (needs Accessibility trust) ----------
if not bool(HIServices.AXIsProcessTrusted()):
    print("[skip] live event-stream test: this process is not Accessibility-"
          "trusted. Run from your trusted terminal for the full check.")
else:
    live_received = []
    live_monitor = MacOSHotkeyMonitor(live_received.append)
    live_monitor.start()
    time.sleep(1.0)
    assert live_monitor._tap is not None, "live tap creation failed despite trust"

    Quartz.CGEventPost(Quartz.kCGHIDEventTap, make_key_event(F6))
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, make_key_event(F7))
    time.sleep(1.0)
    assert live_received == ["capture", "process"], \
        f"live hotkeys not detected: {live_received!r}"
    print("[ok] live: F6/F7 detected from the real event stream")

    # Simulate macOS disabling the tap (what happens on Space switches).
    # The monitor's recovery path should re-enable it immediately, so hotkeys
    # keep flowing without any gap visible to the user.
    with live_monitor._tap_lock:
        tap = live_monitor._tap
    Quartz.CGEventTapEnable(tap, False)
    time.sleep(0.3)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, make_key_event(F6))
    time.sleep(1.0)
    assert live_received == ["capture", "process", "capture"], \
        f"hotkeys stopped after tap disable: {live_received!r}"
    print("[ok] live: hotkeys keep working after tap disable (auto-recovery)")

    live_monitor.stop()
    assert live_monitor._stopped.is_set()
    print("[ok] live: monitor stops cleanly")

print("SMOKE_TEST_PASSED")
