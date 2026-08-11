"""Verify async capture: capture returns instantly, OCR completes in the
background worker, and the process-time drain (queue join) works."""

import sys
import time

sys.path.insert(0, "/Users/alberto/Documents/projects/exam-online")

from main import ExamPipeline  # noqa: E402

pipeline = ExamPipeline()

# 1. capture with deferred OCR must be fast (no 84-pass tesseract inline)
t0 = time.time()
img = pipeline.shots.capture(extract_text=False)
elapsed = time.time() - t0
assert img.file_path.exists(), "screenshot file not saved"
assert img.ocr_text == "", "OCR should be deferred"
assert img.phash, "phash missing"
print(f"[ok] capture returned in {elapsed:.2f}s with OCR deferred "
      f"(file={img.file_path.name}, phash={img.phash})")
assert elapsed < 10, f"capture too slow: {elapsed:.1f}s"

# 2. background OCR worker fills ocr_text, and queue join drains it
pipeline.session_images.append(img)
pipeline._ocr_queue.put(img)
t0 = time.time()
pipeline._ocr_queue.join()
ocr_elapsed = time.time() - t0
assert img.ocr_text.strip(), "background OCR produced no text"
print(f"[ok] background OCR completed in {ocr_elapsed:.1f}s, "
      f"{len(img.ocr_text)} chars extracted")

# 3. duplicate marking works on the OCR-filled image (process-time path)
from main import ScreenshotManager  # noqa: E402
marked = ScreenshotManager.mark_duplicates(pipeline.session_images)
assert len(marked) == 1 and not marked[0].is_duplicate
print("[ok] mark_duplicates works after async OCR")

print("ASYNC_CAPTURE_TEST_PASSED")
