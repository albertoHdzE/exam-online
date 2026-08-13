# Debug Session: screenshots-batch-parse
- **Status**: [OPEN]
- **Issue**: Existing screenshots in the screenshots folder are valid, but the pipeline does not reconstruct the question and answer correctly from them.
- **Debug Server**: pending startup
- **Log File**: .dbg/trae-debug-log-screenshots-batch-parse.ndjson

## Reproduction Steps
1. Use the existing screenshots in [screenshots](file:///Users/alberto/Documents/projects/exam-online/screenshots).
2. Run the application in a non-interactive batch mode against those files.
3. Observe OCR extraction, duplicate decisions, parse classification, and answer generation.

## Hypotheses & Verification
| ID | Hypothesis | Likelihood | Effort | Evidence |
|----|------------|------------|--------|----------|
| A | OCR on the narrow prompt panel is still extracting too little text for parsing. | High | Low | Pending |
| B | The application lacks a reliable batch ingestion path for existing screenshots. | High | Medium | Pending |
| C | The online parse path relies too heavily on OCR text and underuses image content. | Medium | Medium | Pending |
| D | Distinct screenshots are still being marked as duplicates and key context is lost. | Medium | Low | Pending |
| E | The parse stage fails before the Python answer/test stage can run. | High | Low | Pending |

## Log Evidence
- `.dbg/trae-debug-log-screenshots-batch-parse.ndjson`
- Hypothesis A confirmed: OCR produced no candidates for both screenshots.
- Hypothesis C partially confirmed: merged OCR sent to parser was only image markers plus `(no text detected)`.
- Hypothesis D rejected for this run: both screenshots were kept as unique.
- Hypothesis E confirmed: parse returned `problem_type=unknown` and `full_question="[IMAGE-1] [IMAGE-2]"`.
- Additional environment evidence: `pytesseract` is available in Python, but the `tesseract` binary is not installed on PATH.
- Post-fix evidence: the same screenshots were parsed into a full programming prompt, classified as `programming`, and verified successfully as question `#5`.
- Post-fix evidence: OCR runtime is configured at `/opt/homebrew/bin/tesseract`, and the batch mode path processed the screenshots directory directly.
- Post-fix evidence: verification passed after normalizing generated Python tests so they import the produced `solution` function.

## Verification Conclusion
- Pre-fix root cause is the missing `tesseract` runtime dependency, which makes OCR empty and forces the downstream parser into `unknown`.
- A secondary product gap remains: the application needs a deterministic batch mode to process an existing screenshots directory without hotkeys.
- Minimal fix applied:
  - install and configure Tesseract OCR runtime
  - add directory batch processing mode
  - strengthen OCR preprocessing around the prompt panel
  - normalize generated Python test modules before pytest execution
- Current status: reproducible success on the provided screenshots. Debug session remains open until user confirmation.
