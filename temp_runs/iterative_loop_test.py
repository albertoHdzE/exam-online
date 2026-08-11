"""Verify _stage_programming iterates until tests pass (bounded by
MAX_SOLUTION_ATTEMPTS), using a scripted fake LLM but the REAL code executor,
pytest sandbox, database, and clipboard delivery."""

import sys

sys.path.insert(0, "/Users/alberto/Documents/projects/exam-online")

from main import (ExamPipeline, ParsedQuestion, ProblemType, QuestionState,  # noqa: E402
                  SingleSolutionCode, MAX_SOLUTION_ATTEMPTS)

WRONG_CODE = '''\
def solution(queries):
    counts = {}
    results = []
    for query in queries:
        operation, value = query[0], int(query[1])
        if operation == "ADD":
            counts[value] = counts.get(value, 0) + 1
            results.append("")
        elif operation == "EXISTS":
            results.append("true" if counts.get(value, 0) > 0 else "false")
        elif operation == "REMOVE":
            if counts.get(value, 0) > 0:
                counts[value] -= 1
            results.append("")  # BUG: REMOVE must return "true"/"false"
    return results
'''

CORRECT_CODE = '''\
def solution(queries):
    counts = {}
    results = []
    for query in queries:
        operation, value = query[0], int(query[1])
        if operation == "ADD":
            counts[value] = counts.get(value, 0) + 1
            results.append("")
        elif operation == "EXISTS":
            results.append("true" if counts.get(value, 0) > 0 else "false")
        elif operation == "REMOVE":
            if counts.get(value, 0) > 0:
                counts[value] -= 1
                results.append("true")
            else:
                results.append("false")
    return results
'''

CORRECT_TESTS = '''\
from solution import *

def test_level2_example():
    queries = [["ADD","1"],["ADD","2"],["ADD","2"],["ADD","3"],
               ["EXISTS","1"],["EXISTS","2"],["EXISTS","3"],
               ["REMOVE","2"],["REMOVE","1"],["EXISTS","2"],["EXISTS","1"]]
    assert solution(queries) == ["","","","","true","true","true",
                                 "true","true","true","false"]

def test_remove_nonexistent():
    assert solution([["REMOVE","5"]]) == ["false"]
'''


class FakeLLM:
    def __init__(self, scripted):
        self.scripted = scripted
        self.prompts = []

    def generate_solution(self, parsed, images):
        self.prompts.append(parsed.full_question)
        return self.scripted[len(self.prompts) - 1]


def make_pipeline(scripted):
    pipeline = ExamPipeline()
    pipeline.llm = FakeLLM(scripted)
    return pipeline


def sol(code, tests):
    return SingleSolutionCode(solution_code=code, explanation="e",
                              programming_language="Python",
                              suggested_test_code=tests)


PARSED = ParsedQuestion(problem_type=ProblemType.PROGRAMMING,
                        full_question="container with ADD/EXISTS/REMOVE")

# --- Scenario A: wrong first attempt, corrected second -> VERIFIED -----------
p = make_pipeline([sol(WRONG_CODE, CORRECT_TESTS), sol(CORRECT_CODE, CORRECT_TESTS)])
qid = p.db.create_question(p.session_id, "[]")
state, correctness, note = p._stage_programming(qid, PARSED)
assert state == QuestionState.VERIFIED, state
assert len(p.llm.prompts) == 2, "loop must run exactly 2 attempts"
assert "1 PREVIOUS ATTEMPT(S) FAILED VERIFICATION" in p.llm.prompts[1], \
    "attempt 2 must receive the failure history"
versions = p.db.get_code_versions(qid)
assert len(versions) == 2 and versions[0]["tests_passed"] == 0 and versions[1]["tests_passed"] == 1
row = p.db.get_question(qid)
assert row["correctness"] == "VERIFIED" and "REMOVE" in row["proposed_answer"]
assert '"true"' in row["proposed_answer"], "final answer must be the corrected code"
print("[ok] scenario A: loop converges on attempt 2, failure history fed back, "
      "corrected code becomes the proposed answer")

# --- Scenario B: never passes -> FAILED after MAX_SOLUTION_ATTEMPTS ----------
p = make_pipeline([sol(WRONG_CODE, CORRECT_TESTS)] * MAX_SOLUTION_ATTEMPTS)
qid = p.db.create_question(p.session_id, "[]")
state, correctness, note = p._stage_programming(qid, PARSED)
assert state == QuestionState.FAILED, state
assert len(p.llm.prompts) == MAX_SOLUTION_ATTEMPTS, "must exhaust all attempts"
assert len(p.db.get_code_versions(qid)) == MAX_SOLUTION_ATTEMPTS
row = p.db.get_question(qid)
assert row["correctness"] == "FAILED_VERIFICATION"
assert f"{MAX_SOLUTION_ATTEMPTS - 1} PREVIOUS ATTEMPT(S)" in p.llm.prompts[-1], \
    "last attempt must see the full failure history"
print(f"[ok] scenario B: exhausts exactly {MAX_SOLUTION_ATTEMPTS} attempts, "
      "all versions recorded, marked FAILED_VERIFICATION")

# --- Scenario C: first attempt passes -> single attempt, VERIFIED ------------
p = make_pipeline([sol(CORRECT_CODE, CORRECT_TESTS)])
qid = p.db.create_question(p.session_id, "[]")
state, correctness, note = p._stage_programming(qid, PARSED)
assert state == QuestionState.VERIFIED and len(p.llm.prompts) == 1
print("[ok] scenario C: immediate pass stops after one attempt")

# --- Scenario D: budget exhausted, user continues -> converges on attempt 5 --
import builtins  # noqa: E402
import main as main_mod  # noqa: E402


class _FakeTty:
    def isatty(self):
        return True


_orig_stdin, _orig_input, _orig_notify = (
    main_mod.sys.stdin, builtins.input, main_mod.notify_user)
main_mod.sys.stdin = _FakeTty()
builtins.input = lambda prompt="": ""  # user presses Enter -> keep iterating
main_mod.notify_user = lambda *a, **k: None
try:
    scripted = [sol(WRONG_CODE, CORRECT_TESTS)] * MAX_SOLUTION_ATTEMPTS + \
               [sol(CORRECT_CODE, CORRECT_TESTS)]
    p = make_pipeline(scripted)
    qid = p.db.create_question(p.session_id, "[]")
    state, correctness, note = p._stage_programming(qid, PARSED)
    assert state == QuestionState.VERIFIED, state
    assert len(p.llm.prompts) == MAX_SOLUTION_ATTEMPTS + 1, \
        "loop must extend past the initial budget when the user continues"
    assert f"{MAX_SOLUTION_ATTEMPTS} PREVIOUS ATTEMPT(S)" in p.llm.prompts[-1], \
        "extended attempt must see the full failure history"
finally:
    main_mod.sys.stdin, builtins.input, main_mod.notify_user = (
        _orig_stdin, _orig_input, _orig_notify)
print(f"[ok] scenario D: after {MAX_SOLUTION_ATTEMPTS} failures, Enter extends "
      "the budget and the loop converges on attempt 5")

# --- Scenario E: budget exhausted, user stops -> FAILED at budget ------------
_orig_stdin, _orig_input, _orig_notify = (
    main_mod.sys.stdin, builtins.input, main_mod.notify_user)
main_mod.sys.stdin = _FakeTty()
builtins.input = lambda prompt="": "stop"
main_mod.notify_user = lambda *a, **k: None
try:
    p = make_pipeline([sol(WRONG_CODE, CORRECT_TESTS)] * (MAX_SOLUTION_ATTEMPTS + 2))
    qid = p.db.create_question(p.session_id, "[]")
    state, correctness, note = p._stage_programming(qid, PARSED)
    assert state == QuestionState.FAILED, state
    assert len(p.llm.prompts) == MAX_SOLUTION_ATTEMPTS, \
        "'stop' must end the loop at the initial budget"
finally:
    main_mod.sys.stdin, builtins.input, main_mod.notify_user = (
        _orig_stdin, _orig_input, _orig_notify)
print(f"[ok] scenario E: 'stop' ends the loop after {MAX_SOLUTION_ATTEMPTS} attempts")

print("ITERATIVE_LOOP_TEST_PASSED")
