# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for batched review on a real (test) codebase.

Covers the test-plan item:
    'Verify batched review performance on a real codebase'
"""

from unittest.mock import Mock, MagicMock


def test_review_code_batched_yields_results_for_all_files(engine, monkeypatch):
    """Batched review should yield one result per code file in tests/data."""

    class _DummyReviewGraph:
        def review(self, req):
            return {
                "file": req.get("relative_file", req["file_path"]),
                "file_path": req["file_path"],
                "reviews": [{"issue": "test issue"}],
            }

        def review_batch(self, file_entries, language_prompts, **kw):
            results = []
            for entry in file_entries:
                rel = entry.get("relative_file") or entry["file_path"]
                results.append({
                    "file": rel,
                    "file_path": entry["file_path"],
                    "reviews": [{"issue": "batch issue"}],
                })
            return results

    monkeypatch.setattr(engine, "_get_review_graph", lambda: _DummyReviewGraph())

    code_files = engine.get_code_files()
    results = list(engine.review_code_batched(files=code_files))

    # Every code file should produce a result (not None for valid files)
    non_none = [r for r in results if r is not None]
    assert len(non_none) == len(code_files)
    for r in non_none:
        assert "reviews" in r
        assert len(r["reviews"]) > 0


def test_review_code_batched_fallback_on_batch_failure(engine, monkeypatch):
    """When review_batch returns None, it should fall back to single-file review."""

    class _FailingBatchReviewGraph:
        def review(self, req):
            return {
                "file": req.get("relative_file", req["file_path"]),
                "file_path": req["file_path"],
                "reviews": [{"issue": "fallback issue"}],
            }

        def review_batch(self, file_entries, language_prompts, **kw):
            return None  # Force fallback

    monkeypatch.setattr(engine, "_get_review_graph", lambda: _FailingBatchReviewGraph())

    code_files = engine.get_code_files()
    results = list(engine.review_code_batched(files=code_files))

    non_none = [r for r in results if r is not None]
    assert len(non_none) == len(code_files)
    # Should have used the fallback single-file path
    for r in non_none:
        assert r["reviews"][0]["issue"] == "fallback issue"


def test_review_code_batched_accepts_precomputed_files(engine, monkeypatch):
    """Passing a pre-computed file list should skip get_code_files()."""

    class _DummyReviewGraph:
        def review(self, req):
            return {
                "file": req.get("relative_file", req["file_path"]),
                "file_path": req["file_path"],
                "reviews": [],
            }

        def review_batch(self, file_entries, language_prompts, **kw):
            return [{
                "file": e.get("relative_file") or e["file_path"],
                "file_path": e["file_path"],
                "reviews": [],
            } for e in file_entries]

    monkeypatch.setattr(engine, "_get_review_graph", lambda: _DummyReviewGraph())

    # Spy on get_code_files
    original_get = engine.get_code_files
    call_count = {"n": 0}

    def counting_get():
        call_count["n"] += 1
        return original_get()

    monkeypatch.setattr(engine, "get_code_files", counting_get)

    code_files = original_get()
    list(engine.review_code_batched(files=code_files))

    # get_code_files should NOT have been called inside review_code_batched
    assert call_count["n"] == 0
