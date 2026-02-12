# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for parallelized review_patch.

Covers the test-plan item:
    'Verify review_patch parallelization produces correct results'
"""

import metis.engine.core as coremod


def test_review_patch_parallel_multiple_files(engine, monkeypatch, tmp_path):
    """review_patch should process multiple file diffs concurrently and
    return correct results for each one."""

    patch = """\
--- a/file1.c
+++ b/file1.c
@@ -0,0 +1,2 @@
+int a = 1;
+int b = 2;
--- a/file2.c
+++ b/file2.c
@@ -0,0 +1,2 @@
+int x = 10;
+int y = 20;
"""
    patch_file = tmp_path / "multi.diff"
    patch_file.write_text(patch)

    reviewed_files = []

    class _TrackingReviewGraph:
        def review(self, req):
            rel = req.get("relative_file", req["file_path"])
            reviewed_files.append(rel)
            return {
                "file": rel,
                "file_path": req["file_path"],
                "reviews": [{"issue": f"Issue in {rel}"}],
            }

    monkeypatch.setattr(engine, "_get_review_graph", lambda: _TrackingReviewGraph())
    monkeypatch.setattr(coremod, "summarize_changes", lambda *a, **k: "summary")
    monkeypatch.setattr(coremod, "build_summary_chain", lambda *a, **k: None)

    result = engine.review_patch(str(patch_file))

    assert "reviews" in result
    assert len(result["reviews"]) == 2

    # Both files should have been reviewed
    review_files = {r["file"] for r in result["reviews"]}
    assert any("file1.c" in f for f in review_files)
    assert any("file2.c" in f for f in review_files)

    # Overall summary should be present
    assert result["overall_changes"]


def test_review_patch_parallel_preserves_order_independence(engine, monkeypatch, tmp_path):
    """Results should be correct regardless of thread execution order."""

    patch = """\
--- a/a.c
+++ b/a.c
@@ -0,0 +1 @@
+int a;
--- a/b.c
+++ b/b.c
@@ -0,0 +1 @@
+int b;
--- a/c.c
+++ b/c.c
@@ -0,0 +1 @@
+int c;
"""
    patch_file = tmp_path / "three.diff"
    patch_file.write_text(patch)

    class _DummyReviewGraph:
        def review(self, req):
            rel = req.get("relative_file", req["file_path"])
            return {
                "file": rel,
                "file_path": req["file_path"],
                "reviews": [{"issue": f"found in {rel}"}],
            }

    monkeypatch.setattr(engine, "_get_review_graph", lambda: _DummyReviewGraph())
    monkeypatch.setattr(coremod, "summarize_changes", lambda *a, **k: "sum")
    monkeypatch.setattr(coremod, "build_summary_chain", lambda *a, **k: None)

    result = engine.review_patch(str(patch_file))

    assert len(result["reviews"]) == 3
    # Each file's review should reference the correct file
    for review_dict in result["reviews"]:
        file_name = review_dict["file"]
        issue = review_dict["reviews"][0]["issue"]
        assert file_name in issue


def test_review_patch_parallel_handles_single_file_error(engine, monkeypatch, tmp_path):
    """If one file review fails, others should still succeed."""

    patch = """\
--- a/good.c
+++ b/good.c
@@ -0,0 +1 @@
+int ok;
--- a/bad.c
+++ b/bad.c
@@ -0,0 +1 @@
+int fail;
"""
    patch_file = tmp_path / "mixed.diff"
    patch_file.write_text(patch)

    class _PartialFailReviewGraph:
        def review(self, req):
            rel = req.get("relative_file", req["file_path"])
            if "bad.c" in rel:
                raise RuntimeError("simulated failure")
            return {
                "file": rel,
                "file_path": req["file_path"],
                "reviews": [{"issue": "ok"}],
            }

    monkeypatch.setattr(engine, "_get_review_graph", lambda: _PartialFailReviewGraph())
    monkeypatch.setattr(coremod, "summarize_changes", lambda *a, **k: "sum")
    monkeypatch.setattr(coremod, "build_summary_chain", lambda *a, **k: None)

    result = engine.review_patch(str(patch_file))

    # Should still have the successful review
    assert len(result["reviews"]) >= 1
    assert any("good.c" in r["file"] for r in result["reviews"])
