# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for count_tokens, split_snippet, and the tiktoken encoder cache."""

from metis.utils import count_tokens, split_snippet, _encoding_cache, _get_encoding


class TestCountTokens:
    def test_known_string(self):
        # "hello world" is 2 tokens on gpt-4 (cl100k_base)
        assert count_tokens("hello world") == 2

    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_multiline(self):
        text = "line one\nline two\nline three\n"
        tokens = count_tokens(text)
        assert tokens > 0

    def test_encoder_is_cached(self):
        _get_encoding("gpt-4")
        assert "gpt-4" in _encoding_cache
        # Second call should return the same object
        enc1 = _get_encoding("gpt-4")
        enc2 = _get_encoding("gpt-4")
        assert enc1 is enc2


class TestSplitSnippet:
    def test_small_snippet_single_chunk(self):
        text = "a = 1\nb = 2\n"
        chunks = split_snippet(text, max_tokens=100)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_empty_snippet(self):
        assert split_snippet("", max_tokens=100) == []

    def test_splits_at_line_boundaries(self):
        # Build a snippet where each line is roughly 1 token,
        # and set max_tokens low enough to force splitting.
        lines = [f"x{i}\n" for i in range(20)]
        text = "".join(lines)
        total = count_tokens(text)
        # Force at least 2 chunks
        chunks = split_snippet(text, max_tokens=total // 2)
        assert len(chunks) >= 2
        # Reconstructed text should match original
        assert "".join(chunks) == text

    def test_single_line_exceeding_max(self):
        # A single line that exceeds max_tokens should still be returned
        long_line = "word " * 200
        chunks = split_snippet(long_line, max_tokens=10)
        assert len(chunks) == 1
        assert chunks[0] == long_line

    def test_each_chunk_within_budget(self):
        lines = [f"variable_{i} = {i} + {i*2}\n" for i in range(50)]
        text = "".join(lines)
        max_tok = 30
        chunks = split_snippet(text, max_tokens=max_tok)
        # Every chunk except possibly one with a single oversized line
        # should be within budget
        for chunk in chunks:
            tokens = count_tokens(chunk)
            chunk_lines = chunk.splitlines(keepends=True)
            if len(chunk_lines) > 1:
                assert tokens <= max_tok
