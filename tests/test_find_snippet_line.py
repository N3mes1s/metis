# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for the optimized find_snippet_line function."""

from metis.utils import find_snippet_line


class TestFindSnippetLine:
    def test_exact_match(self):
        file_lines = ["int a = 1;\n", "int b = 2;\n", "int c = 3;\n"]
        assert find_snippet_line("int b = 2;", file_lines) == 2

    def test_exact_match_multiline(self):
        file_lines = ["a\n", "b\n", "c\n", "d\n"]
        assert find_snippet_line("b\nc\n", file_lines) == 2

    def test_fuzzy_match(self):
        file_lines = ["int   a =  1;\n", "int b=2;\n", "int c = 3;\n"]
        # Whitespace is stripped during normalization, so "int b = 2;" matches "int b=2;"
        assert find_snippet_line("int b = 2;", file_lines) == 2

    def test_no_match_returns_1(self):
        file_lines = ["int a = 1;\n", "int b = 2;\n"]
        assert find_snippet_line("totally_different_code();", file_lines) == 1

    def test_empty_file_lines(self):
        assert find_snippet_line("anything", []) == 1
        assert find_snippet_line("anything", None) == 1

    def test_empty_snippet(self):
        file_lines = ["a\n", "b\n"]
        assert find_snippet_line("", file_lines) == 1
        assert find_snippet_line("   \n  \n", file_lines) == 1

    def test_snippet_at_start(self):
        file_lines = ["first\n", "second\n", "third\n"]
        assert find_snippet_line("first", file_lines) == 1

    def test_snippet_at_end(self):
        file_lines = ["first\n", "second\n", "third\n"]
        assert find_snippet_line("third", file_lines) == 3

    def test_snippet_longer_than_file(self):
        file_lines = ["a\n"]
        assert find_snippet_line("a\nb\nc\n", file_lines) == 1

    def test_returns_first_match(self):
        file_lines = ["dup\n", "x\n", "dup\n"]
        assert find_snippet_line("dup", file_lines) == 1
