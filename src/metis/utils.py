# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

import codecs
import json
import os
import difflib
import re
import sys

import tiktoken

_WHITESPACE_RE = re.compile(r"\s+")


def safe_decode_unicode(s):
    if isinstance(s, str):
        return codecs.decode(json.dumps(s), "unicode_escape").strip('"')
    return s


_encoding_cache: dict[str, tiktoken.Encoding] = {}


def _get_encoding(model: str = "gpt-4") -> tiktoken.Encoding:
    """Return a cached tiktoken encoder for *model*."""
    enc = _encoding_cache.get(model)
    if enc is None:
        enc = tiktoken.encoding_for_model(model)
        _encoding_cache[model] = enc
    return enc


def count_tokens(text, model="gpt-4"):
    return len(_get_encoding(model).encode(text))


def split_snippet(snippet, max_tokens, model="gpt-4"):
    encoding = _get_encoding(model)
    lines = snippet.splitlines(keepends=True)
    # Encode all lines in one batch to avoid per-line encoder overhead.
    line_token_counts = [len(encoding.encode(line)) for line in lines]

    chunks = []
    current_chunk = ""
    current_token_count = 0

    for line, line_tokens in zip(lines, line_token_counts):
        if current_token_count + line_tokens > max_tokens:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = line
            current_token_count = line_tokens
        else:
            current_chunk += line
            current_token_count += line_tokens

    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def parse_json_output(model_output):
    """
    Clean up and parse model output as JSON.
    """
    cleaned = extract_json_content(model_output)
    try:
        parsed = json.loads(cleaned)
        return parsed
    except Exception:
        return cleaned


def extract_json_content(model_output):
    """
    Extract JSON content from LLM output that may contain explanatory text.
    Handles cases like:
    - Pure JSON
    - JSON wrapped in ```json ... ```
    - JSON embedded in explanatory text
    """
    cleaned = model_output.strip()

    # Remove markdown code blocks first
    if "```json" in cleaned:
        # Extract content between ```json and ```
        start_idx = cleaned.find("```json") + len("```json")
        end_idx = cleaned.find("```", start_idx)
        if end_idx != -1:
            cleaned = cleaned[start_idx:end_idx].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```") :].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[: -len("```")].strip()

    # If still not valid JSON, try to extract JSON object/array from text
    if not cleaned.startswith("{") and not cleaned.startswith("["):
        json_start = -1

        # Find first JSON structure (object or array)
        for i, char in enumerate(cleaned):
            if char == "{" or char == "[":
                json_start = i
                break

        if json_start == -1:
            return cleaned

        # Find matching closing brace/bracket using stack
        stack = []
        json_end = -1

        for i in range(json_start, len(cleaned)):
            char = cleaned[i]
            if char == "{" or char == "[":
                stack.append(char)
            elif char == "}" or char == "]":
                if stack:
                    stack.pop()
                    if not stack:
                        json_end = i + 1
                        break

        if json_end != -1:
            extracted = cleaned[json_start:json_end]
            # Verify it's valid JSON
            try:
                json.loads(extracted)
                return extracted
            except json.JSONDecodeError:
                pass

    return cleaned


def read_file_content(file_path):
    """Read file content if it exists"""
    if not os.path.exists(file_path):
        return ""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def normalize_lines(lines):
    """Remove all whitespace characters from the joined lines."""
    joined = "".join(lines)
    return _WHITESPACE_RE.sub("", joined)


def find_snippet_line(snippet, file_lines, threshold=0.80):
    """
    Finds the first line number where the snippet matches a window in the file
    above the given similarity threshold. Returns 1 if not found.
    Expects caller to provide file_lines to avoid redundant I/O.
    """
    if not file_lines:
        return 1

    snippet_lines = snippet.strip().splitlines()
    snippet_len = len(snippet_lines)
    if snippet_len == 0:
        return 1
    norm_snippet = normalize_lines(snippet_lines)
    if not norm_snippet:
        return 1

    num_windows = len(file_lines) - snippet_len + 1
    if num_windows <= 0:
        return 1

    # Pre-normalize each file line once to avoid redundant work per window.
    norm_file_lines = [_WHITESPACE_RE.sub("", line) for line in file_lines]

    matcher = difflib.SequenceMatcher(None, "", norm_snippet)
    for i in range(num_windows):
        norm_window = "".join(norm_file_lines[i : i + snippet_len])
        matcher.set_seq1(norm_window)
        if matcher.quick_ratio() >= threshold and matcher.ratio() >= threshold:
            return i + 1

    return 1


def retry_on_recursion_error(fn, *args, bump=5000, retries=10, **kwargs):
    """
    Calls `fn(*args, **kwargs)`, catching RecursionError up to `retries` times.
    On each failure, increase the recursion limit by `bump` * `attempt` and retry.
    Restores the original limit before returning.
    """
    original_limit = sys.getrecursionlimit()
    try:
        return fn(*args, **kwargs)
    except RecursionError as e:
        for attempt in range(1, retries + 1):
            new_limit = original_limit + bump * attempt
            sys.setrecursionlimit(new_limit)
            try:
                return fn(*args, **kwargs)
            except RecursionError:
                continue
        raise e
    finally:
        sys.setrecursionlimit(original_limit)


def normalize_severity(value):
    """
    Normalize various textual severity labels to a canonical form.
    Keeps unknown/non-matching values unchanged.
    """
    # Accept only strings; passthrough for other types
    if isinstance(value, str):
        v = value.strip()
        if v:
            # Compare using upper-case to match multiple variants
            upper = v.upper()
            return {
                "LOW": "Low",
                "MED": "Medium",
                "MEDIUM": "Medium",
                "MID": "Medium",
                "HIGH": "High",
                "CRIT": "Critical",
                "CRITICAL": "Critical",
            }.get(upper, v)
    return value


def normalize_issue_fields(issue):
    """
    Ensure issue fields are present and normalized (CWE, severity).
    Mutates and returns the same dict.
    """
    # Default CWE when missing/empty
    issue["cwe"] = issue.get("cwe") if issue.get("cwe") else "CWE-Unknown"
    sev = issue.get("severity")
    if sev is not None:
        issue["severity"] = normalize_severity(sev)
    return issue


def enrich_issues(file_path, issues):
    """
    Enrich issues with derived fields (line_number, normalized CWE/severity).
    Reads the file once and reuses its lines for matching.
    """
    if not issues:
        return issues

    try:
        # Load file content once; matching relies on these lines
        with open(file_path, "r", encoding="utf-8") as _f:
            file_lines = _f.readlines()
    except Exception:
        # If reading fails, line lookup will default to 1
        file_lines = None

    for issue in issues:
        # Only enrich dict-shaped issues; skip plain strings or other types
        if not isinstance(issue, dict):
            continue

        raw_snippet = issue.get("code_snippet", "")
        if isinstance(raw_snippet, list):
            snippet_text = "".join(str(x) for x in raw_snippet)
        elif isinstance(raw_snippet, str):
            snippet_text = raw_snippet
        else:
            snippet_text = str(raw_snippet)
        snippet_text = snippet_text.strip()

        line_number = find_snippet_line(snippet_text, file_lines)
        issue["line_number"] = line_number

        # Normalize and fill other standard fields
        normalize_issue_fields(issue)

    return issues
