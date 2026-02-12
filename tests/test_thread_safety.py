# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Thread-safety tests for ReviewGraph caches under concurrent load.

Covers the test-plan item:
    'Spot-check thread safety under concurrent load'
"""

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock

from metis.engine.graphs.review import ReviewGraph, _bind_structured_output


def _make_review_graph():
    """Build a ReviewGraph with a mocked LLM provider."""
    llm_provider = MagicMock()
    chat_model = MagicMock()
    # Make with_structured_output work
    chat_model.with_structured_output.return_value = MagicMock()
    llm_provider.get_chat_model.return_value = chat_model

    from metis.configuration import load_plugin_config

    plugin_config = load_plugin_config()

    return ReviewGraph(
        llm_provider=llm_provider,
        plugin_config=plugin_config,
        custom_prompt_text=None,
        custom_guidance_precedence="",
        llama_query_model="gpt-test",
        max_token_length=2048,
    )


class TestAppCacheThreadSafety:
    def test_concurrent_build_app_calls(self):
        """Multiple threads calling _build_app with the same prompts should
        not corrupt the cache or raise errors."""
        rg = _make_review_graph()
        prompts = {"security_review_file": "Review [[REVIEW_SCHEMA_FIELDS]]", "security_review_checks": "Checks"}
        errors = []

        def worker():
            try:
                app = rg._build_app(prompts, "security_review_file")
                assert app is not None
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Errors during concurrent _build_app: {errors}"

    def test_concurrent_build_app_different_keys(self):
        """Concurrent calls with different cache keys should all succeed."""
        rg = _make_review_graph()
        results = {}
        errors = []

        def worker(idx):
            try:
                # Each thread uses a distinct prompt dict (different id())
                prompts = {
                    "security_review_file": f"Review {idx} [[REVIEW_SCHEMA_FIELDS]]",
                    "security_review_checks": "Checks",
                }
                app = rg._build_app(prompts, "security_review_file")
                results[idx] = app
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Errors: {errors}"
        assert len(results) == 10


class TestBatchRunnableThreadSafety:
    def test_concurrent_batch_init(self):
        """Multiple threads triggering _init_batch_runnables should not
        cause double-initialization or errors."""
        rg = _make_review_graph()
        init_count = {"n": 0}
        original_init = rg._init_batch_runnables

        def counting_init():
            init_count["n"] += 1
            original_init()

        rg._init_batch_runnables = counting_init
        errors = []

        def worker():
            try:
                if not rg._batch_runnables_ready:
                    with rg._batch_init_lock:
                        if not rg._batch_runnables_ready:
                            rg._init_batch_runnables()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Errors: {errors}"
        # Should only have been initialized once despite 20 threads
        assert init_count["n"] == 1


class TestBindStructuredOutputThreadSafety:
    def test_concurrent_bind_calls(self):
        """_bind_structured_output should be safe to call from multiple threads."""
        chat_model = MagicMock()
        sentinel = MagicMock(name="bound")
        chat_model.with_structured_output.return_value = sentinel
        errors = []
        results = []

        def worker():
            try:
                from metis.engine.graphs.schemas import ReviewResponseModel
                result = _bind_structured_output(chat_model, ReviewResponseModel)
                results.append(result)
            except Exception as exc:
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(worker) for _ in range(20)]
            for f in as_completed(futures):
                f.result()

        assert errors == [], f"Errors: {errors}"
        assert all(r is sentinel for r in results)
