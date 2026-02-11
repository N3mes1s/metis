# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for the structured output binding fallback in review.py."""

from unittest.mock import MagicMock

from metis.engine.graphs.review import _bind_structured_output
from metis.engine.graphs.schemas import ReviewResponseModel


class TestBindStructuredOutput:
    def test_function_calling_succeeds(self):
        chat_model = MagicMock()
        sentinel = MagicMock(name="bound-model")
        chat_model.with_structured_output.return_value = sentinel

        result = _bind_structured_output(chat_model, ReviewResponseModel)

        assert result is sentinel
        chat_model.with_structured_output.assert_called_once_with(
            ReviewResponseModel, method="function_calling"
        )

    def test_falls_back_to_json_schema(self):
        chat_model = MagicMock()
        sentinel = MagicMock(name="bound-model")

        def side_effect(model, method):
            if method == "function_calling":
                raise TypeError("not supported")
            return sentinel

        chat_model.with_structured_output.side_effect = side_effect

        result = _bind_structured_output(chat_model, ReviewResponseModel)

        assert result is sentinel
        assert chat_model.with_structured_output.call_count == 2
        # Second call should be json_schema
        second_call = chat_model.with_structured_output.call_args_list[1]
        assert second_call[1]["method"] == "json_schema"

    def test_returns_none_when_both_fail(self):
        chat_model = MagicMock()
        chat_model.with_structured_output.side_effect = TypeError("nope")

        result = _bind_structured_output(chat_model, ReviewResponseModel)

        assert result is None
        assert chat_model.with_structured_output.call_count == 2
