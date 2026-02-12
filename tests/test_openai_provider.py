# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest

from metis.providers.openai_compatible import (
    OpenAICompatibleProvider,
    _RESPONSES_ONLY_MODELS,
)


def _make_provider(**overrides):
    config = {
        "llm_api_key": "test-key",
        "model": "gpt-4o",
        "llama_query_model": "gpt-4o",
        "llama_query_temperature": 0.0,
        "llama_query_max_tokens": 512,
    }
    config.update(overrides)
    return OpenAICompatibleProvider(config)


# -- _needs_responses_api -----------------------------------------------------


class TestNeedsResponsesApi:
    def test_returns_false_for_normal_model(self):
        provider = _make_provider()
        assert provider._needs_responses_api("gpt-4o") is False

    def test_returns_true_for_codex_model(self):
        provider = _make_provider()
        for model in _RESPONSES_ONLY_MODELS:
            assert provider._needs_responses_api(model) is True

    def test_explicit_true_overrides_model_check(self):
        provider = _make_provider(use_responses_api=True)
        assert provider._needs_responses_api("gpt-4o") is True

    def test_explicit_false_overrides_model_check(self):
        provider = _make_provider(use_responses_api=False)
        for model in _RESPONSES_ONLY_MODELS:
            assert provider._needs_responses_api(model) is False

    def test_none_falls_through_to_model_check(self):
        provider = _make_provider(use_responses_api=None)
        assert provider._needs_responses_api("gpt-4o") is False
        assert provider._needs_responses_api("gpt-5.2-codex") is True


# -- get_chat_model ------------------------------------------------------------


class TestGetChatModel:
    @patch("metis.providers.openai_compatible.ChatOpenAI")
    def test_passes_responses_api_for_codex_model(self, mock_chat):
        provider = _make_provider()
        provider.get_chat_model(model="gpt-5.2-codex")
        call_kwargs = mock_chat.call_args[1]
        assert call_kwargs["use_responses_api"] is True

    @patch("metis.providers.openai_compatible.ChatOpenAI")
    def test_omits_responses_api_for_normal_model(self, mock_chat):
        provider = _make_provider()
        provider.get_chat_model(model="gpt-4o")
        call_kwargs = mock_chat.call_args[1]
        assert "use_responses_api" not in call_kwargs

    @patch("metis.providers.openai_compatible.ChatOpenAI")
    def test_explicit_config_overrides_model(self, mock_chat):
        provider = _make_provider(use_responses_api=True)
        provider.get_chat_model(model="gpt-4o")
        call_kwargs = mock_chat.call_args[1]
        assert call_kwargs["use_responses_api"] is True

    @patch("metis.providers.openai_compatible.ChatOpenAI")
    def test_explicit_false_suppresses_for_codex(self, mock_chat):
        provider = _make_provider(use_responses_api=False)
        provider.get_chat_model(model="gpt-5.3-codex")
        call_kwargs = mock_chat.call_args[1]
        assert "use_responses_api" not in call_kwargs
