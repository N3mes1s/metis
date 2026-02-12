# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for use_responses_api config propagation in configuration.py."""

from unittest.mock import patch

import pytest

from metis.configuration import load_runtime_config


def _mock_config(provider_name, extra_llm=None):
    """Build a minimal metis config dict for the given provider."""
    llm = {"name": provider_name, "model": "test-model"}
    if extra_llm:
        llm.update(extra_llm)
    return {"llm_provider": llm, "metis_engine": {}, "query": {}}


class TestResponsesApiConfigOpenAI:
    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"})
    @patch("metis.configuration.load_metis_config")
    def test_propagated_when_true(self, mock_cfg):
        mock_cfg.return_value = _mock_config("openai", {"use_responses_api": True})
        rt = load_runtime_config()
        assert rt["use_responses_api"] is True

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"})
    @patch("metis.configuration.load_metis_config")
    def test_propagated_when_false(self, mock_cfg):
        mock_cfg.return_value = _mock_config("openai", {"use_responses_api": False})
        rt = load_runtime_config()
        assert rt["use_responses_api"] is False

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"})
    @patch("metis.configuration.load_metis_config")
    def test_omitted_when_not_set(self, mock_cfg):
        mock_cfg.return_value = _mock_config("openai")
        rt = load_runtime_config()
        assert "use_responses_api" not in rt


class TestResponsesApiConfigVLLM:
    @patch("metis.configuration.load_metis_config")
    def test_propagated_when_true(self, mock_cfg):
        mock_cfg.return_value = _mock_config(
            "vllm",
            {"api_key": "k", "base_url": "http://localhost:8000/v1", "use_responses_api": True},
        )
        rt = load_runtime_config()
        assert rt["use_responses_api"] is True

    @patch("metis.configuration.load_metis_config")
    def test_omitted_when_not_set(self, mock_cfg):
        mock_cfg.return_value = _mock_config(
            "vllm", {"api_key": "k", "base_url": "http://localhost:8000/v1"},
        )
        rt = load_runtime_config()
        assert "use_responses_api" not in rt


class TestResponsesApiConfigOllama:
    @patch("metis.configuration.load_metis_config")
    def test_propagated_when_true(self, mock_cfg):
        mock_cfg.return_value = _mock_config(
            "ollama", {"use_responses_api": True},
        )
        rt = load_runtime_config()
        assert rt["use_responses_api"] is True

    @patch("metis.configuration.load_metis_config")
    def test_omitted_when_not_set(self, mock_cfg):
        mock_cfg.return_value = _mock_config("ollama")
        rt = load_runtime_config()
        assert "use_responses_api" not in rt
