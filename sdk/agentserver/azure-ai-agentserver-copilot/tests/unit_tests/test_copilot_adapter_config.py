# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Unit tests for the BYOK / Managed Identity config builder."""

import os
from unittest.mock import MagicMock, patch

import pytest

from azure.ai.agentserver.copilot.copilot_adapter import CopilotAdapter, _build_session_config


@pytest.mark.unit
class TestBuildSessionConfig:
    """Tests for _build_session_config()."""

    def test_default_github_copilot_model(self):
        """Without AZURE_AI_FOUNDRY_RESOURCE_URL, uses default GitHub Copilot model."""
        env = {"COPILOT_MODEL": ""}
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("AZURE_AI_FOUNDRY_RESOURCE_URL", None)
            os.environ.pop("COPILOT_MODEL", None)

            config = _build_session_config()
            assert config["model"] == "gpt-5"
            assert "provider" not in config

    def test_custom_github_model(self):
        """COPILOT_MODEL overrides default when not in BYOK mode."""
        with patch.dict(os.environ, {"COPILOT_MODEL": "gpt-4.1"}, clear=False):
            os.environ.pop("AZURE_AI_FOUNDRY_RESOURCE_URL", None)

            config = _build_session_config()
            assert config["model"] == "gpt-4.1"

    def test_byok_foundry_mode(self):
        """With AZURE_AI_FOUNDRY_RESOURCE_URL, builds BYOK ProviderConfig with placeholder token."""
        env = {
            "AZURE_AI_FOUNDRY_RESOURCE_URL": "https://myresource.openai.azure.com",
            "COPILOT_MODEL": "gpt-4.1",
        }
        with patch.dict(os.environ, env, clear=False):
            config = _build_session_config()

        assert config["model"] == "gpt-4.1"
        assert config["provider"]["type"] == "openai"
        assert config["provider"]["base_url"] == "https://myresource.openai.azure.com/openai/v1/"
        assert config["provider"]["bearer_token"] == "placeholder"
        assert config["provider"]["wire_api"] == "responses"

    @patch("azure.identity.DefaultAzureCredential")
    def test_byok_trailing_slash_stripped(self, mock_cred_cls):
        """Trailing slash on resource URL is handled correctly."""
        mock_cred = MagicMock()
        mock_cred.get_token.return_value = MagicMock(token="t")
        mock_cred_cls.return_value = mock_cred

        env = {
            "AZURE_AI_FOUNDRY_RESOURCE_URL": "https://myresource.openai.azure.com/",
        }
        with patch.dict(os.environ, env, clear=False):
            config = _build_session_config()

        assert config["provider"]["base_url"] == "https://myresource.openai.azure.com/openai/v1/"

    @patch("azure.identity.DefaultAzureCredential")
    def test_byok_default_model(self, mock_cred_cls):
        """BYOK mode defaults to gpt-4.1 when COPILOT_MODEL is not set."""
        mock_cred = MagicMock()
        mock_cred.get_token.return_value = MagicMock(token="t")
        mock_cred_cls.return_value = mock_cred

        env = {
            "AZURE_AI_FOUNDRY_RESOURCE_URL": "https://myresource.openai.azure.com",
        }
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("COPILOT_MODEL", None)

            config = _build_session_config()

        assert config["model"] == "gpt-4.1"


@pytest.mark.unit
class TestTokenRefresh:
    """Tests for _refresh_token_if_needed()."""

    @pytest.mark.asyncio(loop_scope="function")
    async def test_refresh_updates_bearer_token(self):
        """Token refresh replaces the bearer_token in the provider config."""
        adapter = CopilotAdapter.__new__(CopilotAdapter)
        adapter._session_config = {
            "model": "gpt-4.1",
            "provider": {
                "type": "openai",
                "base_url": "https://x.openai.azure.com/openai/v1/",
                "bearer_token": "old-token",
                "wire_api": "responses",
            },
        }
        mock_cred = MagicMock()
        mock_cred.get_token.return_value = MagicMock(token="new-token-456")
        adapter._credential = mock_cred

        config = await adapter._refresh_token_if_needed()

        assert config["provider"]["bearer_token"] == "new-token-456"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_no_refresh_without_credential(self):
        """Without a credential (non-BYOK mode), config is returned as-is."""
        adapter = CopilotAdapter.__new__(CopilotAdapter)
        adapter._session_config = {"model": "gpt-5"}
        adapter._credential = None

        config = await adapter._refresh_token_if_needed()
        assert config == {"model": "gpt-5"}
