"""Tests for LLM and embedding provider adapters.

Real SDKs are mocked. These tests verify wiring (auth, request shape,
response parsing), not third-party correctness.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from doomscroll.embedding import (
    EmbeddingConfig,
    FakeEmbedder,
    OpenAIEmbedder,
    make_embedder,
)
from doomscroll.llm import (
    AnthropicClient,
    FakeLLM,
    LLMConfig,
    OpenAIClient,
    OpenRouterClient,
    make_llm,
)


# ---------------- FakeLLM ----------------

class TestFakeLLM:
    def test_returns_queued_responses_in_order(self):
        llm = FakeLLM(["first", "second"])
        assert llm.generate("p1") == "first"
        assert llm.generate("p2") == "second"

    def test_returns_empty_when_drained(self):
        llm = FakeLLM(["one"])
        llm.generate("p1")
        assert llm.generate("p2") == ""

    def test_records_calls(self):
        llm = FakeLLM(["x"])
        llm.generate("the prompt")
        assert llm.calls == ["the prompt"]

    def test_queue_method_appends(self):
        llm = FakeLLM()
        llm.queue("late")
        assert llm.generate("p") == "late"


# ---------------- AnthropicClient ----------------

class TestAnthropicClient:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            AnthropicClient(LLMConfig(model="claude-sonnet-4-6"))

    def test_extracts_text_from_response(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_block = MagicMock()
            mock_block.type = "text"
            mock_block.text = "hello world"
            mock_response = MagicMock()
            mock_response.content = [mock_block]
            mock_anthropic.return_value.messages.create.return_value = mock_response

            client = AnthropicClient(LLMConfig(model="claude-sonnet-4-6"))
            result = client.generate("test prompt")
            assert result == "hello world"

    def test_concatenates_multiple_text_blocks(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch("anthropic.Anthropic") as mock_anthropic:
            blocks = []
            for txt in ["part1 ", "part2"]:
                b = MagicMock()
                b.type = "text"
                b.text = txt
                blocks.append(b)
            mock_response = MagicMock()
            mock_response.content = blocks
            mock_anthropic.return_value.messages.create.return_value = mock_response

            client = AnthropicClient(LLMConfig(model="claude-sonnet-4-6"))
            assert client.generate("p") == "part1 part2"

    def test_skips_non_text_blocks(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch("anthropic.Anthropic") as mock_anthropic:
            text_block = MagicMock()
            text_block.type = "text"
            text_block.text = "kept"
            tool_block = MagicMock()
            tool_block.type = "tool_use"
            mock_response = MagicMock()
            mock_response.content = [text_block, tool_block]
            mock_anthropic.return_value.messages.create.return_value = mock_response

            client = AnthropicClient(LLMConfig(model="claude-sonnet-4-6"))
            assert client.generate("p") == "kept"


# ---------------- OpenAIClient ----------------

class TestOpenAIClient:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            OpenAIClient(LLMConfig(model="gpt-4o"))

    def test_returns_message_content(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with patch("openai.OpenAI") as mock_openai:
            choice = MagicMock()
            choice.message.content = "openai response"
            mock_response = MagicMock()
            mock_response.choices = [choice]
            mock_openai.return_value.chat.completions.create.return_value = mock_response

            client = OpenAIClient(LLMConfig(model="gpt-4o"))
            assert client.generate("p") == "openai response"

    def test_handles_null_content(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with patch("openai.OpenAI") as mock_openai:
            choice = MagicMock()
            choice.message.content = None
            mock_response = MagicMock()
            mock_response.choices = [choice]
            mock_openai.return_value.chat.completions.create.return_value = mock_response

            client = OpenAIClient(LLMConfig(model="gpt-4o"))
            assert client.generate("p") == ""


# ---------------- OpenRouterClient ----------------

class TestOpenRouterClient:
    def test_uses_openrouter_env_var(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        with patch("openai.OpenAI") as mock_openai:
            client = OpenRouterClient(LLMConfig(model="anthropic/claude-sonnet-4.6"))
            # Called with OpenRouter base_url and the OR key
            kwargs = mock_openai.call_args.kwargs
            assert kwargs["api_key"] == "or-key"
            assert "openrouter.ai" in kwargs["base_url"]

    def test_missing_openrouter_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            OpenRouterClient(LLMConfig(model="anything"))

    def test_explicit_base_url_wins_over_default(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        with patch("openai.OpenAI") as mock_openai:
            cfg = LLMConfig(model="x", base_url="https://custom.example.com/v1")
            OpenRouterClient(cfg)
            kwargs = mock_openai.call_args.kwargs
            assert kwargs["base_url"] == "https://custom.example.com/v1"


# ---------------- make_llm factory ----------------

class TestMakeLLM:
    def test_fake_no_config_needed(self):
        llm = make_llm("fake")
        assert isinstance(llm, FakeLLM)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="unknown provider"):
            make_llm("nonexistent")

    def test_real_provider_requires_config(self):
        with pytest.raises(ValueError, match="requires an LLMConfig"):
            make_llm("anthropic")


# ---------------- FakeEmbedder ----------------

class TestFakeEmbedder:
    def test_deterministic(self):
        e = FakeEmbedder(dim=32)
        v1 = e.embed("hello")
        v2 = e.embed("hello")
        np.testing.assert_array_equal(v1, v2)

    def test_different_text_different_vector(self):
        e = FakeEmbedder(dim=32)
        v1 = e.embed("hello")
        v2 = e.embed("world")
        assert not np.array_equal(v1, v2)

    def test_unit_norm(self):
        e = FakeEmbedder(dim=64)
        v = e.embed("anything")
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-6)

    def test_correct_dim(self):
        e = FakeEmbedder(dim=128)
        assert e.embed("x").shape == (128,)

    def test_batch(self):
        e = FakeEmbedder(dim=32)
        batch = e.embed_batch(["a", "b", "c"])
        assert batch.shape == (3, 32)
        # First row should match individual embed("a")
        np.testing.assert_array_equal(batch[0], e.embed("a"))

    def test_empty_batch(self):
        e = FakeEmbedder(dim=32)
        batch = e.embed_batch([])
        assert batch.shape == (0, 32)


# ---------------- OpenAIEmbedder ----------------

class TestOpenAIEmbedder:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            OpenAIEmbedder()

    def test_embed_single(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with patch("openai.OpenAI") as mock_openai:
            item = MagicMock()
            item.embedding = [0.1, 0.2, 0.3]
            mock_response = MagicMock()
            mock_response.data = [item]
            mock_openai.return_value.embeddings.create.return_value = mock_response

            embedder = OpenAIEmbedder()
            v = embedder.embed("text")
            np.testing.assert_allclose(v, [0.1, 0.2, 0.3])

    def test_embed_batch_preserves_order(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with patch("openai.OpenAI") as mock_openai:
            items = []
            for vec in [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]:
                item = MagicMock()
                item.embedding = vec
                items.append(item)
            mock_response = MagicMock()
            mock_response.data = items
            mock_openai.return_value.embeddings.create.return_value = mock_response

            embedder = OpenAIEmbedder()
            batch = embedder.embed_batch(["a", "b", "c"])
            assert batch.shape == (3, 2)
            np.testing.assert_allclose(batch[0], [1.0, 0.0])
            np.testing.assert_allclose(batch[2], [1.0, 1.0])

    def test_empty_batch_no_api_call(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with patch("openai.OpenAI") as mock_openai:
            embedder = OpenAIEmbedder(EmbeddingConfig(dim=8))
            result = embedder.embed_batch([])
            assert result.shape == (0, 8)
            mock_openai.return_value.embeddings.create.assert_not_called()


class TestMakeEmbedder:
    def test_fake_factory(self):
        assert isinstance(make_embedder("fake"), FakeEmbedder)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown embedding provider"):
            make_embedder("nope")
