# -*- coding: utf-8 -*-
"""
tests.test_llm_guard - LLM 网关空响应防护与重试机制单元测试

覆盖目标：app/core/cost_tracker.py
- _validate_response_not_empty：三层空响应校验
- _is_retryable_exception：429/5xx 可重试，4xx 不重试
- tracked_chat_completion：
  - 空响应重试成功路径
  - 空响应重试耗尽抛 LLMEmptyResponseError
  - 不可重试异常直接抛出
  - 可重试异常（RateLimitError）重试成功
  - 正常路径不触发重试
"""
import sys
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# 把工程根加入 path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app.core.cost_tracker import (
    LLMEmptyResponseError,
    _is_retryable_exception,
    _validate_response_not_empty,
    tracked_chat_completion,
)


# ========== 测试用 Fake Response 构造 ==========

def _make_response(content="hello"):
    """构造一个最小可用的 OpenAI 风格 response 对象"""
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    return SimpleNamespace(choices=[choice], usage=usage)


def _make_client(responses):
    """构造一个 mock client，按顺序抛出异常或返回 response

    responses: list，元素可以是 response 对象或 Exception 实例（抛出）
    """
    client = MagicMock()
    side_effects = []
    for r in responses:
        if isinstance(r, Exception):
            side_effects.append(r)
        else:
            side_effects.append(r)
    client.chat.completions.create = MagicMock(side_effect=side_effects)
    return client


# ========== _validate_response_not_empty ==========

class TestValidateResponseNotEmpty:
    def test_valid_response_passes(self):
        """正常响应不抛异常"""
        resp = _make_response("ok")
        _validate_response_not_empty(resp, "gpt-4o", "test")  # 不应抛异常

    def test_none_response_raises(self):
        """response 为 None 时抛 LLMEmptyResponseError"""
        with pytest.raises(LLMEmptyResponseError, match="response is None"):
            _validate_response_not_empty(None, "gpt-4o", "test")

    def test_empty_choices_raises(self):
        """choices 为空 list 时抛 LLMEmptyResponseError"""
        resp = SimpleNamespace(choices=[], usage={})
        with pytest.raises(LLMEmptyResponseError, match="空 choices"):
            _validate_response_not_empty(resp, "gpt-4o", "test")

    def test_choices_none_raises(self):
        """choices 为 None 时抛 LLMEmptyResponseError"""
        resp = SimpleNamespace(choices=None, usage={})
        with pytest.raises(LLMEmptyResponseError, match="空 choices"):
            _validate_response_not_empty(resp, "gpt-4o", "test")

    def test_empty_message_content_raises(self):
        """message.content 为 None 时抛 LLMEmptyResponseError"""
        resp = _make_response(content=None)
        with pytest.raises(LLMEmptyResponseError, match="空 message.content"):
            _validate_response_not_empty(resp, "gpt-4o", "test")

    def test_empty_string_content_passes(self):
        """content 为空字符串也算有效（LLM 可能真的返回空字符串），不应抛"""
        resp = _make_response(content="")
        _validate_response_not_empty(resp, "gpt-4o", "test")  # 不应抛


# ========== _is_retryable_exception ==========

class TestIsRetryableException:
    def test_generic_exception_not_retryable(self):
        """普通 Exception 不可重试"""
        assert _is_retryable_exception(ValueError("bad")) is False
        assert _is_retryable_exception(KeyError("k")) is False
        assert _is_retryable_exception(RuntimeError("x")) is False

    def test_openai_rate_limit_retryable(self):
        """openai.RateLimitError 可重试"""
        try:
            from openai import RateLimitError
            # RateLimitError 构造签名在不同 openai 版本有差异，用 try 构造
            try:
                exc = RateLimitError("rate limit", response=MagicMock(status_code=429), body=None)
            except TypeError:
                exc = RateLimitError("rate limit")
            assert _is_retryable_exception(exc) is True
        except ImportError:
            pytest.skip("openai 未安装")

    def test_openai_api_error_4xx_not_retryable(self):
        """openai.APIError 4xx（除 429）不可重试"""
        try:
            from openai import APIError
            import httpx
        except ImportError:
            pytest.skip("openai/httpx 未安装")
        # 新版 openai APIError 需要 request 参数
        mock_request = httpx.Request("POST", "https://api.example.com/v1/chat")
        exc = APIError("bad request", request=mock_request, body=None)
        # 确保 status_code 被设置（不同版本赋值方式不同）
        if not hasattr(exc, "status_code") or exc.status_code is None:
            exc.status_code = 400
        assert _is_retryable_exception(exc) is False


# ========== tracked_chat_completion ==========

class TestTrackedChatCompletionGuard:
    """tracked_chat_completion 的空响应防护与重试"""

    def test_normal_response_no_retry(self):
        """正常响应一次成功，不触发重试"""
        client = _make_client([_make_response("answer")])
        resp = tracked_chat_completion(
            client=client, model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            source="test",
        )
        assert resp.choices[0].message.content == "answer"
        assert client.chat.completions.create.call_count == 1

    def test_empty_response_retry_then_success(self):
        """空响应触发重试，第二次成功"""
        empty_resp = SimpleNamespace(choices=[], usage={})
        good_resp = _make_response("recovered")
        client = _make_client([empty_resp, good_resp])
        # 加速重试：retry_base_delay=0
        resp = tracked_chat_completion(
            client=client, model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            source="test",
            retry_base_delay=0,
        )
        assert resp.choices[0].message.content == "recovered"
        assert client.chat.completions.create.call_count == 2

    def test_empty_response_retry_exhausted(self):
        """空响应重试耗尽后抛 LLMEmptyResponseError"""
        empty_resp = SimpleNamespace(choices=[], usage={})
        client = _make_client([empty_resp, empty_resp, empty_resp, empty_resp])
        with pytest.raises(LLMEmptyResponseError):
            tracked_chat_completion(
                client=client, model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
                source="test",
                max_retries=3,
                retry_base_delay=0,
            )
        # 1 次原始 + 3 次重试 = 4 次调用
        assert client.chat.completions.create.call_count == 4

    def test_none_response_retry_exhausted(self):
        """response 为 None 也走重试路径"""
        client = _make_client([None, None])
        with pytest.raises(LLMEmptyResponseError, match="response is None"):
            tracked_chat_completion(
                client=client, model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
                source="test",
                max_retries=1,
                retry_base_delay=0,
            )

    def test_non_retryable_exception_raises_immediately(self):
        """不可重试异常（ValueError）直接抛出，不重试"""
        client = _make_client([ValueError("bad params")])
        with pytest.raises(ValueError, match="bad params"):
            tracked_chat_completion(
                client=client, model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
                source="test",
                retry_base_delay=0,
            )
        assert client.chat.completions.create.call_count == 1

    def test_retryable_exception_succeeds_after_retry(self):
        """可重试异常（mock 一个模拟 429 的异常）重试后成功"""
        try:
            from openai import RateLimitError
            mock_resp = MagicMock()
            mock_resp.status_code = 429
            try:
                rate_exc = RateLimitError("rate limit", response=mock_resp, body=None)
            except TypeError:
                rate_exc = RateLimitError("rate limit")
        except ImportError:
            pytest.skip("openai 未安装")

        client = _make_client([rate_exc, _make_response("ok")])
        resp = tracked_chat_completion(
            client=client, model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            source="test",
            retry_base_delay=0,
        )
        assert resp.choices[0].message.content == "ok"
        assert client.chat.completions.create.call_count == 2

    def test_backoff_delay_increases_exponentially(self):
        """重试间隔按 1s/2s/4s 指数退避（通过 patch time.sleep 验证）"""
        empty_resp = SimpleNamespace(choices=[], usage={})
        client = _make_client([empty_resp, empty_resp, empty_resp, _make_response("ok")])
        sleep_calls = []
        with patch("app.core.cost_tracker.time.sleep", side_effect=lambda s: sleep_calls.append(s)):
            tracked_chat_completion(
                client=client, model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
                source="test",
                max_retries=3,
                retry_base_delay=1.0,
            )
        # 3 次重试的延迟应为 1, 2, 4
        assert sleep_calls == [1.0, 2.0, 4.0]

    def test_cost_recorded_only_on_success(self):
        """重试失败的调用不应记录 token 成本（无 usage 可记）"""
        from app.core.cost_tracker import CostTracker
        tracker = CostTracker()
        empty_resp = SimpleNamespace(choices=[], usage={})
        client = _make_client([empty_resp, _make_response("ok")])
        tracked_chat_completion(
            client=client, model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            source="test",
            tracker=tracker,
            retry_base_delay=0,
        )
        # 只有成功的 1 次被记录
        assert len(tracker.calls) == 1
        assert tracker.calls[0]["total_tokens"] == 15
