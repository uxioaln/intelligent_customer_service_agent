# -*- coding: utf-8 -*-
"""
tests.test_tiered_validation - 验证器分层判定单元测试

覆盖目标：app/agent/validator.py
- _should_skip_claim_check：三维全 pass + vote_samples>=2 返回 True，否则 False
- _skipped_claim_result：占位结果字段完整性
- validate_answer（mock LLM）：
  - 三维全 pass 时不调用 claim 级判定，返回 claim_skipped=True
  - 任一维度非 pass 时仍调用 claim 级判定
  - ENABLE_TIERED_VALIDATION=False 时不论维度结果都调用 claim 级判定
"""
import sys
import os
from unittest.mock import MagicMock, patch

# 把工程根加入 path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app.agent import validator
from app.agent.validator import (
    _should_skip_claim_check,
    _skipped_claim_result,
    validate_answer,
)


# ========== 工具：构造维度打分结果 ==========

def _dim_result(acc="pass", cit="pass", rea="pass", vote_samples=3, overall=1.0):
    return {
        "accuracy": {"grade": acc, "score": 1.0 if acc == "pass" else 0.5, "reason": "r"},
        "citation": {"grade": cit, "score": 1.0 if cit == "pass" else 0.5, "reason": "r"},
        "reasoning": {"grade": rea, "score": 1.0 if rea == "pass" else 0.5, "reason": "r"},
        "overall": overall,
        "passed": overall >= 0.6,
        "vote_samples": vote_samples,
    }


# ========== _should_skip_claim_check ==========

class TestShouldSkipClaimCheck:
    def test_all_pass_with_sufficient_votes_returns_true(self):
        """三维全 pass + vote_samples=3 → 跳过"""
        assert _should_skip_claim_check(_dim_result()) is True

    def test_all_pass_with_min_votes_returns_true(self):
        """vote_samples=2（最小阈值）→ 跳过"""
        assert _should_skip_claim_check(_dim_result(vote_samples=2)) is True

    def test_insufficient_votes_returns_false(self):
        """vote_samples=1 → 不跳过"""
        assert _should_skip_claim_check(_dim_result(vote_samples=1)) is False

    def test_zero_votes_returns_false(self):
        """vote_samples=0 → 不跳过"""
        assert _should_skip_claim_check(_dim_result(vote_samples=0)) is False

    def test_accuracy_marginal_returns_false(self):
        """accuracy=marginal → 不跳过"""
        assert _should_skip_claim_check(_dim_result(acc="marginal")) is False

    def test_citation_fail_returns_false(self):
        """citation=fail → 不跳过"""
        assert _should_skip_claim_check(_dim_result(cit="fail", overall=0.67)) is False

    def test_reasoning_marginal_returns_false(self):
        """reasoning=marginal → 不跳过"""
        assert _should_skip_claim_check(_dim_result(rea="marginal")) is False

    def test_missing_dimension_returns_false(self):
        """缺少维度字段 → 不跳过（防御性）"""
        result = {"vote_samples": 3, "accuracy": {"grade": "pass"}}
        assert _should_skip_claim_check(result) is False


# ========== _skipped_claim_result ==========

class TestSkippedClaimResult:
    def test_fields_complete(self):
        """占位结果字段完整且语义安全"""
        r = _skipped_claim_result("test reason")
        assert r["claims"] == []
        assert r["claim_count"] == 0
        assert r["unsupported_count"] == 0
        assert r["claim_hallucination_rate"] == 0.0
        assert r["claim_level_hallucination"] is False
        assert r["claim_skipped"] is True
        assert r["claim_skipped_reason"] == "test reason"
        # 与失败占位区分：没有 claim_error 字段
        assert "claim_error" not in r


# ========== validate_answer 分层路径（mock LLM） ==========

def _make_dim_llm_response(grade="pass"):
    """构造维度打分 LLM 返回的 JSON 字符串"""
    import json
    return json.dumps({
        "accuracy": {"grade": grade, "reason": "r"},
        "citation": {"grade": grade, "reason": "r"},
        "reasoning": {"grade": grade, "reason": "r"},
    })


def _make_claim_llm_response(supported=True):
    import json
    return json.dumps({
        "claims": [{"text": "c1", "supported": supported, "reason": "r"}]
    })


def _mock_tracked_chat(content_seq):
    """构造一个 tracked_chat_completion 的 mock，按顺序返回不同 content"""
    from types import SimpleNamespace
    side_effects = []
    for c in content_seq:
        msg = SimpleNamespace(content=c)
        choice = SimpleNamespace(message=msg)
        resp = SimpleNamespace(choices=[choice], usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
        side_effects.append(resp)
    return MagicMock(side_effect=side_effects)


class TestValidateAnswerTiered:
    """validate_answer 的分层判定路径"""

    def setup_method(self):
        """每个测试前确保开关默认开启"""
        validator.ENABLE_TIERED_VALIDATION = True

    def teardown_method(self):
        validator.ENABLE_TIERED_VALIDATION = True

    def test_all_pass_skips_claim_check(self):
        """三维全 pass：3 次维度采样 + 0 次 claim 调用，返回 claim_skipped=True"""
        references = [{"source": "s", "content": "c"}]
        # 3 次维度采样全 pass，无 claim 调用
        mock_fn = _mock_tracked_chat([_make_dim_llm_response("pass")] * 3)
        with patch("app.agent.validator.tracked_chat_completion", mock_fn):
            result = validate_answer(references, "answer", tracker=None)

        assert mock_fn.call_count == 3  # 只调用了 3 次维度采样
        assert result["passed"] is True
        assert result["overall"] == 1.0
        assert result["claim_skipped"] is True
        assert result["claim_count"] == 0
        assert result["claim_level_hallucination"] is False

    def test_marginal_grade_triggers_claim_check(self):
        """任一维度 marginal：3 次维度采样 + 1 次 claim 调用"""
        references = [{"source": "s", "content": "c"}]
        # 3 次维度采样：第一次 marginal，后两次 pass → 众数仍可能 pass
        # 这里让所有 3 次都返回 marginal，保证众数是 marginal
        contents = [_make_dim_llm_response("marginal")] * 3 + [_make_claim_llm_response(supported=True)]
        mock_fn = _mock_tracked_chat(contents)
        with patch("app.agent.validator.tracked_chat_completion", mock_fn):
            result = validate_answer(references, "answer", tracker=None)

        assert mock_fn.call_count == 4  # 3 次维度 + 1 次 claim
        assert "claim_skipped" not in result or result.get("claim_skipped") is False
        assert result["claim_count"] == 1

    def test_tiered_disabled_always_calls_claim(self):
        """ENABLE_TIERED_VALIDATION=False：即使全 pass 也调用 claim 级判定"""
        validator.ENABLE_TIERED_VALIDATION = False
        references = [{"source": "s", "content": "c"}]
        contents = [_make_dim_llm_response("pass")] * 3 + [_make_claim_llm_response(supported=True)]
        mock_fn = _mock_tracked_chat(contents)
        with patch("app.agent.validator.tracked_chat_completion", mock_fn):
            result = validate_answer(references, "answer", tracker=None)

        assert mock_fn.call_count == 4  # 3 次维度 + 1 次 claim
        assert result.get("claim_skipped") is not True
        assert result["claim_count"] == 1

    def test_low_vote_samples_triggers_claim_check(self):
        """vote_samples<2（部分采样解析失败）：仍走 claim 级判定"""
        references = [{"source": "s", "content": "c"}]
        # 3 次维度采样：1 次有效 pass + 2 次非法 JSON（解析失败） + 1 次 claim
        contents = [_make_dim_llm_response("pass"), "not a json", "also not json", _make_claim_llm_response(True)]
        mock_fn = _mock_tracked_chat(contents)
        with patch("app.agent.validator.tracked_chat_completion", mock_fn):
            result = validate_answer(references, "answer", tracker=None)

        assert mock_fn.call_count == 4
        assert result["vote_samples"] == 1  # 只 1 次解析成功
        # vote_samples<2 → 不跳过
        assert result.get("claim_skipped") is not True
        assert result["claim_count"] == 1
