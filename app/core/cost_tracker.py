# -*- coding: utf-8 -*-
"""
app.core.cost_tracker - 成本追踪模块

功能：包装大模型 chat 调用，记录每次调用的起止时间、token 用量与估算成本。
所有调用大模型的地方建议统一使用 tracked_chat_completion，便于集中统计。

内建防护：
- LLM 网关空响应防护（HTTP 200 但 choices 为空的瞬时故障，AGICTO 平台偶发）
- 限流 / 5xx 指数退避重试（默认最多 3 次，间隔 1s/2s/4s）
- 空响应最后一次仍失败时抛出 LLMEmptyResponseError，由调用方捕获处理
"""
import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.config import COST_STATS_FILE

logger = logging.getLogger(__name__)


class LLMEmptyResponseError(Exception):
    """LLM 网关返回空响应（HTTP 200 但 choices 为空或 response 为 None）"""
    pass


# 重试配置：默认最多重试 3 次，指数退避 1s/2s/4s
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BASE_DELAY = 1.0  # 秒，第 n 次重试等待 base * 2^(n-1)

# 模型单价表（单位：元 / 1K tokens）
# 价格按当前平台常见价格估算，后续如有调整直接改这里即可
MODEL_PRICING = {
    # AGICTO 平台
    "deepseek-v4-flash": {"input": 0.0001, "output": 0.0004},
    "deepseek-v4-pro": {"input": 0.0004, "output": 0.0016},
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    # DASHSCOPE 平台（OpenAI 兼容模式）
    "qwen-flash": {"input": 0.0001, "output": 0.0004},
    "qwen-plus": {"input": 0.0008, "output": 0.0020},
    # 默认兜底
    "default": {"input": 0.0001, "output": 0.0004},
}


def estimate_cost(prompt_tokens, completion_tokens, model):
    """根据 token 数和模型单价估算成本（单位：元）"""
    price = MODEL_PRICING.get(model, MODEL_PRICING["default"])
    cost = prompt_tokens * price["input"] / 1000 + completion_tokens * price["output"] / 1000
    return round(cost, 6)


class CostTracker:
    """单次任务的成本追踪器，可累加多次 LLM 调用记录。"""

    def __init__(self):
        self.calls = []
        self.start_time = None
        self.end_time = None

    def begin(self):
        """记录任务开始时间"""
        self.start_time = time.time()

    def finish(self):
        """记录任务结束时间"""
        self.end_time = time.time()

    def add_call(self, call_record):
        """添加一次调用记录"""
        self.calls.append(call_record)

    def get_summary(self):
        """返回当前累计统计"""
        prompt_tokens = sum(c.get("prompt_tokens", 0) for c in self.calls)
        completion_tokens = sum(c.get("completion_tokens", 0) for c in self.calls)
        total_tokens = sum(c.get("total_tokens", 0) for c in self.calls)
        estimated_cost = round(sum(c.get("estimated_cost", 0) for c in self.calls), 6)

        return {
            "call_count": len(self.calls),
            "calls": self.calls,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost": estimated_cost,
            "cost_unit": "元",
            "cost_note": "按平台典型单价估算，仅供参考",
            "start_time": datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
            "end_time": datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
            "duration_seconds": round(self.end_time - self.start_time, 2) if self.start_time and self.end_time else 0,
        }

    def reset(self):
        """清空记录，用于新一轮任务"""
        self.calls = []
        self.start_time = None
        self.end_time = None


class GlobalStats:
    """
    全局持久化统计器：累计所有 LLM 调用的调用次数、运行时间和成本。
    按模块（问答/健康检查/索引构建）分类统计，结果写入 cost_stats.json。
    """

    def __init__(self, filepath=COST_STATS_FILE):
        self.filepath = filepath
        self._lock = threading.Lock()
        self._data = self._load()

    def _load(self):
        """从文件加载统计"""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "by_module": {},
            "last_reset": datetime.now().isoformat(),
        }

    def _save(self):
        """保存统计到文件"""
        try:
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("保存全局统计失败: %s", e)

    def record_call(self, source, duration_seconds, prompt_tokens, completion_tokens, estimated_cost):
        """记录一次 LLM 调用"""
        with self._lock:
            module = self._data.setdefault("by_module", {}).setdefault(source, {
                "call_count": 0,
                "total_duration_seconds": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_cost": 0.0,
                "cost_unit": "元",
            })
            module["call_count"] += 1
            module["total_duration_seconds"] = round(module["total_duration_seconds"] + duration_seconds, 3)
            module["prompt_tokens"] += prompt_tokens
            module["completion_tokens"] += completion_tokens
            module["total_tokens"] += prompt_tokens + completion_tokens
            module["estimated_cost"] = round(module["estimated_cost"] + estimated_cost, 6)
            self._save()

    def get_summary(self):
        """返回全局累计统计"""
        with self._lock:
            data = self._data.copy()
            by_module = data.get("by_module", {})
            total = {
                "call_count": sum(m.get("call_count", 0) for m in by_module.values()),
                "total_duration_seconds": round(sum(m.get("total_duration_seconds", 0) for m in by_module.values()), 3),
                "prompt_tokens": sum(m.get("prompt_tokens", 0) for m in by_module.values()),
                "completion_tokens": sum(m.get("completion_tokens", 0) for m in by_module.values()),
                "total_tokens": sum(m.get("total_tokens", 0) for m in by_module.values()),
                "estimated_cost": round(sum(m.get("estimated_cost", 0) for m in by_module.values()), 6),
                "cost_unit": "元",
            }
            return {
                "by_module": by_module,
                "total": total,
                "last_reset": data.get("last_reset"),
            }

    def reset(self):
        """清空所有全局统计"""
        with self._lock:
            self._data = {
                "by_module": {},
                "last_reset": datetime.now().isoformat(),
            }
            self._save()


# 全局默认 tracker：适合单任务场景直接调用 tracked_chat_completion 后统一读取
_default_tracker = CostTracker()

# 全局累计统计器（持久化）
_global_stats = GlobalStats()


def get_default_tracker():
    """获取默认全局 tracker"""
    return _default_tracker


def get_global_stats():
    """获取全局累计统计器"""
    return _global_stats


Message = Dict[str, str]


def _is_retryable_exception(exc: Exception) -> bool:
    """判断异常是否可重试（限流/5xx/网络瞬断）"""
    try:
        from openai import RateLimitError, APIError, APIConnectionError, InternalServerError
        if isinstance(exc, (RateLimitError, APIConnectionError, InternalServerError)):
            return True
        if isinstance(exc, APIError):
            # 5xx 可重试，4xx（除 429）不重试
            status = getattr(exc, "status_code", None)
            if status is None:
                return False
            return status == 429 or 500 <= int(status) < 600
    except ImportError:
        pass
    return False


def _validate_response_not_empty(response: Any, model: str, source: str) -> None:
    """校验 LLM 响应非空，空响应抛 LLMEmptyResponseError 触发上层重试

    空响应特征：
    - response 本身为 None
    - response.choices 为 None / 空 list
    - response.choices[0].message.content 为 None
    """
    if response is None:
        raise LLMEmptyResponseError(f"LLM 返回空响应（response is None）: model={model}, source={source}")
    choices = getattr(response, "choices", None)
    if not choices:
        raise LLMEmptyResponseError(f"LLM 返回空 choices: model={model}, source={source}")
    first = choices[0]
    message = getattr(first, "message", None)
    if message is None or getattr(message, "content", None) is None:
        raise LLMEmptyResponseError(f"LLM 返回空 message.content: model={model}, source={source}")


def tracked_chat_completion(
    client: Any,
    model: str,
    messages: List[Message],
    tracker: Optional["CostTracker"] = None,
    source: str = "未分类",
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
    **kwargs: Any,
) -> Any:
    """
    包装 client.chat.completions.create，自动记录时间、token 与成本，内建空响应防护与重试。

    参数：
        client: OpenAI 风格客户端
        model: 模型名称
        messages: messages 数组
        tracker: 指定的 CostTracker 实例；不传则使用全局默认 tracker
        max_retries: 最大重试次数（默认 3，仅对空响应/限流/5xx 生效）
        retry_base_delay: 重试基础间隔（秒），第 n 次重试等待 base * 2^(n-1)
        **kwargs: 其他传给 create 的参数

    返回：
        response（原始响应对象，包含 content/usage 等）

    异常：
        LLMEmptyResponseError: 重试 max_retries 次后仍为空响应
        其他异常: 不可重试的错误（4xx 参数错误等）直接抛出
    """
    tracker = tracker or _default_tracker

    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        start_ts = time.time()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            # 空响应防护：校验 choices 与 content
            _validate_response_not_empty(response, model, source)

            end_ts = time.time()

            usage = getattr(response, "usage", None) or {}
            prompt_tokens = usage.get("prompt_tokens", 0) if isinstance(usage, dict) else getattr(usage, "prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0) if isinstance(usage, dict) else getattr(usage, "completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0) if isinstance(usage, dict) else getattr(usage, "total_tokens", 0)
            duration_seconds = round(end_ts - start_ts, 3)
            estimated_cost = estimate_cost(prompt_tokens, completion_tokens, model)

            call_record = {
                "model": model,
                "start_time": datetime.fromtimestamp(start_ts).isoformat(),
                "end_time": datetime.fromtimestamp(end_ts).isoformat(),
                "duration_seconds": duration_seconds,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "estimated_cost": estimated_cost,
                "cost_unit": "元",
                "source": source,
            }
            tracker.add_call(call_record)

            # 同步写入全局持久化统计
            _global_stats.record_call(source, duration_seconds, prompt_tokens, completion_tokens, estimated_cost)

            if attempt > 0:
                logger.info("LLM 重试成功: model=%s, source=%s, attempt=%s/%s",
                            model, source, attempt + 1, max_retries + 1)
            return response

        except LLMEmptyResponseError as e:
            last_exc = e
            if attempt < max_retries:
                delay = retry_base_delay * (2 ** attempt)
                logger.warning("LLM 空响应，%ss 后第 %s/%s 次重试: model=%s, source=%s, err=%s",
                               delay, attempt + 1, max_retries, model, source, e)
                time.sleep(delay)
            else:
                logger.error("LLM 空响应重试耗尽: model=%s, source=%s, attempts=%s",
                             model, source, max_retries + 1)
                raise

        except Exception as e:
            if _is_retryable_exception(e) and attempt < max_retries:
                delay = retry_base_delay * (2 ** attempt)
                logger.warning("LLM 可重试异常，%ss 后第 %s/%s 次重试: model=%s, source=%s, err=%s",
                               delay, attempt + 1, max_retries, model, source, e)
                time.sleep(delay)
                last_exc = e
            else:
                # 不可重试或重试耗尽，直接抛出
                raise

    # 理论不可达，保险起见
    if last_exc is not None:
        raise last_exc
    raise LLMEmptyResponseError(f"LLM 调用失败（未知原因）: model={model}, source={source}")
