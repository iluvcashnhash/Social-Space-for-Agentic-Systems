"""Algorithmic feed (Infocracy) — content scoring, attention, RPE, burnout."""

from hub.algorithmic_feed import (
    Content,
    FeedConfig,
    ScoredContent,
    score_content_for_agent,
    attention_decay,
    reward_prediction_error,
    update_cognitive_load,
    rank_feed,
)

__all__ = [
    "Content",
    "FeedConfig",
    "ScoredContent",
    "score_content_for_agent",
    "attention_decay",
    "reward_prediction_error",
    "update_cognitive_load",
    "rank_feed",
]
