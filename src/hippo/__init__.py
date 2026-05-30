"""Hippo — memory layer for AI agents with conflict resolution."""

from .client import Hippo
from .consolidation import Consolidator
from .exceptions import BatchPartialFailure
from .importance import ImportanceScorer, LLMImportanceScorer
from .models import Memory

__all__ = [
    "Hippo",
    "Memory",
    "Consolidator",
    "ImportanceScorer",
    "LLMImportanceScorer",
    "BatchPartialFailure",
]
__version__ = "0.1.3"
