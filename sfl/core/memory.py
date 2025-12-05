"""
Field Memory - Persistent memory for the Synthetic Field.

The field memory stores the history of all reports and patterns,
enabling learning from accumulated experience.
"""

import json
import os
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict
import torch

from sfl.core.types import Report


class FieldMemory:
    """
    Persistent memory for the Synthetic Field.

    Stores reports, pattern archives, and learned embeddings.
    """

    def __init__(self, path: str = "./field_memory"):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

        # In-memory caches
        self._report_cache: Dict[bytes, List[Report]] = defaultdict(list)
        self._archive: Dict[bytes, Dict[str, Any]] = {}
        self._embeddings: Dict[bytes, torch.Tensor] = {}

        # Stats
        self._total_reports = 0
        self._total_archived = 0

        # Load existing data
        self._load()

    def integrate(self, report: Report) -> None:
        """
        Integrate a new report into the memory.

        Args:
            report: The report to integrate
        """
        nh_id = report.nh_id
        self._report_cache[nh_id].append(report)
        self._total_reports += 1

        # Update embedding for pattern
        self._update_embedding(nh_id, report)

        # Persist periodically
        if self._total_reports % 100 == 0:
            self._persist_reports()

    def archive_pattern(self, nh_id: bytes) -> None:
        """
        Archive a pattern when it's dissolved.

        Args:
            nh_id: The pattern's identifier
        """
        reports = self._report_cache.get(nh_id, [])

        self._archive[nh_id] = {
            "nh_id": nh_id.hex(),
            "archived_at": int(time.time() * 1000),
            "total_reports": len(reports),
            "embedding": self._embeddings.get(nh_id, torch.zeros(256)).tolist(),
        }
        self._total_archived += 1

        # Clean up
        if nh_id in self._report_cache:
            del self._report_cache[nh_id]
        if nh_id in self._embeddings:
            del self._embeddings[nh_id]

        # Persist archive
        self._persist_archive()

    def get_reports(self, nh_id: bytes, limit: int = 100) -> List[Report]:
        """
        Get recent reports for a pattern.

        Args:
            nh_id: The pattern's identifier
            limit: Maximum number of reports to return

        Returns:
            List of reports
        """
        reports = self._report_cache.get(nh_id, [])
        return reports[-limit:]

    def get_embedding(self, nh_id: bytes) -> Optional[torch.Tensor]:
        """
        Get the learned embedding for a pattern.

        Args:
            nh_id: The pattern's identifier

        Returns:
            Embedding tensor or None
        """
        return self._embeddings.get(nh_id)

    def get_similar_patterns(
        self, nh_id: bytes, k: int = 5
    ) -> List[tuple[bytes, float]]:
        """
        Find patterns with similar embeddings.

        Args:
            nh_id: The pattern's identifier
            k: Number of similar patterns to return

        Returns:
            List of (pattern_id, similarity_score) tuples
        """
        embedding = self._embeddings.get(nh_id)
        if embedding is None:
            return []

        similarities = []
        for other_id, other_emb in self._embeddings.items():
            if other_id == nh_id:
                continue

            # Cosine similarity
            sim = torch.nn.functional.cosine_similarity(
                embedding.unsqueeze(0), other_emb.unsqueeze(0)
            ).item()
            similarities.append((other_id, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def query(
        self,
        state_pattern: Optional[torch.Tensor] = None,
        action_pattern: Optional[torch.Tensor] = None,
        limit: int = 100,
    ) -> List[Report]:
        """
        Query memory for similar reports.

        Args:
            state_pattern: State pattern to match
            action_pattern: Action pattern to match
            limit: Maximum results

        Returns:
            List of matching reports
        """
        results = []

        for reports in self._report_cache.values():
            for report in reports:
                score = 0.0

                if state_pattern is not None:
                    # Match state
                    sim = torch.nn.functional.cosine_similarity(
                        state_pattern.unsqueeze(0).flatten(),
                        report.state.flatten(),
                        dim=0,
                    ).item()
                    score += sim

                if action_pattern is not None:
                    # Match action
                    sim = torch.nn.functional.cosine_similarity(
                        action_pattern.unsqueeze(0).flatten(),
                        report.action.flatten(),
                        dim=0,
                    ).item()
                    score += sim

                if score > 0:
                    results.append((score, report))

        # Sort by score and return top results
        results.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in results[:limit]]

    def size(self) -> int:
        """Get total number of reports in memory."""
        return sum(len(reports) for reports in self._report_cache.values())

    def stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "total_reports": self._total_reports,
            "cached_reports": self.size(),
            "patterns_with_reports": len(self._report_cache),
            "archived_patterns": self._total_archived,
            "embeddings": len(self._embeddings),
        }

    def _update_embedding(self, nh_id: bytes, report: Report) -> None:
        """Update embedding for a pattern based on new report."""
        # Get or initialize embedding
        if nh_id not in self._embeddings:
            self._embeddings[nh_id] = torch.zeros(256)

        # Simple exponential moving average update
        current = self._embeddings[nh_id]

        # Concatenate report features
        features = torch.cat([
            report.state.flatten()[:64] if report.state.numel() >= 64 else torch.nn.functional.pad(report.state.flatten(), (0, 64 - report.state.numel())),
            report.action.flatten()[:32] if report.action.numel() >= 32 else torch.nn.functional.pad(report.action.flatten(), (0, 32 - report.action.numel())),
            report.outcome.flatten()[:64] if report.outcome.numel() >= 64 else torch.nn.functional.pad(report.outcome.flatten(), (0, 64 - report.outcome.numel())),
        ])

        # Pad to 256
        if features.numel() < 256:
            features = torch.nn.functional.pad(features, (0, 256 - features.numel()))
        else:
            features = features[:256]

        # EMA update
        alpha = 0.1
        self._embeddings[nh_id] = (1 - alpha) * current + alpha * features

    def _load(self) -> None:
        """Load persisted data."""
        # Load archive
        archive_path = self.path / "archive.json"
        if archive_path.exists():
            with open(archive_path, "r") as f:
                data = json.load(f)
                self._archive = {
                    bytes.fromhex(k): v for k, v in data.items()
                }
                self._total_archived = len(self._archive)

    def _persist_reports(self) -> None:
        """Persist reports to disk."""
        # For now, just keep in memory
        # In production, would write to disk or database
        pass

    def _persist_archive(self) -> None:
        """Persist archive to disk."""
        archive_path = self.path / "archive.json"
        with open(archive_path, "w") as f:
            data = {k.hex(): v for k, v in self._archive.items()}
            json.dump(data, f)

    def clear(self) -> None:
        """Clear all memory."""
        self._report_cache.clear()
        self._archive.clear()
        self._embeddings.clear()
        self._total_reports = 0
        self._total_archived = 0
