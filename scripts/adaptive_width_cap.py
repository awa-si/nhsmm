"""Research-only exact EM with a maximum constant-density block width."""

from __future__ import annotations

import time

import numpy as np

from scripts.compare_adaptive_duration_em import PSEUDOCOUNT, objective
from scripts.compare_unlabelled_duration_em import aggregate, emission_scores
from scripts.verify_isolated_trading_hsmm import D


def capped_mstep(counts, penalty, max_width, *, pseudocount=PSEUDOCOUNT):
    """Globally maximize augmented-count log density over bounded partitions."""
    counts = np.asarray(counts, dtype=float)
    if (
        counts.ndim != 2
        or not counts.size
        or not np.isfinite(counts).all()
        or np.any(counts < 0)
        or not np.isfinite(penalty)
        or penalty < 0
        or not isinstance(max_width, int)
        or isinstance(max_width, bool)
        or max_width < 1
        or not np.isfinite(pseudocount)
        or pseudocount <= 0
    ):
        raise ValueError("finite counts/penalty, positive pseudocount and integer width required")
    p = np.empty_like(counts)
    partitions = []
    for k, row in enumerate(counts + pseudocount):
        prefix = np.r_[0.0, np.cumsum(row)]
        best = np.full(len(prefix), -np.inf)
        parent = np.zeros(len(prefix), dtype=int)
        best[0] = penalty
        for end in range(1, len(prefix)):
            starts = np.arange(max(0, end - max_width), end)
            masses = prefix[end] - prefix[starts]
            if np.any(masses <= 0):
                raise FloatingPointError("augmented prefix mass lost precision")
            scores = best[starts] + masses * np.log(masses / (end - starts)) - penalty
            parent[end] = int(starts[np.argmax(scores)])
            best[end] = float(np.max(scores))
        blocks = []
        end = len(row)
        while end:
            start = int(parent[end])
            blocks.append((start, end))
            p[k, start:end] = row[start:end].sum() / row.sum() / (end - start)
            end = start
        partitions.append(list(reversed(blocks)))
    return p, partitions


def validate_state(p, blocks, max_width=D):
    p = np.asarray(p, dtype=float)
    if (
        p.shape != (4, D)
        or not np.isfinite(p).all()
        or np.any(p <= 0)
        or not np.allclose(p.sum(1), 1, rtol=0, atol=1e-12)
        or len(blocks) != 4
    ):
        raise ValueError("positive normalized full-age PMF and four partitions required")
    for row, state in zip(p, blocks, strict=True):
        cursor = 0
        for start, end in state:
            if (
                not isinstance(start, (int, np.integer))
                or not isinstance(end, (int, np.integer))
                or start != cursor
                or not start < end <= D
                or end - start > max_width
                or not np.allclose(row[start:end], row[start], rtol=0, atol=1e-15)
            ):
                raise ValueError("partition must cover all ages and obey density/width")
            cursor = end
        if cursor != D:
            raise ValueError("partition must cover all ages")


def refine_state(p, blocks, max_width):
    if not isinstance(max_width, int) or isinstance(max_width, bool) or max_width < 1:
        raise ValueError("positive integer width required")
    validate_state(p, blocks)
    refined = [
        [(a, min(a + max_width, end)) for start, end in state for a in range(start, end, max_width)]
        for state in blocks
    ]
    result = np.array(p, dtype=float, copy=True)
    validate_state(result, refined, max_width)
    return result, refined


def fit_capped(
    x,
    penalty,
    max_width,
    *,
    iterations,
    initial_state,
    resume_history=None,
    checkpoint=None,
    tag="width8",
):
    if not isinstance(iterations, int) or iterations < 0:
        raise ValueError("nonnegative integer iterations required")
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError("finite nonnegative penalty required")
    if not isinstance(max_width, int) or isinstance(max_width, bool) or max_width < 1:
        raise ValueError("positive integer width required")
    p = np.array(initial_state[0], dtype=float, copy=True)
    blocks = [[tuple(b) for b in state] for state in initial_state[1]]
    validate_state(p, blocks, max_width)
    first, offset, history = 0, 0.0, []
    if resume_history is not None:
        first = len(resume_history) - 1
        if (
            first < 0
            or first > iterations
            or [h["completed_updates"] for h in resume_history] != list(range(first + 1))
        ):
            raise ValueError("invalid resume update indices")
        if not all(
            np.isfinite(h[k])
            for h in resume_history
            for k in ("ll", "penalized_objective", "elapsed_seconds")
        ):
            raise ValueError("nonfinite resume history")
        if any(
            b["penalized_objective"] < a["penalized_objective"] - 1e-6
            for a, b in zip(resume_history, resume_history[1:])
        ):
            raise ValueError("invalid resume objective history")
        history = [dict(h) for h in resume_history[:-1]]
        offset = resume_history[-1]["elapsed_seconds"]
    scores = emission_scores(x)
    start = time.monotonic()
    for update in range(first, iterations + 1):
        ll, counts = aggregate(scores, p)
        score = objective(ll, p, blocks, penalty)
        if resume_history is not None and update == first:
            saved = resume_history[-1]
            if abs(ll - saved["ll"]) > 1e-7 or abs(score - saved["penalized_objective"]) > 1e-7:
                raise ValueError("resume state does not reproduce saved evidence/objective")
        history.append(
            {
                "completed_updates": update,
                "ll": ll,
                "penalized_objective": score,
                "blocks_by_state": [len(b) for b in blocks],
                "elapsed_seconds": offset + time.monotonic() - start,
            }
        )
        if update and score < history[-2]["penalized_objective"] - 1e-6:
            raise AssertionError("penalized marginal objective decreased")
        if update == first or update % 5 == 0 or update == iterations:
            print(
                f"{tag} EM={update}/{iterations} blocks={history[-1]['blocks_by_state']}",
                flush=True,
            )
            if checkpoint is not None:
                checkpoint(p.copy(), [list(b) for b in blocks], [dict(h) for h in history])
        if update < iterations:
            p, blocks = capped_mstep(counts, penalty, max_width)
    return p, blocks, history
