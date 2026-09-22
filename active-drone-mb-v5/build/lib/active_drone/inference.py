"""Structured blanket inversion and a learned intervention model."""

from __future__ import annotations

from collections import deque
import numpy as np

from .structure import ACTIVE, EXTERNAL, INTERNAL, NODES, PARTITIONS, SENSORY, TEMPLATE_MASKS, TEMPLATES
from .world import DesignAction


def entropy(p: np.ndarray) -> float:
    p = np.asarray(p, float)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


class InterventionModel:
    """Beta-Bernoulli model learned from causal changes caused by interventions."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.counts = {"shield": np.array([1.0, 1.0]), "swap": np.array([1.0, 1.0])}

    def success_probability(self, kind: str) -> float:
        if kind == "noop":
            return 0.0
        c = self.counts[kind]
        return float(c[1] / c.sum())

    def uncertainty(self, kind: str) -> float:
        if kind == "noop":
            return 0.0
        a, b = self.counts[kind][1], self.counts[kind][0]
        return float(a * b / ((a + b) ** 2 * (a + b + 1)))

    def update(self, action: DesignAction, changed: bool) -> None:
        if action.kind != "noop":
            self.counts[action.kind][int(bool(changed))] += 1


class StructuredBlanketLearner:
    """Posterior on all complete I/S/A/E partitions plus learned dynamics."""

    def __init__(self, window: int = 34, hazard: float = 0.018):
        self.window, self.hazard = window, hazard
        self.buffer = deque(maxlen=window)
        self.interventions = InterventionModel()
        self.reset()

    def reset(self) -> None:
        self.logits = np.zeros(len(PARTITIONS), float)
        self.posterior = np.full(len(PARTITIONS), 1 / len(PARTITIONS))
        self.a_hat = np.zeros((NODES, NODES), float)
        self.residual_var = np.ones(NODES)
        self.buffer.clear()
        self.interventions.reset()
        self.steps = 0
        self._marginals = np.full((NODES, 4), 0.25)

    def predict(self) -> None:
        self.posterior = (1 - self.hazard) * self.posterior + self.hazard / len(self.posterior)

    def observe_transition(self, x: np.ndarray, x_next: np.ndarray, control: float,
                           action: DesignAction, changed: bool) -> None:
        self.buffer.append((np.asarray(x, float), np.asarray(x_next, float), float(control)))
        self.interventions.update(action, changed)
        self.steps += 1
        if len(self.buffer) >= 12 and self.steps % 2 == 0:
            self._fit()

    def _fit(self) -> None:
        x = np.asarray([r[0] for r in self.buffer])
        y = np.asarray([r[1] for r in self.buffer])
        u = np.asarray([r[2] for r in self.buffer])[:, None]
        design = np.column_stack([x, u, np.ones(len(x))])
        ridge = 0.18 * np.eye(design.shape[1])
        beta = np.linalg.solve(design.T @ design + ridge, design.T @ y)
        self.a_hat = beta[:NODES].T
        resid = y - design @ beta
        self.residual_var = np.mean(resid * resid, axis=0) + 1e-4

        observed = np.clip(abs(self.a_hat), 0, 0.55)
        # Dynamic likelihood: fit of the learned coupling matrix to each role
        # template, with semantic evidence learned from control response and
        # innovation variance rather than a supplied role-cue likelihood.
        structural = -42.0 * np.mean((TEMPLATES - observed[None, :, :]) ** 2, axis=(1, 2))
        control_gain = abs(beta[NODES])
        innovation = self.residual_var / max(np.mean(self.residual_var), 1e-9)
        semantic = np.zeros(len(PARTITIONS))
        for role, signal, sign in ((ACTIVE, control_gain, 1.0), (EXTERNAL, innovation, 0.55)):
            mask = PARTITIONS == role
            semantic += sign * np.mean(mask * signal[None, :], axis=1)
        # S and I orientation follows from the direction of the learned matrix.
        self.logits = structural + 1.7 * semantic
        shifted = self.logits - np.max(self.logits)
        q = np.exp(shifted)
        self.posterior = q / q.sum()
        self._update_marginals()

    def _update_marginals(self) -> None:
        for role in range(4):
            self._marginals[:, role] = self.posterior @ (PARTITIONS == role)

    @property
    def marginals(self) -> np.ndarray:
        return self._marginals

    @property
    def map_roles(self) -> np.ndarray:
        return PARTITIONS[int(np.argmax(self.posterior))].copy()

    @property
    def partition_entropy(self) -> float:
        return entropy(self.posterior)

    def expected_gap(self) -> float:
        # Posterior expectation of direct I--E coupling energy in learned A.
        sq = self.a_hat * self.a_hat
        value = 0.0
        for weight, roles in zip(self.posterior, PARTITIONS):
            i, e = roles == INTERNAL, roles == EXTERNAL
            value += weight * (sq[np.ix_(i, e)].sum() + sq[np.ix_(e, i)].sum())
        return float(value)

    def action_features(self, action: DesignAction, quality: np.ndarray) -> tuple[float, float, float]:
        """Predicted structural benefit, pragmatic benefit, and epistemic value."""
        if action.kind == "noop":
            return 0.0, 0.0, 0.0
        a, b = action.first, action.second
        marg = self.marginals
        if action.kind == "shield":
            ie = marg[a, INTERNAL] * marg[b, EXTERNAL] + marg[b, INTERNAL] * marg[a, EXTERNAL]
            strength = self.a_hat[a, b] ** 2 + self.a_hat[b, a] ** 2
            structural = float(ie * (0.15 + 18 * strength))
            pragmatic = 0.0
        else:
            boundary_a = marg[a, SENSORY] + marg[a, ACTIVE]
            boundary_b = marg[b, SENSORY] + marg[b, ACTIVE]
            # Precision crafting: route the uniquely precise module into a
            # likely boundary port and route the degraded boundary module out.
            gain_ab = boundary_a * (quality[b] - quality[a]) * (1 - marg[b, INTERNAL])
            gain_ba = boundary_b * (quality[a] - quality[b]) * (1 - marg[a, INTERNAL])
            extremes = {int(np.argmin(quality)), int(np.argmax(quality))}
            contrast_bonus = 0.75 if {a, b} == extremes else 0.0
            pragmatic = float(max(gain_ab, gain_ba) + contrast_bonus)
            structural = 0.04 * abs(float(boundary_a - boundary_b))
        uncertainty = self.interventions.uncertainty(action.kind)
        local_h = entropy(marg[a]) + entropy(marg[b])
        epistemic = float(uncertainty * local_h)
        return structural, pragmatic, epistemic
