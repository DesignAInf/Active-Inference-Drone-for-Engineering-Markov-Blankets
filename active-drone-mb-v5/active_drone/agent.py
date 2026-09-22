"""Joint expected-free-energy selection over control and boundary design."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .inference import StructuredBlanketLearner
from .structure import ACTIVE, EXTERNAL, SENSORY
from .world import DesignAction

CONDITIONS = ("constitutive", "diagnostic_only", "factorized", "random_design", "passive", "oracle")


@dataclass
class JointPolicy:
    control: float
    design: DesignAction
    risk: float
    ambiguity: float
    structural_value: float
    epistemic_value: float
    efe: float


class ConstitutiveDrone:
    controls = np.array([-1.25, -0.62, 0.0, 0.62, 1.25])

    def __init__(self, rng: np.random.Generator, condition: str):
        if condition not in CONDITIONS:
            raise ValueError(condition)
        self.rng, self.condition = rng, condition
        self.learner = StructuredBlanketLearner()
        self.swap_used = False

    def reset(self) -> None:
        self.learner.reset()
        self.swap_used = False

    def _control_risk(self, world, control: float) -> tuple[float, float]:
        marg = self.learner.marginals
        p_external = marg[:, EXTERNAL]
        estimated_wind = float(np.sum(p_external * world.x) / max(np.sum(p_external), 1e-9))
        predicted = 0.91 * world.y + world.dt * (estimated_wind + control)
        risk = predicted * predicted + 4.2 * max(0.0, abs(predicted) - 1.55) ** 2 + 0.035 * control * control
        ambiguity = float(np.mean(self.learner.residual_var) / (1 + len(self.learner.buffer)))
        return float(risk), ambiguity

    def _oracle_action(self, world) -> DesignAction:
        if world.leaks:
            i, e = max(world.leaks, key=world.leaks.get)
            return DesignAction("shield", min(i, e), max(i, e))
        mismatch = np.flatnonzero(world.roles != world.target_roles)
        if len(mismatch) >= 2:
            return DesignAction("swap", int(mismatch[0]), int(mismatch[1]))
        return DesignAction()

    def select(self, world) -> tuple[JointPolicy, list[JointPolicy]]:
        actions = world.candidate_actions()
        if self.swap_used:
            actions = [a for a in actions if a.kind != "swap"]
        if self.condition == "passive":
            actions = [DesignAction()]
        elif self.condition == "oracle":
            actions = [self._oracle_action(world)]
        elif self.condition == "random_design":
            actions = [actions[int(self.rng.integers(len(actions)))]]

        scored = []
        for action in actions:
            structural, pragmatic, epistemic = self.learner.action_features(action, world.quality)
            success = self.learner.interventions.success_probability(action.kind)
            design_cost = 0.0 if action.kind == "noop" else (0.055 if action.kind == "shield" else 0.085)
            for control in self.controls:
                risk, ambiguity = self._control_risk(world, float(control))
                # Genuine EFE: expected pragmatic risk and ambiguity minus
                # information gain; structure enters as a future-risk reduction.
                efe = risk + 0.18 * ambiguity + design_cost - success * (1.2 * structural + 3.0 * pragmatic) - 0.30 * epistemic
                scored.append(JointPolicy(float(control), action, risk, ambiguity,
                                          success * structural, epistemic, float(efe)))

        if self.condition == "factorized":
            # V4-style separation: choose control ignoring design, then choose
            # design ignoring its effect on viability.
            best_control = min(self.controls, key=lambda u: self._control_risk(world, float(u))[0])
            subset = [p for p in scored if p.control == float(best_control)]
            chosen = min(subset, key=lambda p: -p.structural_value + (p.design.kind != "noop") * 0.055)
        else:
            values = np.array([p.efe for p in scored])
            temperature = 0.055 if self.condition not in ("random_design",) else 0.04
            logits = -(values - values.min()) / temperature
            probs = np.exp(np.clip(logits, -50, 0)); probs /= probs.sum()
            chosen = scored[int(self.rng.choice(len(scored), p=probs))]
        if chosen.design.kind == "swap":
            self.swap_used = True
        return chosen, scored

    @property
    def constitutive(self) -> bool:
        return self.condition not in ("diagnostic_only", "passive")
