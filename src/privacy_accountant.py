"""
Privacy Accountant
==============================

Responsible for:
  - taking target epsilon, delta, sampling rate, max_steps/iterations and
    calibrating the Gaussian noise multiplier sigma
  - tracking privacy expenditure epsilon(t)  during training
  - reporting a final epsilon and the full privacy curve
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import dp_accounting
from dp_accounting.rdp import rdp_privacy_accountant

# Dense default set of RDP orders
DEFAULT_RDP_ORDERS: Sequence[float] = ([1 + x / 10. for x in range(1, 100)]
                                       + list(range(11, 65)) 
                                       + [128, 256, 512, 1024])


@dataclass
class PrivacyReport:
    """Everything the experiment runner/metrics module need to log."""

    target_epsilon: float
    achieved_epsilon: float
    delta: float
    sigma: float
    sampling_probability: float
    steps_taken: int
    epsilon_curve: List[Tuple[int, float]] = field(default_factory=list)  # (step, eps(t))

    def as_dict(self) -> dict:
        return {
            "target_epsilon": self.target_epsilon,
            "achieved_epsilon": self.achieved_epsilon,
            "delta": self.delta,
            "sigma": self.sigma,
            "sampling_probability": self.sampling_probability,
            "steps_taken": self.steps_taken,
        }


class PrivacyAccountant:
    """
     - Compute the noise multiplier σ needed to achieve a target privacy budget (ε,δ)
     - Track the accumulated privacy loss during training
     - Produce a final report after training
    """

    def __init__(
        self,
        num_users: int,
        user_batch_size: int,
        delta: float,
        rdp_orders: Sequence[float] = DEFAULT_RDP_ORDERS,
    ):
        if user_batch_size > num_users:
            raise ValueError("user_batch_size cannot exceed num_users")
        self.num_users = num_users
        self.user_batch_size = user_batch_size
        self.delta = delta
        self.rdp_orders = list(rdp_orders)

        self.sampling_probability= user_batch_size/num_users # for Poisson sampling
        self.sigma: Optional[float] = None

        self._accountant = rdp_privacy_accountant.RdpAccountant(self.rdp_orders)
        self._steps_taken = 0
        self._epsilon_curve: List[Tuple[int, float]] = []


    def calibrate_sigma(
        self,
        target_epsilon: float,
        steps: int,
        sigma_guess: float = 1.0,
        tol: float = 1e-6,
    ) -> float:
        """
        calibrate sigma for a target (epsilon, delta) budget
        """
        if math.isinf(target_epsilon):
            self.sigma = 0
            return 0

        q = self.sampling_probability

        # new accountant is created for every candidate σ
        #this is necessary because each trial starts from zero privacy loss
        def make_fresh_accountant():
            return rdp_privacy_accountant.RdpAccountant(self.rdp_orders)
        
        #represent one DP step
        def make_event(sigma: float):
            return dp_accounting.SelfComposedDpEvent(
                dp_accounting.PoissonSampledDpEvent(
                    q, dp_accounting.GaussianDpEvent(sigma)
                ),
                steps,
            )
        # tries to diff values of σ
        sigma = dp_accounting.calibrate_dp_mechanism(
            make_fresh_accountant,
            make_event,
            target_epsilon,
            self.delta,
            bracket_interval=dp_accounting.LowerEndpointAndGuess(0.0, sigma_guess),
            tol=tol,
        )
        self.sigma = float(sigma)
        return self.sigma


    def step(self) -> float:
        """
        tracking during training
        """
        if self.sigma is None:
            raise RuntimeError("Call calibrate_sigma() before step().")
        if self.sigma<=0:
            # epsilon = inf regime (non-private): nothing meaningful to track
            self._steps_taken+=1
            eps_t = float("inf")
            self._epsilon_curve.append((self._steps_taken, eps_t))
            return eps_t
        
        # represent Aggregate and add noise
        event = dp_accounting.PoissonSampledDpEvent(
            self.sampling_probability, dp_accounting.GaussianDpEvent(self.sigma)
        )
        self._accountant.compose(event)
        self._steps_taken+=1
        eps_t = self._accountant.get_epsilon(self.delta)
        self._epsilon_curve.append((self._steps_taken, eps_t))
        return eps_t

    def current_epsilon(self) -> float:
        """Epsilon accrued so far without advancing the accountant"""
        if self.sigma <=0.0:
            return float("inf")
        return self._accountant.get_epsilon(self.delta)


    def final_report(self, target_epsilon: float) -> PrivacyReport:
        achieved = self.current_epsilon()
        return PrivacyReport(
            target_epsilon=target_epsilon,
            achieved_epsilon=achieved,
            delta=self.delta,
            sigma=self.sigma if self.sigma is not None else float("nan"),
            sampling_probability=self.sampling_probability,
            steps_taken=self._steps_taken,
            epsilon_curve=list(self._epsilon_curve),
        )

    @property
    def epsilon_curve(self) -> List[Tuple[int, float]]:
        return list(self._epsilon_curve)


def non_private_report(steps_taken: int) -> PrivacyReport:
    """ epsilon = inf"""
    return PrivacyReport(
        target_epsilon=float("inf"),
        achieved_epsilon=float("inf"),
        delta=float("nan"),
        sigma=0.0,
        sampling_probability=float("nan"),
        steps_taken=steps_taken,
        epsilon_curve=[(t, float("inf")) for t in range(1, steps_taken+1)],
    )
