from enum import IntEnum
from typing import NamedTuple

import chex
from chex import dataclass


class Action(IntEnum):
    # NOOP: int = -1
    COOPERATE: int = 0
    DEFECT: int = 1


@dataclass
class State:
    active_agents: chex.Array
    cumulative_rewards: chex.Array
    ranking: chex.Array
    step_count: int
    action_mask: chex.Array
    key: chex.PRNGKey


class Observation(NamedTuple):
    agents_view: chex.Array
    action_mask: chex.Array
