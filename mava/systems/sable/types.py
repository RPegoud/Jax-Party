# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Dict, Tuple

from chex import Array, PRNGKey
from flax.core.frozen_dict import FrozenDict
from jumanji.types import TimeStep
from optax._src.base import OptState
from typing_extensions import NamedTuple


class SableNetworkConfig(NamedTuple):
    """Configuration for the Sable network."""

    n_block: int
    n_head: int
    embed_dim: int


class HiddenStates(NamedTuple):
    """Hidden states for the encoder and decoder."""

    encoder: Array
    decoder: Tuple[Array, Array]


class RecLearnerState(NamedTuple):
    """State of the learner for Memory Sable"""

    params: FrozenDict
    opt_states: OptState
    key: PRNGKey
    env_state: Array
    timestep: TimeStep
    hidden_state: HiddenStates


class FFLearnerState(NamedTuple):
    """State of the learner for ff-Sable"""

    params: FrozenDict
    opt_states: OptState
    key: PRNGKey
    env_state: Array
    timestep: TimeStep


class Transition(NamedTuple):
    """Transition tuple."""

    done: Array
    action: Array
    value: Array
    reward: Array
    log_prob: Array
    obs: Array
    info: Dict


ActorApply = Callable[
    [FrozenDict, Array, Array, HiddenStates, PRNGKey],
    Tuple[Array, Array, Array, Array, HiddenStates],
]
LearnerApply = Callable[
    [FrozenDict, Array, Array, Array, HiddenStates, Array, PRNGKey], Tuple[Array, Array, Array]
]
