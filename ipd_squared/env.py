from enum import IntEnum
from typing import NamedTuple, Optional, Sequence, Tuple

from colorama import Fore, Style
from jumanji.types import TimeStep, restart, termination, transition

from jumanji.env import Environment
from jumanji import specs
from jumanji.types import StepType
import jax
import jax.numpy as jnp
import chex
from functools import cached_property

from mava.wrappers.jumanji import JumanjiMarlWrapper


def register_IPDSquared():
    from jumanji.registration import register, registered_environments

    register("IPDSquared-v0", "ipd_squared.env:IPDSquared")

    assert "IPDSquared-v0" in registered_environments()
    print(f"{Fore.GREEN}IPDSquared-v0 registered successfully!{Style.RESET_ALL}")


NUM_AGENTS = 4
NUM_ACTIONS = 2


class Action(IntEnum):
    COOPERATE: int = 0
    DEFECT: int = 1


@chex.dataclass
class State:
    power: chex.Array
    history: chex.Array
    step_count: int
    action_mask: chex.Array
    key: chex.PRNGKey


class Observation(NamedTuple):
    agents_view: chex.Array
    action_mask: chex.Array


def _get_action_mask() -> chex.Array:
    """
    Returns legal actions depending on the agent's state.
    """
    return jnp.ones((NUM_AGENTS, NUM_ACTIONS), dtype=jnp.bool_)


class IPDSquaredGenerator:
    """
    Generator for the JaxParty Environment (essentially the reset function).
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, key: chex.PRNGKey) -> State:
        init_power = jnp.full(shape=(2, 2), fill_value=0.5)
        action_mask = _get_action_mask()
        history = jnp.full((NUM_AGENTS,), -1)

        state = State(
            power=init_power,
            history=history,
            step_count=0,
            action_mask=action_mask,
            key=key,
        )

        return state


class IPDSquared(Environment[State, specs.DiscreteArray, Observation]):
    """
    Iterated Prisoner's Dilemma between 3 agents with observable ranking.
    """

    def __init__(
        self,
        generator: IPDSquaredGenerator,
        epsilon_min: float,
        epsilon_max: float,
        scaling_factor: int,
        time_limit: int = 1000,
        cc: float = 1,
        cd: float = 4,
        dd: float = -1,
    ):
        self.env_name = "IPDSquared-v0"
        self.num_agents = NUM_AGENTS
        self.num_actions = NUM_ACTIONS
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.scaling_factor = scaling_factor
        self.time_limit = time_limit
        self.PAYOFF_MATRIX = jnp.array(
            [
                [cc, -cd],  # COOPERATE row
                [cd, dd],  # DEFECT row
            ]
        )
        self.generator = generator
        super().__init__()

    def _make_observation(self, state: State) -> Observation:
        """
        Concatenates the power and history into an observation array.
        """
        agents_view = jnp.concatenate((state.power.flatten(), state.history.flatten()))
        return Observation(
            agents_view=agents_view,
            action_mask=state.action_mask,
        )

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        state = self.generator(key)
        observation = self._make_observation(state)
        timestep = TimeStep(
            step_type=StepType.FIRST,
            reward=jnp.zeros(self.num_agents),
            discount=jnp.ones(()),
            observation=observation,
            extras={},
        )

        return state, timestep

    def _update_power(
        self, power: chex.Array, inner_actions: chex.Array, team_payoff: chex.Array
    ) -> chex.Array:
        def _disagreement():
            update = team_payoff / self.scaling_factor
            return jax.lax.cond(
                power[0] >= power[1],
                lambda _: jnp.array([power[0] + update, power[1] - update]),
                lambda _: jnp.array([power[0] - update, power[1] + update]),
                operand=None,
            )

        def _agreement():
            return power.reshape(2, 1)

        return jax.lax.cond(
            inner_actions[0] != inner_actions[1],
            _disagreement,
            _agreement,
        )

    def step(
        self, state: State, inner_actions: chex.Array
    ) -> Tuple[State, TimeStep[Observation]]:
        # TODO: we might want to sample the actions based on power instead of picking greedily
        # TODO: do we want to allow different actions at the local and global level at some point?

        inner_actions = inner_actions.reshape(2, 2)
        next_key, epsilon_key = jax.random.split(state.key, num=2)
        epsilons = jax.random.uniform(
            epsilon_key, (2, 1), minval=self.epsilon_min, maxval=self.epsilon_max
        )
        epsilons = epsilons * jnp.array([-1, 1])

        power = jax.nn.softmax(state.power + epsilons, axis=-1)
        outer_actions = jnp.take_along_axis(
            inner_actions, jnp.argmax(power, axis=-1, keepdims=True), axis=-1
        )
        outer_payoffs = jnp.array(
            [
                self.PAYOFF_MATRIX[outer_actions[0], outer_actions[1]],
                self.PAYOFF_MATRIX[outer_actions[1], outer_actions[0]],
            ]
        )

        power = jax.vmap(self._update_power)(
            power, inner_actions, outer_payoffs
        ).squeeze()
        rewards = (power * outer_payoffs).flatten()

        history = inner_actions.flatten()

        steps = state.step_count + 1
        done = steps >= self.time_limit

        next_state = State(
            power=power,
            history=history,
            step_count=steps,
            action_mask=_get_action_mask(),
            key=next_key,
        )
        next_observation = self._make_observation(next_state)

        timestep = jax.lax.cond(
            done,
            termination,
            transition,
            rewards,
            next_observation,
        )

        return next_state, timestep

    @cached_property
    def observation_spec(self) -> specs.DiscreteArray:
        agents_view = specs.Array((self.num_agents * 2,), jnp.bool_, "agents_view")
        action_mask = specs.Array((len(Action),), jnp.bool_, "action_mask")
        return specs.Spec(
            Observation,
            "ObservationSpec",
            agents_view=agents_view,
            action_mask=action_mask,
        )

    @cached_property
    def action_spec(self) -> specs.DiscreteArray:
        return specs.MultiDiscreteArray(
            num_values=jnp.array([len(Action)] * self.num_agents, jnp.int32),
            name="action",
        )


class IPDSquaredMARLWrapper(JumanjiMarlWrapper):
    """
    Duplicates the timesteps to extend them to multi-agent format.
    """

    def __init__(self, env: IPDSquared, add_global_state: bool = False):
        super().__init__(env, add_global_state)
        self._env: IPDSquared

    def modify_timestep(self, timestep: TimeStep) -> TimeStep[Observation]:
        """Duplicates the observation for each agent."""
        replicate = lambda x: jnp.tile(x, self._env.num_agents).reshape(
            self._env.num_agents, -1
        )
        agents_view = replicate(timestep.observation.agents_view)
        marl_observation = Observation(agents_view, timestep.observation.action_mask)
        marl_timestep = TimeStep(
            observation=marl_observation,
            reward=timestep.reward,
            discount=timestep.discount,
            step_type=timestep.step_type,
        )

        return marl_timestep
