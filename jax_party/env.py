from typing import Optional, Sequence, Tuple
from jax_party import Action, State, Observation, tree_slice
from jumanji.types import TimeStep, restart, termination, transition

from jumanji.env import Environment
from jumanji import specs
import jax
import jax.numpy as jnp
import chex
from functools import cached_property


class PartyEnvironment(Environment[State, specs.DiscreteArray, Observation]):

    def __init__(self, time_limit: int):
        self.num_agents = 3
        self.time_limit = time_limit
        self.POSSIBLE_MATCHUPS = jnp.array(
            [
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
            ],
        )
        self.PAYOFF_MATRIX_AGENT_1 = jnp.array(
            [
                [-1, -1, -1],  # NOOP actions get negative rewards
                [3, 3, 0],  # COOPERATE row
                [5, 5, 1],  # DEFECT row
            ]
        )
        self.PAYOFF_MATRIX_AGENT_2 = jnp.array(
            [
                [-1, 3, 5],  # NOOP actions get negative rewards
                [-1, 3, 5],  # COOPERATE row
                [-1, 0, 1],  # DEFECT row
            ]
        )
        super().__init__()

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        active_agents = jax.random.choice(key, self.POSSIBLE_MATCHUPS, shape=())
        action_mask = jax.vmap(self._get_action_mask)(active_agents)
        ranking = jnp.zeros(self.num_agents)
        next_key, _ = jax.random.split(key)

        state = State(
            active_agents=active_agents,
            cumulative_rewards=0,
            ranking=ranking,
            step_count=0,
            action_mask=action_mask,
            key=next_key,
        )

        observation = Observation(
            active_agents=active_agents, action_mask=action_mask, ranking=ranking
        )
        timestep = restart(observation=observation)

        return state, timestep

    def step(
        self, state: State, actions: chex.Array
    ) -> Tuple[State, TimeStep[Observation]]:
        actions = self._get_valid_actions(actions, state.action_mask)
        rewards = self._get_rewards(state, actions)
        state.cumulative_rewards += rewards

        ranking = jnp.argsort(state.cumulative_rewards, descending=True)
        steps = state.step_count + 1
        done = steps >= self.time_limit
        next_active_agents = jax.random.choice(
            state.key, self.POSSIBLE_MATCHUPS, shape=()
        )
        next_action_mask = jax.vmap(self._get_action_mask)(next_active_agents)
        next_key, _ = jax.random.split(state.key)

        next_observation = Observation(
            active_agents=next_active_agents,
            action_mask=next_action_mask,
            ranking=ranking,
        )

        next_state = State(
            active_agents=next_active_agents,
            cumulative_rewards=state.cumulative_rewards,
            ranking=ranking,
            step_count=steps,
            action_mask=next_action_mask,
            key=next_key,
        )

        timestep = jax.lax.cond(
            done,
            termination,
            transition,
            rewards,
            next_observation,
        )

        return next_state, timestep

    def _get_action_mask(self, is_active: chex.Scalar) -> chex.Array:
        """
        Returns legal actions depending on the agent's state.
        """
        return jax.lax.select(
            is_active,
            jnp.array([False, True, True]),  # COOPERATE, DEFECT
            jnp.array([True, False, False]),  # NOOP
        )

    def _get_valid_actions(
        self, actions: chex.Array, action_mask: chex.Array
    ) -> chex.Array:
        """
        Converts illegal actions into NOOP (e.g. COOPERATE or DEFECT for inactive agents).
        """

        def _get_valid_action(
            action: chex.Array, action_mask: chex.Array
        ) -> chex.Array:
            return jax.lax.cond(
                action_mask[action], lambda: action, lambda: Action.NOOP
            )

        return jax.vmap(_get_valid_action)(actions, action_mask)

    def _get_rewards(self, state: State, actions: chex.Array) -> chex.Array:
        active_agents_indices = jnp.where(state.active_agents == 1, size=2)[0]
        active_agents_actions = actions.at[active_agents_indices].get()
        active_agents_rewards = tree_slice(
            [self.PAYOFF_MATRIX_AGENT_1, self.PAYOFF_MATRIX_AGENT_2],
            (active_agents_actions[0], active_agents_actions[1]),
        )
        rewards = (
            jnp.zeros(self.num_agents)
            .at[active_agents_indices]
            .set(active_agents_rewards)
        )

        return rewards

    @cached_property
    def observation_spec(self) -> specs.DiscreteArray:
        active_agents = specs.Array((self.num_agents,), jnp.bool_, "action_mask")
        action_mask = specs.Array((self.num_agents,), jnp.bool_, "action_mask")
        ranking = specs.Array((self.num_agents,), jnp.int32, "ranking")
        return specs.Spec(
            Observation,
            "ObservationSpec",
            active_agents=active_agents,
            action_mask=action_mask,
            ranking=ranking,
        )

    @cached_property
    def action_spec(self) -> specs.DiscreteArray:
        return specs.MultiDiscreteArray(
            num_values=jnp.array([len(Action)] * self.num_agents, jnp.int32),
            name="action",
        )
