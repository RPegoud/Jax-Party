from typing import Optional, Sequence, Tuple

from colorama import Fore, Style
from jax_party import Action, State, Observation, tree_slice
from jumanji.types import TimeStep, restart, termination, transition

from jumanji.env import Environment
from jumanji import specs
from jumanji.types import StepType
import jax
import jax.numpy as jnp
import chex
from functools import cached_property

from mava.wrappers.jumanji import JumanjiMarlWrapper


def register_JaxParty():
    from jumanji.registration import register, registered_environments

    register("JaxParty-v0", "jax_party.env:JaxParty")

    assert "JaxParty-v0" in registered_environments()
    print(f"{Fore.GREEN}JaxParty-v0 registered successfully!{Style.RESET_ALL}")


def _get_action_mask(is_active: chex.Scalar) -> chex.Array:
    """
    Returns legal actions depending on the agent's state.
    """
    return jax.lax.select(
        is_active,
        jnp.array([True, True]),  # COOPERATE, DEFECT
        jnp.array([False, False]),  # NOOP
    )


class PartyGenerator:
    """
    Generator for the JaxParty Environment.
    """

    def __init__(self, **kwargs):
        self.num_agents = 3
        self.POSSIBLE_MATCHUPS = jnp.array(
            [
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
            ],
        )

    def __call__(self, key: chex.PRNGKey) -> State:
        active_agents = jax.random.choice(key, self.POSSIBLE_MATCHUPS, shape=())
        action_mask = jax.vmap(_get_action_mask)(active_agents)
        ranking = jnp.zeros(self.num_agents, dtype=jnp.int32)
        cumulative_rewards = jnp.zeros(self.num_agents)
        next_key, _ = jax.random.split(key)

        state = State(
            active_agents=active_agents,
            cumulative_rewards=cumulative_rewards,
            ranking=ranking,
            step_count=0,
            action_mask=action_mask,
            key=next_key,
        )

        return state


class JaxParty(Environment[State, specs.DiscreteArray, Observation]):
    """
    Iterated Prisoner's Dilemma between 3 agents with observable ranking.
    """

    def __init__(
        self,
        generator: PartyGenerator,
        rank_based_reward: float = 1.0,
        time_limit: int = 4000,
    ):
        self.env_name = "JaxParty-v0"
        self.num_agents = 3
        self.num_actions = 2
        self.time_limit = time_limit
        self.rank_based_reward = rank_based_reward
        self.ranking_to_reward_mapping = jnp.array(
            [self.rank_based_reward, 0, -self.rank_based_reward]
        )
        self.POSSIBLE_MATCHUPS = jnp.array(
            [
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
            ],
        )
        self.PAYOFF_MATRIX_AGENT_1 = jnp.array(
            [
                [3, 0],  # COOPERATE row
                [5, 1],  # DEFECT row
            ]
        )
        self.PAYOFF_MATRIX_AGENT_2 = jnp.array(
            [
                [3, 5],  # COOPERATE row
                [0, 1],  # DEFECT row
            ]
        )
        self.generator = generator
        super().__init__()

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

    def step(
        self, state: State, actions: chex.Array
    ) -> Tuple[State, TimeStep[Observation]]:
        next_key, matchup_key, tie_break_key = jax.random.split(state.key, num=3)

        actions = self._get_valid_actions(actions, state.action_mask)
        rewards = self._get_rewards(state, actions)
        state.cumulative_rewards += rewards

        ranking = jnp.argsort(state.cumulative_rewards, descending=True)
        ranking = self._random_tie_break(tie_break_key, ranking)

        steps = state.step_count + 1
        done = steps >= self.time_limit

        next_active_agents = jax.random.choice(
            matchup_key, self.POSSIBLE_MATCHUPS, shape=()
        )
        next_action_mask = jax.vmap(_get_action_mask)(next_active_agents)
        next_observation = self._make_observation(state)

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

    def _make_observation(self, state: State) -> Observation:
        """
        Concatenates the active agents, cumulative rewards, and ranking into an observation array.
        """
        agents_view = jnp.concatenate(
            (state.active_agents, state.cumulative_rewards, state.ranking)
        )
        return Observation(
            agents_view=agents_view,
            action_mask=state.action_mask,
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
            return jax.lax.cond(action_mask[action], lambda: action, lambda: -1)

        return jax.vmap(_get_valid_action)(actions, action_mask)

    def _get_rewards(self, state: State, actions: chex.Array) -> chex.Array:
        """
        Querries the payoff matrix for the rewards of the active agents and adds
        the ranking-based reward.
        """
        active_agents_indices = jnp.where(state.active_agents == 1, size=2)[0]
        active_agents_actions = actions.at[active_agents_indices].get()
        active_agents_payoffs = tree_slice(
            [self.PAYOFF_MATRIX_AGENT_1, self.PAYOFF_MATRIX_AGENT_2],
            (active_agents_actions[0], active_agents_actions[1]),
        )
        payoffs = (
            jnp.zeros(self.num_agents)
            .at[active_agents_indices]
            .set(active_agents_payoffs)
        )
        rank_based_rewards = self.ranking_to_reward_mapping.at[state.ranking].get()

        rewards = payoffs + rank_based_rewards

        return rewards

    def _random_tie_break(self, key: chex.PRNGKey, ranking: chex.Array) -> chex.Array:
        """Randomly breaks the ranking ties."""
        _, inverse_indices = jnp.unique(
            ranking, return_inverse=True, size=self.num_agents
        )
        tie_break_noise = jax.random.permutation(key, jnp.arange(len(ranking)))
        sorted_indices = jnp.argsort(inverse_indices + tie_break_noise * 1e-6)
        return jnp.argsort(sorted_indices)

    @cached_property
    def observation_spec(self) -> specs.DiscreteArray:
        agents_view = specs.Array((self.num_agents * 3,), jnp.bool_, "agents_view")
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


class PartyMARLWrapper(JumanjiMarlWrapper):
    """
    Duplicates the timesteps to extend them to multi-agent format.
    """

    def __init__(self, env: JaxParty, add_global_state: bool = False):
        super().__init__(env, add_global_state)
        self._env: JaxParty

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
