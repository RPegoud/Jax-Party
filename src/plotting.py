from flashbax.vault import Vault
from src import get_latest_folder
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import pandas as pd
import plotly.express as px
import chex
from typing import TypeVar

Figure = TypeVar("Figure")


def plot_vault(
    alg_name: str,
    env_name: str,
    latest: bool = True,
    vault_id: str = None,
) -> dict[str:Figure]:
    merge_n_leading_dims = lambda array, n: array.reshape(-1, *array.shape[n:])
    action_bincount = lambda array: jnp.bincount(array, length=2)

    def _plot_cumulative_rewards(): ...

    def _plot_action_proportion(actions: chex.Array) -> Figure:
        action_counts = jax.vmap(action_bincount, in_axes=(1))(actions)
        return px.bar(
            pd.DataFrame(action_counts, columns=["cooperate", "defect"]),
            title="Proportion of cooperation and deffection per agent",
            labels={"index": "agent", "value": "count"},
        )

    def _plot_final_sum_of_rewards(rewards: chex.Array) -> Figure:
        fig = px.bar(
            rewards.sum(axis=0),
            title="Final reward per agent",
            labels={"index": "agent", "value": "sum of rewards"},
        )
        fig.update_layout(showlegend=False)
        fig.update_xaxes(type="category")
        return fig

    if latest:
        vault_id = get_latest_folder(f"./experiment_results/{alg_name}")
    v = Vault(
        rel_dir=f"experiment_results/{alg_name}/{vault_id}",
        vault_name=env_name,
        vault_uid=vault_id,
    )
    buffer_state = v.read()
    actions = buffer_state.experience["action"]
    rewards = buffer_state.experience["reward"]
    obs = buffer_state.experience["observation"]

    # flatten the batch dimensions of experience samples
    actions, rewards, obs = merge_n_leading_dims((actions, rewards, obs), (2, 2, 2))

    action_counts_fig: Figure = _plot_action_proportion(actions)

    return {
        "action_counts": action_counts_fig,
    }
