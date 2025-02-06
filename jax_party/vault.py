from functools import partial
from typing import Callable, Dict
from colorama import Fore, Style
import flashbax as fbx
from flashbax.vault import Vault
import jax
import jax.numpy as jnp
import chex
from omegaconf import OmegaConf
from jax_party import JaxParty
from typing import TypeVar

Transition = TypeVar("Transition")
BufferState = TypeVar("BufferState")
Experience = TypeVar("Experience")


class PartyVault(Vault):
    def __init__(
        self,
        vault_name,
        save_interval: int,
        buffer_add: Callable[[BufferState, Experience], BufferState],
        experience_structure=None,
        rel_dir="vaults",
        vault_uid=None,
        compression=None,
        metadata=None,
    ):
        super().__init__(
            vault_name,
            experience_structure,
            rel_dir,
            vault_uid,
            compression,
            metadata,
        )
        self.save_interval = save_interval
        self.buffer_add = buffer_add

    @partial(jax.jit, static_argnames="self")
    def _reshape_experience(
        self,
        experience: Dict[str, chex.Array],
    ) -> Dict[str, chex.Array]:
        """Reshape experience to match buffer."""
        # Shape legend:
        # D: Number of devices
        # NU: Number of updates per evaluation
        # UB: Update batch size
        # T: Time steps per rollout
        # NE: Number of environments
        # Swap the T and NE axes (D, NU, UB, T, NE, ...) -> (D, NU, UB, NE, T, ...)
        experience = jax.tree.map(lambda x: x.swapaxes(3, 4), experience)
        # Merge 4 leading dimensions into 1. (D, NU, UB, NE, T ...) -> (D * NU * UB * NE, T, ...)
        experience = jax.tree.map(lambda x: x.reshape(-1, *x.shape[4:]), experience)
        return experience

    def add_and_write(
        self, buffer_state: BufferState, experience: Experience, eval_step: int
    ) -> BufferState:
        flashbax_transition = self._reshape_experience(
            {
                # (D, NU, UB, T, NE, ...)
                "done": experience.done,
                "action": experience.action,
                "reward": experience.reward,
                "observation": experience.obs.agents_view,
                "legal_action_mask": experience.obs.action_mask,
            }
        )
        # Add to fbx buffer
        buffer_state = self.buffer_add(buffer_state, flashbax_transition)

        # Save buffer into vault
        if eval_step % self.save_interval == 0:
            write_length = self.write(buffer_state)
            print(
                f"{Fore.WHITE}{Style.BRIGHT}(Wrote {write_length}) Vault index = {self.vault_index}{Style.RESET_ALL}"
            )
        return buffer_state

    def push_to_neptune(self):  # TODO:
        raise NotImplementedError


def make_buffer_and_vault(
    env: JaxParty,
    config: dict,
    vault_id: str = None,  # None is converted to timestamp
) -> tuple[
    BufferState,
    PartyVault,
]:
    # TODO: convert int32 to int8
    n_devices = len(jax.devices())

    dummy_flashbax_transition = {
        "done": jnp.zeros((config.system.num_agents,), dtype=bool),
        "action": jnp.zeros((config.system.num_agents,), dtype=jnp.int32),
        "reward": jnp.zeros((config.system.num_agents,), dtype=jnp.float32),
        "observation": jnp.zeros(
            (
                config.system.num_agents,
                env.observation_spec.agents_view.shape[1],
            ),
            dtype=jnp.float32,
        ),
        "legal_action_mask": jnp.zeros(
            (
                config.system.num_agents,
                env.num_actions,
            ),
            dtype=bool,
        ),
    }

    buffer = fbx.make_flat_buffer(
        max_length=int(1e6),  # Max number of transitions to store
        min_length=int(1),
        sample_batch_size=1,
        add_sequences=True,
        add_batch_size=(
            n_devices
            * config.system.num_updates_per_eval
            * config.system.update_batch_size
            * config.arch.num_envs
        ),
    )
    buffer_state = buffer.init(
        dummy_flashbax_transition,
    )
    buffer_add = jax.jit(buffer.add, donate_argnums=(0))
    vault = PartyVault(
        vault_name=config.vault.vault_name,
        experience_structure=buffer_state.experience,
        vault_uid=vault_id,
        metadata=OmegaConf.to_container(
            config, resolve=True
        ),  # Metadata must be a python dictionary
        buffer_add=buffer_add,
        save_interval=config.vault.save_interval,
    )

    return buffer_state, vault
