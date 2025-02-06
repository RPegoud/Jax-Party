from typing import Dict
import flashbax as fbx
from flashbax.vault import Vault
import jax
import jax.numpy as jnp
import chex


class PartyVault(Vault):
    def __init__(
        self,
        vault_name,
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

    @jax.jit
    def _reshape_experience(
        self, experience: Dict[str, chex.Array]
    ) -> Dict[str, chex.Array]:
        """Reshape experience to match buffer."""
        # Swap the T and NE axes (D, NU, UB, T, NE, ...) -> (D, NU, UB, NE, T, ...)
        experience = jax.tree.map(lambda x: x.swapaxes(3, 4), experience)
        # Merge 4 leading dimensions into 1. (D, NU, UB, NE, T ...) -> (D * NU * UB * NE, T, ...)
        experience = jax.tree.map(lambda x: x.reshape(-1, *x.shape[4:]), experience)
        return experience

    def _push_vault_to_neptune(): ...
