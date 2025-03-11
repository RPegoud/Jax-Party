import os
import neptune
from omegaconf import DictConfig
from src import plot_vault, get_latest_folder
from colorama import Fore, Style


def push_to_neptune(
    run_name: str,
    alg_name: str,
    env_name: str,
    description: str,
    config: DictConfig,
):
    run = neptune.init_run(
        name=run_name,
        description=description,
        project="Jax-Party/JaxParty",
        # TODO: add variable env
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmZGI2MTg5OC04ZDg5LTQ0ZjMtOGU1Yi03NmYwMjA1MDNjMGIifQ==",
    )

    vault_id = get_latest_folder(f"./experiment_results/{alg_name}")

    """Upload all files in a folder to Neptune"""
    if os.path.exists(vault_id):
        for root, _, files in os.walk(vault_id):
            for file in files:
                full_path = os.path.join(root, file)
                neptune_path = os.path.relpath(full_path, vault_id)
                run[f"artifacts/{neptune_path}"].upload(full_path)

    figures = plot_vault(alg_name, env_name, True, vault_id)
    for name, figure in figures.items():
        run[name].upload(figure)
    run["config"] = config

    print(f"{Fore.BLUE}Pushed config and results to Neptune!{Style.RESET_ALL}")

    run.stop()
