import os
import shutil
from datetime import datetime
from colorama import Fore, Style
import argparse
import neptune


def extract_timestamp(folder_name: str):
    """Extract timestamp from folder name."""
    try:
        return datetime.strptime(folder_name, "%Y%m%d%H%M%S")
    except ValueError:
        return None


def get_latest_folder(base_path: str):
    """Return the latest timestamped folder in a directory."""
    timestamps = [
        folder for folder in os.listdir(base_path) if extract_timestamp(folder)
    ]
    return max(timestamps, key=extract_timestamp) if timestamps else None


def aggregate_outputs(alg_name: str):
    vaults_dir = "vaults/jax_party"
    results_dir = os.path.join("results/json", alg_name)
    outputs_dir = "outputs"
    checkpoints_dir = os.path.join("checkpoints", alg_name)
    experiment_results_dir = os.path.join("experiment_results", alg_name)

    os.makedirs(experiment_results_dir, exist_ok=True)

    latest_vault = get_latest_folder(vaults_dir)
    latest_metrics = get_latest_folder(results_dir)
    latest_checkpoint = get_latest_folder(checkpoints_dir)

    # Get the latest config.yaml from outputs
    latest_config_path = None
    for date_folder in sorted(os.listdir(outputs_dir), reverse=True):
        date_path = os.path.join(outputs_dir, date_folder)
        if not os.path.isdir(date_path):
            continue
        for time_folder in sorted(os.listdir(date_path), reverse=True):
            hydra_path = os.path.join(date_path, time_folder, ".hydra/config.yaml")
            if os.path.exists(hydra_path):
                latest_config_path = hydra_path
                break
        if latest_config_path:
            break

    if not latest_vault:
        print(f"{Fore.RED}No vaults found in {vaults_dir}. Skipping.{Style.RESET_ALL}")
        return

    dt = extract_timestamp(latest_vault)
    date_str = dt.strftime("%y-%m-%d")
    time_str = dt.strftime("%H-%M-%S")
    target_dir = os.path.join(experiment_results_dir, date_str, time_str)
    vault_target_dir = os.path.join(target_dir, "vaults/jax_party", latest_vault)

    os.makedirs(vault_target_dir, exist_ok=True)

    # Move vault
    shutil.move(vaults_dir, vault_target_dir)

    # Move metrics
    if latest_metrics:
        os.makedirs(os.path.join(target_dir, "metrics"), exist_ok=True)
        shutil.move(
            os.path.join(results_dir, latest_metrics),
            os.path.join(target_dir, "metrics"),
        )

    metrics_file = os.path.join(results_dir, "metrics.json")
    if os.path.exists(metrics_file):
        shutil.move(metrics_file, os.path.join(target_dir, "metrics/metrics.json"))

    # Move checkpoint
    if latest_checkpoint:
        os.makedirs(os.path.join(target_dir, "checkpoints"), exist_ok=True)
        shutil.move(
            os.path.join(checkpoints_dir, latest_checkpoint),
            os.path.join(target_dir, "checkpoints"),
        )

    # Copy latest config.yaml
    if latest_config_path:
        shutil.copy(latest_config_path, os.path.join(target_dir, "config.yaml"))

    # Remove original folders after moving data
    for folder in ["vaults", "results", "outputs", "checkpoints"]:
        if os.path.exists(folder):
            shutil.rmtree(folder)

    print(f"{Fore.WHITE}{Style.BRIGHT}Aggregation complete for {alg_name}!")
    print(f"Output path: {target_dir}{Style.RESET_ALL}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate outputs for a given algorithm."
    )
    parser.add_argument("alg_name", type=str, help="Algorithm name (e.g., ff_ippo)")
    args = parser.parse_args()

    aggregate_outputs(args.alg_name)
