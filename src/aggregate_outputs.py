import os
import shutil
from datetime import datetime
from colorama import Fore, Style
import argparse
import neptune
import glob


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


def move_files(src_dir, dest_dir):
    """Move all files from src_dir to dest_dir."""
    for item in os.listdir(src_dir):
        s = os.path.join(src_dir, item)
        d = os.path.join(dest_dir, item)
        if os.path.isdir(s):
            shutil.move(s, d)
        else:
            shutil.move(s, d)


def aggregate_outputs(alg_name: str, env_name: str):
    vaults_dir = f"vaults/{env_name}"
    results_dir = os.path.join("results/json", alg_name)
    outputs_dir = "outputs"
    checkpoints_dir = os.path.join("checkpoints", alg_name)
    experiment_results_dir = os.path.join("experiment_results", alg_name)

    os.makedirs(experiment_results_dir, exist_ok=True)

    timestamp = get_latest_folder(vaults_dir)
    # latest_metrics = get_latest_folder(results_dir)
    # latest_checkpoint = get_latest_folder(checkpoints_dir)

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

    if not timestamp:
        print(f"{Fore.RED}No vaults found in {vaults_dir}. Skipping.{Style.RESET_ALL}")
        return

    dt = extract_timestamp(timestamp)
    target_dir = os.path.join(experiment_results_dir, timestamp)
    # vault_target_dir = target_dir #os.path.join(target_dir, f"vaults/{env_name}", latest_vault)

    os.makedirs(target_dir, exist_ok=True)

    # Move vault
    shutil.move(vaults_dir, target_dir)

    # # Move metrics
    # if latest_metrics:
    #     os.makedirs(os.path.join(target_dir, "metrics"), exist_ok=True)
    #     shutil.move(
    #         os.path.join(results_dir, latest_metrics),
    #         os.path.join(target_dir, "metrics"),
    #     )

    # metrics_file = os.path.join(results_dir, "metrics.json")
    # if os.path.exists(metrics_file):
    #     shutil.move(metrics_file, os.path.join(target_dir, "metrics/metrics.json"))

    # Move checkpoint
    # if latest_checkpoint:
    #     os.makedirs(os.path.join(target_dir, "checkpoints"), exist_ok=True)
    #     shutil.move(
    #         os.path.join(checkpoints_dir, latest_checkpoint),
    #         os.path.join(target_dir, "checkpoints"),
    #     )

    # Copy latest config.yaml
    if latest_config_path:
        shutil.copy(latest_config_path, os.path.join(target_dir, "config.yaml"))

    # Remove original folders after moving data
    for folder in ["vaults", "results", "outputs", "checkpoints"]:
        if os.path.exists(folder):
            shutil.rmtree(folder)

    # reformat vaults and metrics folders to remove intermediate folder
    # env_path = os.path.join(target_dir, env_name, timestamp)
    # print(f"{env_path=}")
    # if os.path.exists(env_path):
    #     for item in os.listdir(env_path):
    #         item_path = os.path.join(env_path, item)
    #         if os.path.isdir(item_path) and item.startswith("d"):
    #             combined_d_path = os.path.join(target_dir, env_name, "d")
    #             os.makedirs(combined_d_path, exist_ok=True)
    #             for file in glob.glob(os.path.join(item_path, "*")):
    #                 shutil.move(file, combined_d_path)
    #             shutil.rmtree(item_path)
    #         else:
    #             shutil.move(item_path, os.path.join(target_dir, env_name))
    #     shutil.rmtree(env_path)

    print(f"{Fore.WHITE}{Style.BRIGHT}Aggregation complete for {alg_name}!")
    print(f"Output path: {target_dir}{Style.RESET_ALL}")
