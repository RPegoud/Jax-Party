import os
import neptune


def push_to_neptune(
    name: str,
    description: str,
    folder_path: str,
):
    run = neptune.init_run(
        name=name,
        description=description,
        project="Jax-Party/JaxParty",
        # TODO: add variable env
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmZGI2MTg5OC04ZDg5LTQ0ZjMtOGU1Yi03NmYwMjA1MDNjMGIifQ==",
    )

    """Upload all files in a folder to Neptune"""
    if os.path.exists(folder_path):
        for root, _, files in os.walk(folder_path):
            for file in files:
                full_path = os.path.join(root, file)
                neptune_path = os.path.relpath(full_path, folder_path)
                run[f"artifacts/{neptune_path}"].upload(full_path)

    # TODO: plot results

    run.stop()
