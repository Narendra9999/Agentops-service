"""Preserve team code and conventional asset layout in MLflow code_paths."""
import os


def team_code_paths(agent_dir: str, config_path: str = '') -> list:
    """Package the existing source tree without rewriting customer file access.

    MLflow retains each directory's subtree. Keep agent/ and configs/ as sibling
    entries, and retain the existing flattened subpackage import aliases. A valid
    source-relative path therefore has the same meaning after staging.
    """
    paths = [agent_dir]
    for entry in sorted(os.listdir(agent_dir)):
        if entry.startswith('.') or entry == '__pycache__':
            continue
        full = os.path.join(agent_dir, entry)
        if os.path.isdir(full):
            paths.append(full)
    configs = os.path.join(os.path.dirname(agent_dir), 'configs')
    if os.path.isdir(configs):
        paths.append(configs)
    if config_path and os.path.isfile(config_path):
        paths.append(config_path)
    return paths
