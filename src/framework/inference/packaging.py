"""Preserve team code and optionally map assets in MLflow's packaged layout."""
import os
from pathlib import Path
import shutil
import tempfile


def _relative_path(value, label):
    if not isinstance(value, str) or not value or '\\' in value:
        raise ValueError(f'{label} must be a nonempty relative path')
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError(f'{label} must stay inside the team directory')
    return path


def _contained(path, root):
    if not path.resolve().is_relative_to(root):
        raise ValueError(f'Asset path escapes the team directory: {path}')
    if path.is_dir():
        # Check directory symlinks without following them recursively.
        for directory, dirs, files in os.walk(path, followlinks=False):
            for name in dirs + files:
                entry = Path(directory) / name
                if entry.is_symlink() and not entry.resolve().is_relative_to(root):
                    raise ValueError(f'Asset symlink escapes the team directory: {entry}')


def team_code_paths(agent_dir: str, config_path: str = '', *, asset_mappings=None) -> list:
    """Retain legacy paths unless explicit mappings require a private staged copy.

    Mapping sources are team-relative files/directories. Destinations are new paths
    under agent/ in the model layout. Existing files/directories cannot be replaced.
    Return original import aliases, not aliases for injected asset directories.
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
    if asset_mappings is None or asset_mappings == []:
        return paths
    if not isinstance(asset_mappings, list):
        raise ValueError('model.asset_mappings must be a list')
    root = Path(agent_dir).resolve().parent
    mappings = []
    for item in asset_mappings:
        if not isinstance(item, dict) or set(item) != {'source', 'destination'}:
            raise ValueError('Each asset mapping requires only source and destination')
        source = root / _relative_path(item['source'], 'source')
        destination = _relative_path(item['destination'], 'destination')
        if destination.parts[0] != 'agent' or len(destination.parts) < 2:
            raise ValueError('Asset destination must be a new path below agent/')
        _contained(source, root)
        if not source.exists():
            raise FileNotFoundError(f'Asset source does not exist: {source}')
        mappings.append((source, destination))
    # Copy top-level paths once; subpackage entries are already included by agent/.
    for original in paths:
        _contained(Path(original), root)
    stage = Path(tempfile.mkdtemp(prefix='agentops-assets-'))
    try:
        for original in paths:
            relative = Path(original).resolve().relative_to(root)
            target = stage / relative
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if Path(original).is_dir():
                shutil.copytree(original, target)
            else:
                shutil.copy2(original, target)
        for source, destination in mappings:
            target = stage / destination
            if target.exists():
                raise ValueError(f'Asset mapping cannot overwrite existing path: {destination}')
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, target)
            else:
                shutil.copy2(source, target)
        return [str(stage / Path(p).resolve().relative_to(root)) for p in paths]
    except Exception:
        shutil.rmtree(stage)
        raise
