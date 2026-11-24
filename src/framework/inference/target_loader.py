"""Resolve trusted packaged team entry points without depending on agent topology."""
from pathlib import Path
import re


def inference_target_options(config):
    """Use an explicit module/class pair; otherwise use the conventional entry."""
    options = (config or {}).get('inference') or {}
    module = options.get('target_module')
    target_class = options.get('target_class')
    if module is not None and (not isinstance(module, str) or module.endswith('.py')
            or not re.fullmatch(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*', module)):
        raise ValueError('inference.target_module must be a dotted Python module name, without .py or a file path')
    if target_class is not None and (not isinstance(target_class, str)
            or not re.fullmatch(r'[A-Za-z_]\w*', target_class)):
        raise ValueError('inference.target_class must be a Python class name')
    return {'target_module': module if target_class is not None else None,
            'target_class': target_class or 'ChatModel'}


def team_target_entry(agent_dir, config):
    """Resolve explicit inference first; use Agent.py for incomplete/absent settings."""
    root = Path(agent_dir)
    module = inference_target_options(config)['target_module']
    if module is None:
        conventional = root / 'Agent.py'
        return conventional if conventional.is_file() else None
    relative = Path(*module.split('.'))
    for candidate in (root / relative.with_suffix('.py'), root / relative / '__init__.py'):
        if candidate.is_file():
            if not candidate.resolve().is_relative_to(root.resolve()):
                raise ValueError('Target module must remain inside the team agent directory')
            return candidate
    return None
