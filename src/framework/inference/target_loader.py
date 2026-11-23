"""Resolve trusted packaged team entry points without depending on agent topology."""
from pathlib import Path
import re


def team_target_entry(agent_dir, config):
    """Prefer Agent.py; otherwise resolve a configured Python module in agent_dir."""
    root = Path(agent_dir)
    conventional = root / 'Agent.py'
    if conventional.is_file():
        return conventional
    module = (config or {}).get('inference', {}).get('target_module')
    if module is None:
        return None
    if not isinstance(module, str) or module.endswith('.py') or not re.fullmatch(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*', module):
        raise ValueError('inference.target_module must be a dotted Python module name, without .py or a file path')
    relative = Path(*module.split('.'))
    for candidate in (root / relative.with_suffix('.py'), root / relative / '__init__.py'):
        if candidate.is_file():
            if not candidate.resolve().is_relative_to(root.resolve()):
                raise ValueError('Target module must remain inside the team agent directory')
            return candidate
    return None
