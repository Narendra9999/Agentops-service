"""Local example import bridge; canonical implementation lives in src/teams."""
from pathlib import Path
__path__ = [str(Path(__file__).resolve().parents[3] / "src/teams/platform-intelligence/agent/fe_multi_agent")]
