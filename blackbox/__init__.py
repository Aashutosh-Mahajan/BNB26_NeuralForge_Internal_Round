"""Black Box: a flight recorder and crash investigator for AI agents."""
from .engine import Engine
from .recorder import SandboxAgent, wrap
from .storage import Store
from .diagnosis import diagnose

__all__ = ["Engine", "Store", "SandboxAgent", "wrap", "diagnose"]
__version__ = "0.2.0"
