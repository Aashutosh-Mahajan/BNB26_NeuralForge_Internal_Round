"""An honest narrow SDK; external LangGraph adapters are not implemented."""
from dataclasses import dataclass, field
from copy import deepcopy

@dataclass(frozen=True)
class SandboxAgent:
    task_family: str = "finance"
    params: dict = field(default_factory=dict)

class WrappedAgent:
    def __init__(self, agent, db_path):
        from ..engine import Engine
        self.agent = agent
        self.engine = Engine(db_path)

    def invoke(self, inputs, on_step=None):
        if isinstance(inputs, str):
            prompt, params = inputs, {}
        elif isinstance(inputs, dict):
            prompt = inputs["prompt"]
            params = inputs.get("params", {})
        else:
            raise TypeError("invoke expects a prompt string or {'prompt': ..., 'params': ...}")
        configured = deepcopy(self.agent.params)
        configured.update(params)
        return self.engine.run(prompt, self.agent.task_family, configured, on_step=on_step)

    run = invoke

def wrap(agent, db_path="data/traces.db"):
    if not isinstance(agent, SandboxAgent):
        raise TypeError("This milestone supports blackbox.SandboxAgent only. Generic LangGraph/OpenTelemetry adapters are not implemented.")
    return WrappedAgent(agent, db_path)
