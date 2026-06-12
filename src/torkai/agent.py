from dataclasses import dataclass
from pathlib import Path


@dataclass
class AgentResult:
    action: str
    output: str


class TorkAgent:
    def __init__(self, workspace="agent_workspace"):
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)

    def run(self, command: str) -> AgentResult:
        text = command.strip()
        lower = text.lower()

        if lower.startswith("calculate:"):
            expr = text.split(":", 1)[1].strip()
            return self.calculate(expr)

        if lower.startswith("note:"):
            note = text.split(":", 1)[1].strip()
            return self.save_note(note)

        if lower.startswith("plan:"):
            goal = text.split(":", 1)[1].strip()
            return self.make_plan(goal)

        return AgentResult(
            action="none",
            output="I can act with commands like calculate:, note:, or plan: for now.",
        )

    def calculate(self, expr: str) -> AgentResult:
        allowed = set("0123456789+-*/(). %")
        if not set(expr) <= allowed:
            return AgentResult("calculate", "Only basic math symbols are allowed.")
        try:
            value = eval(expr, {"__builtins__": {}}, {})
            return AgentResult("calculate", str(value))
        except Exception as exc:
            return AgentResult("calculate", f"Error: {exc}")

    def save_note(self, note: str) -> AgentResult:
        path = self.workspace / "notes.txt"
        with path.open("a", encoding="utf-8") as f:
            f.write(note + "\n")
        return AgentResult("note", f"Saved note to {path}")

    def make_plan(self, goal: str) -> AgentResult:
        steps = [
            f"Goal: {goal}",
            "1. Understand the goal clearly.",
            "2. Break it into small tasks.",
            "3. Choose which task can be done with available tools.",
            "4. Run the safe action.",
            "5. Check result and continue.",
        ]
        return AgentResult("plan", "\n".join(steps))
