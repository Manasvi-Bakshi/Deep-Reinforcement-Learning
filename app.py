from fastapi import FastAPI, Request
from server.pkt_schd_rl_environment import PacketSchedEnv
from models import PacketAction
import subprocess

app = FastAPI()

# single env instance (deterministic)
env = PacketSchedEnv(task="easy")


# -----------------------------
# RESET
# -----------------------------
@app.post("/reset")
def reset():
    result = env.reset()
    return {
        "observation": {
            "observation": result.observation.model_dump()
        },
        "reward": result.reward,
        "done": result.done
    }


# -----------------------------
# STEP
# -----------------------------
@app.post("/step")
def step(action: dict):
    act = PacketAction(**action["action"])
    result = env.step(act)

    return {
        "observation": {
            "observation": result.observation.model_dump()
        },
        "reward": result.reward,
        "done": result.done
    }


# -----------------------------
# TASKS (REQUIRED)
# -----------------------------
@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {"name": "easy", "action_schema": {"priority_ratio": "float (0 to 1)"}},
            {"name": "moderate", "action_schema": {"priority_ratio": "float (0 to 1)"}},
            {"name": "hard", "action_schema": {"priority_ratio": "float (0 to 1)"}},
        ]
    }


# -----------------------------
# GRADER (SPEC COMPLIANT)
# -----------------------------
@app.post("/grader")
async def grader(request: Request):
    data = await request.json()
    rewards = data.get("rewards", [])

    if not rewards:
        return {"score": 0.0}

    max_total = len(rewards) * 5.0  # upper bound
    score = sum(rewards) / max_total
    score = max(0.0, min(1.0, score))

    return {"score": score}


# -----------------------------
# BASELINE
# -----------------------------
@app.post("/baseline")
def baseline():
    result = subprocess.run(
        ["python", "inference.py"],
        capture_output=True,
        text=True
    )
    return {"output": result.stdout}
