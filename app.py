from fastapi import FastAPI, Request
from server.pkt_schd_rl_environment import PacketSchedEnv
from models import PacketAction
import subprocess
import math

app = FastAPI()

# deterministic env instance
env = PacketSchedEnv(task="easy")


# -----------------------------
# RESET
# -----------------------------
@app.post("/reset")
async def reset(request: Request):
    global env

    try:
        body = await request.json()
        task = body.get("task", "easy")
    except Exception:
        task = "easy"

    # reinitialize env with task
    env = PacketSchedEnv(task=task)

    result = env.reset()

    return {
        "observation": result.observation.model_dump(),
        "reward": float(result.reward),
        "done": bool(result.done),
        "info": {}
    }


# -----------------------------
# STEP
# -----------------------------
@app.post("/step")
def step(action: dict):
    act = PacketAction(**action["action"])
    result = env.step(act)

    return {
        "observation": result.observation.model_dump(),
        "reward": float(result.reward),
        "done": bool(result.done),
        "info": {}
    }


# -----------------------------
# TASKS (STRICT SCHEMA)
# -----------------------------
@app.get("/tasks")
def tasks():
    schema = {
        "type": "object",
        "properties": {
            "priority_ratio": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0
            }
        },
        "required": ["priority_ratio"]
    }

    return {
        "tasks": [
            {
                "name": "easy",
                "description": "Low traffic, stable conditions",
                "action_schema": schema
            },
            {
                "name": "moderate",
                "description": "Bursty traffic with regime shifts",
                "action_schema": schema
            },
            {
                "name": "hard",
                "description": "Adversarial congestion + fairness constraints",
                "action_schema": schema
            }
        ]
    }


# -----------------------------
# GRADER (STRICT (0,1))
# -----------------------------
@app.post("/grader")
async def grader(request: Request):
    data = await request.json()
    rewards = data.get("rewards", [])

    if not rewards:
        return {"score": 0.001}  # NEVER 0

    total = sum(rewards)

    # smooth deterministic normalization
    score = 1 / (1 + math.exp(-total / 20))

    # enforce STRICT bounds
    score = max(0.001, min(0.999, score))

    return {"score": float(score)}


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

@app.get("/")
def root():
    return {"status": "ok"}
