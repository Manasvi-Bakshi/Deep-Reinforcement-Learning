from fastapi import FastAPI
from server.pkt_schd_rl_environment import PacketSchedEnv
from models import PacketAction

app = FastAPI()
env = PacketSchedEnv(task="easy")


@app.post("/reset")
def reset():
    result = env.reset()
    return {
        "observation": {"observation": result.observation.model_dump()},
        "reward": result.reward,
        "done": result.done
    }


@app.post("/step")
def step(action: dict):
    act = PacketAction(**action["action"])
    result = env.step(act)

    return {
        "observation": {"observation": result.observation.model_dump()},
        "reward": result.reward,
        "done": result.done
    }
