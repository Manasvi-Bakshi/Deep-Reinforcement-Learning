import os
import requests
from typing import List, Optional
from openai import OpenAI

BASE_ENV_URL = os.getenv("BASE_ENV_URL", "http://localhost:8000")
MAX_STEPS = int(os.getenv("MAX_STEPS", 50))
SUCCESS_THRESHOLD = float(os.getenv("SUCCESS_THRESHOLD", 0.6))

TASK_NAME = "packet_scheduling"
BENCHMARK = "openenv_packet_env"

# 🔥 STRICT: MUST USE PROVIDED PROXY (NO FALLBACKS)
client = OpenAI(
    base_url=os.environ["API_BASE_URL"],
    api_key=os.environ["API_KEY"],
)

MODEL_NAME = os.environ["MODEL_NAME"]  # 🔥 also required


def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# 🔥 GUARANTEED PROXY CALL (NO SILENT FAIL)
def force_llm_call(obs):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Return ONLY a number between 0 and 1."},
            {"role": "user", "content": f"{obs['q_priority']},{obs['q_regular']}"}
        ],
        max_tokens=5,
    )

    text = response.choices[0].message.content.strip()
    return max(0.0, min(1.0, float(text)))


def detect_regime(obs, history):
    q_p = obs["q_priority"]
    q_r = obs["q_regular"]
    incoming = obs["incoming"]
    fairness = obs["fairness_index"]
    loss = obs["loss_rate"]

    n = len(history)
    total_p = q_p
    total_r = q_r

    for i in range(max(0, n - 4), n):
        h = history[i]
        total_p += h["q_priority"]
        total_r += h["q_regular"]

    denom = min(5, n + 1)
    avg_q_p = total_p / denom
    avg_q_r = total_r / denom

    total_q = avg_q_p + avg_q_r + 1e-6
    p_pressure = avg_q_p / total_q

    if incoming > 14 and loss > 0.05:
        return "throughput_race"
    if p_pressure > 0.58 and avg_q_p > avg_q_r * 1.4:
        return "priority_flood"
    if p_pressure < 0.42 and avg_q_r > avg_q_p * 1.4:
        return "regular_surge"
    if fairness < 0.72 and total_q > 4.0:
        return "fairness_stress"

    return "balanced"


def heuristic_action(obs, history, prev_ratio):
    q_p = obs["q_priority"]
    q_r = obs["q_regular"]

    total = q_p + q_r + 1e-6
    base = q_p / total

    return max(0.0, min(1.0, 0.8 * base + 0.2 * prev_ratio))


def safe_post(session, url, payload=None):
    try:
        res = session.post(url, json=payload, timeout=10)
        res.raise_for_status()
        return res.json(), None
    except Exception as e:
        return None, str(e)


def main():
    rewards = []
    total_reward = 0.0
    steps_taken = 0
    success = False
    score = 0.0

    obs_history = []
    prev_ratio = 0.5

    log_start(TASK_NAME, BENCHMARK, "proxy-llm-agent")

    session = requests.Session()

    try:
        data, err = safe_post(session, f"{BASE_ENV_URL}/reset")
        if err:
            log_step(0, "error", 0.0, True, err)
            log_end(False, 0, 0.0, [])
            return

        obs = data["observation"]["observation"]

        # 🔥 MUST SUCCEED (NO TRY/EXCEPT)
        llm_ratio = force_llm_call(obs)

        for step in range(1, MAX_STEPS + 1):
            base_action = heuristic_action(obs, obs_history, prev_ratio)

            # blend LLM influence
            action_val = 0.7 * base_action + 0.3 * llm_ratio
            prev_ratio = action_val

            data, err = safe_post(
                session,
                f"{BASE_ENV_URL}/step",
                {"action": {"priority_ratio": action_val}}
            )

            if err:
                log_step(step, "error", 0.0, True, err)
                break

            obs = data["observation"]["observation"]
            reward = float(data["reward"])
            done = data["done"]

            obs_history.append(obs)
            if len(obs_history) > 8:
                obs_history.pop(0)

            rewards.append(reward)
            total_reward += reward
            steps_taken = step

            log_step(step, f"{action_val:.2f}", reward, done, None)

            if done:
                break

        max_total_reward = MAX_STEPS * 5.0
        score = total_reward / max_total_reward
        score = max(0.0, min(1.0, score))

        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        log_step(steps_taken, "error", 0.0, True, str(e))

    finally:
        session.close()
        log_end(success, steps_taken, score, rewards)


if __name__ == "__main__":
    main()
