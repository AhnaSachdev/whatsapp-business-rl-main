"""
inference.py — WhatsApp Sales RL Inference Script
==================================================
Runs one episode per task (task1, task2, task3) using an LLM agent.

Required environment variables:
  API_BASE_URL   – OpenAI-compatible endpoint (e.g. https://router.huggingface.co/v1)
  MODEL_NAME     – Model identifier (e.g. Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN       – API key / Hugging Face token

STDOUT FORMAT (one [START], N [STEP]s, one [END] per episode):
  [START] task=<task_id> env=whatsapp_sales_rl model=<model_name>
  [STEP]  step=<n> action=<action_type> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

# ── make project importable from root ────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import make_env
from models import Action, Observation

# ── env vars ──────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")

BENCHMARK    = "whatsapp_sales_rl"
TASKS        = ["task1", "task2", "task3"]

# Agent behaviour knobs
TEMPERATURE  = 0.3   # lower = more deterministic action selection
MAX_TOKENS   = 256
# An episode is considered a success if the outcome is SALE or ESCALATED
SUCCESS_OUTCOMES = {"SALE", "ESCALATED"}


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert WhatsApp sales agent. Your goal is to guide the customer
toward a purchase (SALE outcome) while keeping them happy and satisfied.

At each step you receive the current conversation state and must choose exactly
one action from the list below. Reply with a valid JSON object and nothing else.

Available actions:
  ASK_QUESTION      – Ask the customer a qualifying question
  GIVE_PRICE        – Quote a price for the product
  OFFER_DISCOUNT    – Offer a discount (requires discount_pct between 1 and 50)
  PROVIDE_INFO      – Share product information or answer a query
  ESCALATE          – Escalate to a senior agent or manager
  DELAY_RESPONSE    – Ask the customer to wait (use sparingly — penalised)
  END_CONVERSATION  – End the conversation

Response format (JSON only, no extra text):
  {"action_type": "<ACTION>", "message": "<your message to the customer>"}

For OFFER_DISCOUNT also include:
  {"action_type": "OFFER_DISCOUNT", "discount_pct": <number>, "message": "<msg>"}

Strategy hints:
  - Start by asking questions to understand the customer's needs.
  - Build trust before quoting a price.
  - If the customer is skeptical, provide clear information or a small discount.
  - Avoid DELAY_RESPONSE — it lowers satisfaction.
  - Monitor obligation warnings; fulfil them promptly with PROVIDE_INFO.
  - Aim for CLOSING once sentiment is positive and conversion is high.
""").strip()


def _build_user_prompt(obs: Observation, step: int) -> str:
    history_text = "\n".join(obs.chat_history[-6:]) if obs.chat_history else "None"
    uncertainties = ", ".join(obs.uncertainties) if obs.uncertainties else "none"
    pending = obs.obligations.pending
    obligations_text = (
        "\n".join(f"  - [{o.type}] {o.description}" for o in pending)
        if pending else "  none"
    )

    return textwrap.dedent(f"""
        Step {step} — Current conversation state:

        Stage:          {obs.stage}
        Intent:         {obs.intent}
        Sentiment:      {obs.sentiment:+.2f}  (-1 = very negative, +1 = very positive)
        Uncertainties:  {uncertainties}
        Step count:     {obs.step_count}

        Pending obligations (must be fulfilled soon):
        {obligations_text}

        Last 6 conversation lines:
        {history_text}

        Choose the best action. Reply with JSON only.
    """).strip()


# ══════════════════════════════════════════════════════════════════════════════
# LLM CALL
# ══════════════════════════════════════════════════════════════════════════════

def _call_llm(client: OpenAI, obs: Observation, step: int) -> dict:
    """
    Ask the LLM for an action. Returns a dict with at minimum 'action_type'.
    Falls back to a safe default on any error.
    """
    user_prompt = _build_user_prompt(obs, step)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (completion.choices[0].message.content or "").strip()

        # Strip markdown fences if the model wraps its JSON
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        parsed = json.loads(raw)
        return parsed

    except Exception as exc:
        print(f"[DEBUG] LLM call failed at step {step}: {exc}", flush=True)
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# ACTION BUILDER
# ══════════════════════════════════════════════════════════════════════════════

VALID_ACTIONS = {
    "ASK_QUESTION", "GIVE_PRICE", "OFFER_DISCOUNT",
    "PROVIDE_INFO", "ESCALATE", "DELAY_RESPONSE", "END_CONVERSATION",
}

# Stage-aware fallback policy (used when LLM output is invalid/missing)
_STAGE_FALLBACK = {
    "GREETING":          ("ASK_QUESTION",  "How can I help you today?"),
    "DISCOVERY":         ("ASK_QUESTION",  "Could you tell me more about what you're looking for?"),
    "QUALIFICATION":     ("PROVIDE_INFO",  "Here's what makes our product a great fit for you."),
    "OBJECTION_HANDLING":("OFFER_DISCOUNT","10"),
    "NEGOTIATION":       ("OFFER_DISCOUNT","15"),
    "CLOSING":           ("PROVIDE_INFO",  "You've made a great choice!"),
    "POST_SALE":         ("PROVIDE_INFO",  "Thank you for your purchase!"),
    "ESCALATED":         ("ESCALATE",      "Let me connect you with someone senior."),
    "ENDED":             ("END_CONVERSATION","Thank you for your time."),
}


def _build_action(llm_output: dict, obs: Observation) -> Action:
    """
    Convert LLM JSON dict → validated Action.
    Falls back to the stage-aware heuristic on any validation error.
    """
    action_type = str(llm_output.get("action_type", "")).upper()
    message     = str(llm_output.get("message", ""))
    discount    = llm_output.get("discount_pct")

    # Fulfil pending obligations immediately with PROVIDE_INFO
    if obs.obligations.has_pending and action_type not in {"PROVIDE_INFO", "ASK_QUESTION", "ESCALATE"}:
        action_type = "PROVIDE_INFO"
        message     = "Let me follow up on your earlier request."

    try:
        if action_type not in VALID_ACTIONS:
            raise ValueError(f"Unknown action: {action_type}")

        if action_type == "OFFER_DISCOUNT":
            pct = float(discount) if discount is not None else 10.0
            pct = max(1.0, min(50.0, pct))   # clamp to valid range
            return Action(action_type="OFFER_DISCOUNT", message=message, discount_pct=pct)

        return Action(action_type=action_type, message=message)

    except Exception:
        # Stage-aware fallback
        fallback_type, fallback_msg = _STAGE_FALLBACK.get(
            obs.stage, ("ASK_QUESTION", "How can I help?")
        )
        if fallback_type == "OFFER_DISCOUNT":
            return Action(
                action_type="OFFER_DISCOUNT",
                message="Here's a special discount for you.",
                discount_pct=float(fallback_msg),
            )
        return Action(action_type=fallback_type, message=fallback_msg)


# ══════════════════════════════════════════════════════════════════════════════
# EPISODE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_episode(client: OpenAI, task_id: str) -> None:
    """Run one complete episode for the given task and emit structured logs."""
    rewards:      List[float] = []
    steps_taken:  int         = 0
    success:      bool        = False
    final_outcome: str        = "IN_PROGRESS"

    log_start(task=task_id, model=MODEL_NAME)

    try:
        env = make_env(task_id=task_id)
        obs = env.reset()

        while not env.state().episode_done:
            step = steps_taken + 1

            # Ask LLM for action
            llm_output = _call_llm(client, obs, step)
            action     = _build_action(llm_output, obs)

            error_str: Optional[str] = None
            try:
                obs, reward, done, info = env.step(action)
            except Exception as exc:
                # Log the error and attempt a safe fallback
                error_str = str(exc).replace("\n", " ")[:120]
                print(f"[DEBUG] env.step() error: {error_str}", flush=True)
                # Try a guaranteed-valid fallback action
                safe_action = Action(
                    action_type="ASK_QUESTION",
                    message="Could you tell me more?",
                )
                obs, reward, done, info = env.step(safe_action)

            rewards.append(reward)
            steps_taken = step
            final_outcome = info.get("outcome", "IN_PROGRESS")

            log_step(
                step=step,
                action=action.action_type,
                reward=reward,
                done=done,
                error=error_str,
            )

            if done:
                break

    except Exception as exc:
        # Catastrophic failure — still emit [END]
        print(f"[DEBUG] Episode failed: {exc}", flush=True)
        final_outcome = "IN_PROGRESS"

    success = final_outcome in SUCCESS_OUTCOMES
    log_end(success=success, steps=steps_taken, rewards=rewards)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task_id in TASKS:
        run_episode(client, task_id)


if __name__ == "__main__":
    main()