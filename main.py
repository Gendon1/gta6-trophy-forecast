"""Deterministic GTA6 trophy rarity forecaster (Python 3.13 standard library only).

The median follows a stable historical logit trend when there are enough reads. For flat histories without a stable
trend, per-trophy drift anticipates Exophase refresh dilution: 1.0 / 0.6 / 0.3 logit by rarity, grind traits, and
early-story/rate class. The forecast spread combines sigma_refresh 0.90, horizon-scaled random-walk uncertainty, and
a small player-count term.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in python:3.13-slim, standard library only: no
network, no pip packages. -P keeps this file's directory OFF sys.path, so a module you ship beside main.py (say
/bot/model.py) imports only after the sys.path line below, which puts the directory first; -s drops user
site-packages.

What an entry must do:
  - read every <id>.in in the input directory (the grader runs one case at a time, with only that case's file there):
    JSON {"schema": "makefun/gta6-trophies-case/v1", "id", "at", "horizon_minutes", "target", "levels", "game",
    "trophies", "history", "notes"}, where `trophies` lists the case's trophies {id, name, description, tier, points,
    secret} and `history` every read before `at`, oldest first, each {slot, at, players_tracked, trophies_earned,
    rates}, `rates` in the order of `trophies` (null where that read lacked the trophy);
  - write <id>.out to the output directory: {"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}, one row
    for each of the case's trophies, each id once, no other key, each q five finite numbers from 0 to 100 (percent),
    non-decreasing; at most 65,536 bytes. Anything else is an invalid case, scored as if every quantile equalled the
    trophy's rate in the last read, and an entry with an invalid case takes no prize.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]
SIGMA_REFRESH = 0.90
SIGMA_PER_SQRT_DAY = 0.9
BINOMIAL_SCALE = 0.5


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    pct = clamp(pct)
    return math.log(pct / (100 - pct))


def sigmoid_percent(value: float) -> float:
    """Convert logit units back to a bounded percentage without exp overflow."""
    if value >= 0:
        return 100 / (1 + math.exp(-value))
    exp_value = math.exp(value)
    return 100 * exp_value / (1 + exp_value)


def parse_time(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError):
        return None


def history_trend(history: list, trophy_index: int) -> tuple[list[float], float, bool, bool]:
    """Return observed logits, stable LS slope in logit/day, and flat/stable flags.

    Stability requires a useful linear fit (R² >= 0.5) and a consistent direction
    across most non-flat adjacent changes; this avoids extrapolating noisy reversals.
    """
    observations = []
    for read in history:
        rates = read.get("rates", [])
        rate = rates[trophy_index] if trophy_index < len(rates) else None
        if rate is None:
            continue
        timestamp = parse_time(read.get("at"))
        if timestamp is None:
            continue
        observations.append((timestamp, logit(float(rate))))

    if not observations:
        return [], 0.0, True, False

    values = [value for _, value in observations]
    flat_values = len(values) >= 2 and all(value == values[0] for value in values)
    if len(observations) < 3:
        return values, 0.0, flat_values, False

    origin = observations[0][0]
    xs = [(timestamp - origin).total_seconds() / 86400 for timestamp, _ in observations]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(values) / len(values)
    variance_x = sum((x - mean_x) ** 2 for x in xs)
    if variance_x <= 0:
        return values, 0.0, flat_values, False

    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values)) / variance_x
    residual = sum((y - (mean_y + slope * (x - mean_x))) ** 2 for x, y in zip(xs, values))
    variance_y = sum((y - mean_y) ** 2 for y in values)
    r_squared = 1.0 if variance_y == 0 else 1 - residual / variance_y

    directions = []
    for (previous_time, previous_value), (timestamp, value) in zip(observations, observations[1:]):
        elapsed = (timestamp - previous_time).total_seconds()
        change = value - previous_value
        if elapsed > 0 and abs(change) > 1e-12:
            directions.append(1 if change > 0 else -1)
    matching = sum(direction == (1 if slope > 0 else -1) for direction in directions)
    direction_stable = bool(directions) and matching >= max(2, math.ceil(0.7 * len(directions)))
    stable = abs(slope) > 1e-12 and r_squared >= 0.5 and direction_stable
    return values, slope, flat_values, stable


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round(x * 10000) / 10000 rounds it.

    A whole number comes back as an int, so json.dumps writes 100 where JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case.get("history") or []
    horizon_days = max(0.0, float(case.get("horizon_minutes", 0)) / 1440)
    last_players = history[-1].get("players_tracked", 1) if history else 1
    try:
        players = max(1, int(last_players))
    except (TypeError, ValueError):
        players = 1
    # These independent uncertainty sources are combined in quadrature on the logit scale.
    spread = math.sqrt(
        SIGMA_REFRESH ** 2
        + (SIGMA_PER_SQRT_DAY * math.sqrt(horizon_days)) ** 2
        + (BINOMIAL_SCALE / math.sqrt(players)) ** 2
    )
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        values, slope, flat_values, stable = history_trend(history, i)
        # Missing reads inherit the most recent non-null rate; absent history falls back to 1%.
        rate = None
        for read in reversed(history):
            rates = read.get("rates", [])
            if i < len(rates) and rates[i] is not None:
                rate = rates[i]
                break
        if rate is None:
            rate = 1.0

        median_logit = logit(float(rate))
        if stable:
            # Trend extrapolation is deliberately shrunk by half to limit short-sample overreach.
            median_logit += 0.5 * slope * horizon_days
        elif flat_values and values:
            # Apply 1.0 / 0.6 / 0.3 logit only to flat, unstable histories; rare/grind wins over early-story.
            # Grid search over p00 showed no single flat drift beats the house baseline at both a 25 percent uniform
            # drop and an 80 percent uniform drop, so this table is the minimax compromise - it beats the house by
            # wide margins at 40 percent, 80 percent and non-uniform drops, keeps no-drop loss below 0.50, and loses
            # to the house only on a small uniform drop.
            name_lower = str(trophy.get("name", "")).lower()
            early_story = "chapter" in name_lower or "gone fishing" in name_lower or trophy.get("id") == 2
            tier = str(trophy.get("tier", "")).lower()
            description_lower = str(trophy.get("description", "")).lower()
            grind = any(term in description_lower for term in ("100%", "250", "stunts", "gold medal"))
            rare_or_grind = (
                float(rate) < 1.0
                or tier in {"platinum", "gold", "silver"}
                or grind
            )
            if rare_or_grind:
                median_logit -= 1.0
            elif 1.0 <= float(rate) < 2.0:
                median_logit -= 0.6
            else:
                median_logit -= 0.3

        # A trophy already reported at 100% stays at the upper bound for its median.
        if float(rate) >= 100:
            median_logit = logit(99.99)

        quantiles = [clamp(sigmoid_percent(median_logit + z * spread)) for z in Z]
        # Keep rounding monotone even at the percentage bounds.
        rounded = [round4(value) for value in quantiles]
        for j in range(1, len(rounded)):
            rounded[j] = max(rounded[j], rounded[j - 1])
        rows.append({"id": trophy["id"], "q": rounded})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
