from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .client import ArbiterClient as Arbiter
from .color import segment_stats, stats_distance, to_array, to_image
from .operators import (
    TOOL_HELP,
    TOOL_RANGES,
    apply_call,
    normalise_call,
    tools_for_segment,
)
from .vision_chat import VisionChat, parse_json_object

log = logging.getLogger(__name__)

DIMENSIONS = ("content", "lighting", "color_harmony", "composition", "depth_of_field", "technical_quality")
SCORE_EPSILON = 1e-4
DISTANCE_TOLERANCE = 0.5
# one Lab unit closer to the reference is worth this much scorer gain; lets a
# score-neutral fill light on a dark face through while a big score jump that
# drifts from the reference (the saturation exploit) is still refused
DISTANCE_WEIGHT = 0.015
MAX_CONSECUTIVE_REJECTIONS = 14
STUBBORN_DIMENSION_AFTER = 4
BAN_AFTER_REJECTIONS = 2
TOOL_BUDGET = 1.0
STRENGTH_VARIANTS = (1.0, 0.5, 1.5)


# ##################################################################
# climb state
# everything the loop mutates: current frame + scores + distance, spend per
# tool, rejection tallies, and the audit log of every attempt.
@dataclass
class ClimbState:
    rgb: np.ndarray
    scores: dict[str, float]
    distance: float
    spent: dict[tuple[str, str], float] = field(default_factory=dict)
    rejections: dict[tuple[str, str, int], int] = field(default_factory=dict)
    banned: set[tuple[str, str, int]] = field(default_factory=set)
    streak_by_dimension: dict[str, int] = field(default_factory=lambda: {d: 0 for d in DIMENSIONS})
    consecutive_rejections: int = 0
    accepted: list[dict] = field(default_factory=list)
    attempts: list[dict] = field(default_factory=list)


# ##################################################################
# climb result
# the graded frame plus the audit trail.
@dataclass(frozen=True)
class ClimbResult:
    image: Image.Image
    scores: dict[str, float]
    distance: float
    accepted: list[dict]
    attempts: list[dict]
    transcript: list[dict]


# ##################################################################
# format scores
def format_scores(scores: dict[str, float]) -> str:
    return ", ".join(f"{key}={scores[key]:.3f}" for key in ("overall_aesthetic", *DIMENSIONS))


# ##################################################################
# segment crop
# a segment on neutral grey, cropped to its bounding box, for the model.
def segment_crop(image: Image.Image, mask: np.ndarray) -> Image.Image:
    rows, cols = np.where(mask > 0.5)
    box = (int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1)
    rgb = to_array(image)
    composite = rgb * mask[..., None] + 0.5 * (1 - mask[..., None])
    return to_image(composite).crop(box)


# ##################################################################
# delta text
# per-segment "current minus reference" with plain-language hints about
# which tools move each axis the right way.
def delta_text(rgb: np.ndarray, masks: dict[str, np.ndarray], reference: dict[str, dict[str, float]]) -> str:
    lines = []
    for segment, mask in masks.items():
        stats = segment_stats(rgb, mask)
        ref = reference[segment]
        d = {key: stats[key] - ref[key] for key in stats}
        hints = []
        if abs(d["L"]) > 2:
            hints.append(
                "brighten (exposure +, fill_light, lift_shadows)"
                if d["L"] < 0
                else "darken (exposure -, recover_highlights)"
            )
        if abs(d["a"]) > 2:
            hints.append(
                "LESS red (white_balance tint -, skin_tone_refine)"
                if d["a"] > 0
                else "more red/magenta (white_balance tint +)"
            )
        if abs(d["b"]) > 2:
            hints.append("cool (white_balance -)" if d["b"] > 0 else "warm (white_balance +)")
        if abs(d["L_std"]) > 2:
            hints.append("MORE contrast" if d["L_std"] < 0 else "LESS contrast (negative contrast)")
        if abs(d["chroma"]) > 2:
            hints.append(
                "MORE saturation/vibrance" if d["chroma"] < 0 else "LESS saturation (negative saturation), do NOT add"
            )
        lines.append(
            f"  {segment:10s}: L {d['L']:+.1f}, a {d['a']:+.1f}, b {d['b']:+.1f}, contrast {d['L_std']:+.1f}, "
            f"chroma {d['chroma']:+.1f}; distance {stats_distance(stats, ref):.1f} -> "
            + ("; ".join(hints) if hints else "close to reference")
        )
    return "\n".join(lines)


# ##################################################################
# tool text
def tool_text() -> str:
    lines = []
    for segment in ("person", "background"):
        for name in tools_for_segment(segment):
            low, high = TOOL_RANGES[name]
            lines.append(f"- segment={segment} tool={name} amount={low}..{high}: {TOOL_HELP[name]}")
    return "\n".join(lines)


# ##################################################################
# opening text
def opening_text(scores: dict, target: dict, lowest: str, deltas: str) -> str:
    return f"""We are improving ONE photograph together, one edit at a time, using per-segment photometric edits only.
The people must stay exactly who they are (no redrawing; we only adjust tone, colour and texture of the original pixels).

An automatic aesthetic scorer rates the image on 7 dimensions:
  current : {format_scores(scores)}
  target  : {format_scores(target)}   (scores of the reference render, image [4])

Image [4] is a REFERENCE RENDER of this same scene made by a generative model, showing the look we want: how the
subject is lit, the skin tone, the background exposure, colour, contrast and depth of field. It redrew the people,
so NEVER try to match their faces or features - only lighting, colour, tone, contrast and background treatment.

How this works:
- Each turn you propose exactly ONE tool call on exactly ONE segment ('person' or 'background'). No global edits exist.
- I apply it, re-score, and measure how far each segment's tone/colour statistics are from the reference.
  An edit is KEPT only if it improves the objective (overall_aesthetic gain plus {DISTANCE_WEIGHT} per Lab unit of
  reference distance removed) AND does not move further from the reference. Otherwise it is REVERTED and you must
  try something different (other tool, segment, or direction). Moving clearly toward the reference counts even when
  the scorer is flat.
- Each turn I tell you the LOWEST scorer dimension and the per-segment deltas vs the reference (current minus reference).
  Propose the single best step that shrinks those deltas AND raises the lowest dimension. The background usually
  drives lighting, color_harmony and depth_of_field - do not neglect it.
- Every (segment, tool) pair has a cumulative strength budget of {TOOL_BUDGET}; once spent you must use other tools.
  A proposal rejected {BAN_AFTER_REJECTIONS} times in the same direction is banned for the rest of the run.
- Small cumulative edits beat one huge edit; amounts 0.2-0.7 usually behave well.

Per-segment deltas vs reference now:
{deltas}

Tools (every tool takes exactly one strength key, "amount"; white_balance may add "tint", selective_color adds "target"):
{tool_text()}

Images: [1] current frame, [2] person segment on grey, [3] background segment on grey, [4] reference render.
The LOWEST dimension right now is **{lowest}** ({scores[lowest]:.3f}). Propose your first step.
Reply with ONLY a JSON object, e.g. {{"segment": "background", "tool": "exposure", "amount": -0.4, "why": "..."}}"""


# ##################################################################
# verdict text
def verdict_text(attempt: dict, scores: dict, lowest: str, focus: str, deltas: str, notes: str) -> str:
    call = attempt["call"]
    params = {k: v for k, v in call.items() if k not in ("segment", "tool")}
    if attempt["status"] == "accepted":
        head = (
            f"KEPT: {call['segment']}.{call['tool']} {json.dumps(params)} improved the objective "
            f"(score {attempt['gained']:+.4f}, reference distance {attempt['distance'] - attempt['previous_distance']:+.1f}); "
            f"overall now {scores['overall_aesthetic']:.4f}.\n  scores now: {format_scores(scores)}"
        )
    else:
        head = (
            f"REVERTED: {call['segment']}.{call['tool']} {json.dumps(params)} - {attempt['reason']} "
            f"(overall would have been {attempt['scores']['overall_aesthetic']:.4f} vs current "
            f"{scores['overall_aesthetic']:.4f}).\n  scores unchanged: {format_scores(scores)}"
        )
    return (
        f"{head}\n\nPer-segment deltas vs reference now:\n{deltas}\n{notes}\n"
        f"The LOWEST dimension is now **{lowest}** ({scores[lowest]:.3f}). {focus}\n"
        f"Images: [1] updated frame, [2] person segment, [3] background segment, [4] reference render (unchanged).\n"
        f"Propose the next single best step. Reply with ONLY a JSON object."
    )


# ##################################################################
# guided climb
# the conversation: the model proposes one masked edit per turn, the
# scorer and the reference-distance guard decide whether it stays. Runs
# until max_turns or the model can no longer find an accepted step.
class GuidedClimb:
    def __init__(
        self, arbiter: Arbiter, chat: VisionChat, masks: dict[str, np.ndarray], reference: Image.Image
    ) -> None:
        self.arbiter = arbiter
        self.chat = chat
        self.masks = masks
        self.reference = reference
        self.reference_stats = {seg: segment_stats(to_array(reference), mask) for seg, mask in masks.items()}
        self.target_scores = arbiter.aesthetic_score(reference)

    # ##################################################################
    # run
    def run(self, image: Image.Image, max_turns: int) -> ClimbResult:
        rgb = to_array(image)
        state = ClimbState(rgb=rgb, scores=self.arbiter.aesthetic_score(image), distance=self.total_distance(rgb))
        log.info("climb start: %s distance=%.1f", format_scores(state.scores), state.distance)
        pending: dict | None = None
        for turn in range(1, max_turns + 1):
            lowest, focus = self.choose_dimension(state)
            self.prompt(state, pending, lowest, focus)
            pending = None
            call = self.propose(turn)
            if call is None:
                continue
            pending = self.evaluate(state, call, lowest)
            self.record(state, pending, turn)
            if state.consecutive_rejections >= MAX_CONSECUTIVE_REJECTIONS:
                log.info("climb stop: %d consecutive rejections", state.consecutive_rejections)
                break
        return ClimbResult(
            to_image(state.rgb), state.scores, state.distance, state.accepted, state.attempts, self.chat.transcript()
        )

    # ##################################################################
    # total distance
    def total_distance(self, rgb: np.ndarray) -> float:
        return sum(
            stats_distance(segment_stats(rgb, mask), self.reference_stats[seg]) for seg, mask in self.masks.items()
        )

    # ##################################################################
    # choose dimension
    # the lowest-scoring dimension, unless it has resisted several tries,
    # in which case the next lowest gets a turn.
    def choose_dimension(self, state: ClimbState) -> tuple[str, str]:
        ranked = sorted(DIMENSIONS, key=lambda d: state.scores[d])
        for dimension in ranked:
            if state.streak_by_dimension[dimension] < STUBBORN_DIMENSION_AFTER:
                if dimension == ranked[0]:
                    return dimension, ""
                return (
                    dimension,
                    f"(The absolute lowest, {ranked[0]}, has resisted recent attempts, so target {dimension} this turn.)",
                )
        return ranked[0], ""

    # ##################################################################
    # prompt
    # opening turn or verdict-on-last-turn, always with fresh images.
    def prompt(self, state: ClimbState, pending: dict | None, lowest: str, focus: str) -> None:
        image = to_image(state.rgb)
        images = [
            image,
            segment_crop(image, self.masks["person"]),
            segment_crop(image, self.masks["background"]),
            self.reference,
        ]
        deltas = delta_text(state.rgb, self.masks, self.reference_stats)
        if pending is None:
            self.chat.user(opening_text(state.scores, self.target_scores, lowest, deltas), images)
            return
        self.chat.user(verdict_text(pending, state.scores, lowest, focus, deltas, self.notes(state)), images)

    # ##################################################################
    # notes
    # exhausted budgets and bans, spelled out so the model stops proposing them.
    def notes(self, state: ClimbState) -> str:
        spent = [f"{s}.{t}" for (s, t), v in state.spent.items() if v >= TOOL_BUDGET - 1e-6]
        banned = [f"{s}.{t}({'+' if sign > 0 else '-'})" for s, t, sign in sorted(state.banned)]
        lines = []
        if spent:
            lines.append("Budget exhausted (not available again): " + ", ".join(spent))
        if banned:
            lines.append("BANNED this run: " + ", ".join(banned))
        return "\n".join(lines)

    # ##################################################################
    # propose
    # one model turn -> a normalised call, or None after telling the model
    # what was wrong with its reply.
    def propose(self, turn: int) -> dict | None:
        reply = self.chat.ask()
        self.chat.assistant(reply)
        try:
            call = normalise_call(parse_json_object(reply))
        except (ValueError, json.JSONDecodeError) as err:
            log.info("[%d] invalid proposal: %s", turn, err)
            self.chat.user(
                f"INVALID: {err}. Choose a listed tool on 'person' or 'background'. Reply with ONLY one JSON object."
            )
            return None
        call["why"] = parse_json_object(reply).get("why", "")
        return call

    # ##################################################################
    # evaluate
    # try the proposed strength plus a weaker and stronger variant within
    # budget; keep the best that passes the distance guard, and decide.
    def evaluate(self, state: ClimbState, call: dict, lowest: str) -> dict:
        key = (call["segment"], call["tool"])
        sign = 1 if call["amount"] >= 0 else -1
        if (*key, sign) in state.banned:
            return self.refusal(state, call, lowest, "that tool and direction is BANNED this run")
        remaining = TOOL_BUDGET - state.spent.get(key, 0.0)
        if remaining <= 1e-6:
            return self.refusal(state, call, lowest, "budget for that tool on that segment is exhausted")
        best = None
        for factor in STRENGTH_VARIANTS:
            variant = self.scaled(call, factor, remaining)
            if variant is None:
                continue
            rgb = apply_call(state.rgb.copy(), self.masks, variant)
            scores = self.arbiter.aesthetic_score(to_image(rgb))
            distance = self.total_distance(rgb)
            gained = scores["overall_aesthetic"] - state.scores["overall_aesthetic"]
            drifted = distance - state.distance
            objective = gained - DISTANCE_WEIGHT * drifted
            rank = (drifted <= DISTANCE_TOLERANCE, objective)
            if best is None or rank > best["rank"]:
                best = {
                    "call": variant,
                    "rgb": rgb,
                    "scores": scores,
                    "distance": distance,
                    "rank": rank,
                    "gained": gained,
                    "objective": objective,
                    "previous_distance": state.distance,
                }
        drifted = best["distance"] - state.distance
        reason = ""
        if drifted > DISTANCE_TOLERANCE:
            reason = f"it moved AWAY from the reference (distance {drifted:+.1f}, score {best['gained']:+.4f})"
        elif best["objective"] <= SCORE_EPSILON:
            reason = f"no net improvement (score {best['gained']:+.4f}, distance {drifted:+.1f})"
        best["call"]["why"] = call.get("why", "")
        return {**best, "status": "accepted" if not reason else "rejected", "reason": reason, "dimension": lowest}

    # ##################################################################
    # refusal
    # a rejected attempt that was never applied.
    def refusal(self, state: ClimbState, call: dict, lowest: str, reason: str) -> dict:
        return {
            "call": call,
            "status": "rejected",
            "reason": reason,
            "scores": state.scores,
            "distance": state.distance,
            "previous_distance": state.distance,
            "dimension": lowest,
            "gained": 0.0,
            "objective": 0.0,
        }

    # ##################################################################
    # scaled
    # a copy of the call at amount*factor, clipped to range and budget.
    def scaled(self, call: dict, factor: float, remaining: float) -> dict | None:
        low, high = TOOL_RANGES[call["tool"]]
        amount = float(np.clip(call["amount"] * factor, low, high))
        amount = float(np.sign(amount) * min(abs(amount), remaining))
        if factor != 1.0 and abs(amount - call["amount"]) < 1e-3:
            return None
        return {**call, "amount": amount}

    # ##################################################################
    # record
    # apply the verdict to state and log the turn.
    def record(self, state: ClimbState, attempt: dict, turn: int) -> None:
        call = attempt["call"]
        key = (call["segment"], call["tool"])
        sign = 1 if call["amount"] >= 0 else -1
        entry = {k: v for k, v in attempt.items() if k not in ("rgb", "rank")}
        entry["turn"] = turn
        state.attempts.append(entry)
        log.info(
            "[%d] %-8s %s.%s amount=%.3f overall %.4f -> %.4f dist %.1f -> %.1f %s | %s",
            turn,
            attempt["status"],
            call["segment"],
            call["tool"],
            call["amount"],
            state.scores["overall_aesthetic"],
            attempt["scores"]["overall_aesthetic"],
            state.distance,
            attempt["distance"],
            attempt["reason"],
            call.get("why", "")[:80],
        )
        if attempt["status"] == "accepted":
            state.rgb, state.scores, state.distance = attempt["rgb"], attempt["scores"], attempt["distance"]
            state.spent[key] = state.spent.get(key, 0.0) + abs(call["amount"])
            state.accepted.append(call)
            state.consecutive_rejections = 0
            state.streak_by_dimension = {d: 0 for d in DIMENSIONS}
            return
        state.consecutive_rejections += 1
        state.streak_by_dimension[attempt["dimension"]] += 1
        state.rejections[(*key, sign)] = state.rejections.get((*key, sign), 0) + 1
        if state.rejections[(*key, sign)] >= BAN_AFTER_REJECTIONS:
            state.banned.add((*key, sign))
