"""Load DiamondIQ PA transition tables into dense NumPy arrays."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import settings

BASE_STATES = ["---", "1--", "-2-", "--3", "12-", "1-3", "-23", "123"]
BASE_INDEX = {b: i for i, b in enumerate(BASE_STATES)}
N_BASE = 8
N_OUT = 3  # 0,1,2 transient
N_STATE = N_OUT * N_BASE  # 24
OUTCOME_CLASSES = ["K", "BB_HBP", "1B", "2B", "3B", "HR", "OUT_IP"]
OUTCOME_INDEX = {c: i for i, c in enumerate(OUTCOME_CLASSES)}


def state_index(outs: int, bases: str) -> int:
    return outs * N_BASE + BASE_INDEX[bases]


def decode_state(idx: int) -> tuple[int, str]:
    return idx // N_BASE, BASE_STATES[idx % N_BASE]


@dataclass
class TransitionModel:
    """For each (outcome, start_state): list of (end_state_or_-1, runs, prob).

    end_state = -1 means inning over (3 outs).
    Arrays packed for speed:
      next_idx[o, s, k]  : next state index or -1
      next_runs[o, s, k]
      next_prob[o, s, k]
      n_trans[o, s]
    """

    next_idx: np.ndarray
    next_runs: np.ndarray
    next_prob: np.ndarray
    n_trans: np.ndarray
    max_k: int
    source: str

    def expected_from_outcome(self, outcome_i: int, state_i: int) -> tuple[float, np.ndarray, float]:
        """Return (E[runs], P(next_state[24]), P(inning_end))."""
        k = int(self.n_trans[outcome_i, state_i])
        er = 0.0
        p_next = np.zeros(N_STATE, dtype=np.float64)
        p_end = 0.0
        for j in range(k):
            p = float(self.next_prob[outcome_i, state_i, j])
            r = float(self.next_runs[outcome_i, state_i, j])
            nxt = int(self.next_idx[outcome_i, state_i, j])
            er += p * r
            if nxt < 0:
                p_end += p
            else:
                p_next[nxt] += p
        return er, p_next, p_end


def _lookup_key(table: dict, outcome: str, outs: int, bases: str) -> list[dict] | None:
    for key in (f"{outcome}|{outs}|{bases}", f"{outcome}|{outs}|*"):
        cell = table.get(key)
        if cell is not None:
            return cell["transitions"]
    # broader backoff
    for key in (f"{outcome}|*|*",):
        cell = table.get(key)
        if cell is not None:
            return cell["transitions"]
    return None


def load_transitions(path: Path | None = None) -> TransitionModel:
    path = path or (settings.vendor_models_dir / "pa_transitions_v1.json")
    raw = json.loads(path.read_text())
    body = raw.get("body") or raw
    table = body["payload"]["table"]

    # First pass: find max transitions per cell
    max_k = 1
    cells: dict[tuple[int, int], list] = {}
    for oi, outcome in enumerate(OUTCOME_CLASSES):
        for si in range(N_STATE):
            outs, bases = decode_state(si)
            trans = _lookup_key(table, outcome, outs, bases)
            if not trans:
                # Deterministic fallbacks for missing cells
                if outcome == "K":
                    new_outs = outs + 1
                    trans = [{"outs": new_outs if new_outs < 3 else 3, "bases": bases if new_outs < 3 else "---",
                              "runs": 0, "prob": 1.0}]
                elif outcome == "BB_HBP":
                    trans = _default_walk(outs, bases)
                elif outcome == "HR":
                    runs = 1 + sum(1 for c in bases if c != "-")
                    trans = [{"outs": outs, "bases": "---", "runs": runs, "prob": 1.0}]
                elif outcome == "OUT_IP":
                    new_outs = outs + 1
                    trans = [{"outs": new_outs if new_outs < 3 else 3, "bases": bases if new_outs < 3 else "---",
                              "runs": 0, "prob": 1.0}]
                else:
                    # single-like fallback
                    trans = _default_single(outs, bases)
            # renormalize
            sprob = sum(t.get("prob", 0) for t in trans) or 1.0
            normed = []
            for t in trans:
                o2 = int(t["outs"])
                b2 = t["bases"]
                if o2 >= 3:
                    nxt = -1
                else:
                    if b2 not in BASE_INDEX:
                        b2 = "---"
                    nxt = state_index(o2, b2)
                normed.append((nxt, float(t.get("runs", 0)), float(t.get("prob", 0)) / sprob))
            cells[(oi, si)] = normed
            max_k = max(max_k, len(normed))

    next_idx = np.full((len(OUTCOME_CLASSES), N_STATE, max_k), -1, dtype=np.int16)
    next_runs = np.zeros((len(OUTCOME_CLASSES), N_STATE, max_k), dtype=np.float32)
    next_prob = np.zeros((len(OUTCOME_CLASSES), N_STATE, max_k), dtype=np.float32)
    n_trans = np.zeros((len(OUTCOME_CLASSES), N_STATE), dtype=np.int16)
    for (oi, si), normed in cells.items():
        n_trans[oi, si] = len(normed)
        for j, (nxt, r, p) in enumerate(normed):
            next_idx[oi, si, j] = nxt
            next_runs[oi, si, j] = r
            next_prob[oi, si, j] = p

    return TransitionModel(next_idx, next_runs, next_prob, n_trans, max_k, str(path))


def _default_walk(outs: int, bases: str) -> list[dict]:
    # Force batter to first; push runners if forced
    occ = [c != "-" for c in bases]
    # bases string positions: 0=1st, 1=2nd, 2=3rd
    first, second, third = occ
    runs = 0
    if first and second and third:
        runs = 1
        third = True
        second = True
        first = True
    elif first and second:
        third = True
        second = True
        first = True
    elif first:
        second = True
        first = True
    else:
        first = True
    nb = ("1" if first else "-") + ("2" if second else "-") + ("3" if third else "-")
    return [{"outs": outs, "bases": nb, "runs": runs, "prob": 1.0}]


def _default_single(outs: int, bases: str) -> list[dict]:
    # Batter to first; runners +1 base (simplified)
    runners = []
    if bases[0] != "-":
        runners.append(1)
    if bases[1] != "-":
        runners.append(2)
    if bases[2] != "-":
        runners.append(3)
    runs = 0
    new_runners = []
    for r in runners:
        nr = r + 1
        if nr >= 4:
            runs += 1
        else:
            new_runners.append(nr)
    new_runners.append(1)  # batter
    first = 1 in new_runners
    second = 2 in new_runners
    third = 3 in new_runners
    nb = ("1" if first else "-") + ("2" if second else "-") + ("3" if third else "-")
    return [{"outs": outs, "bases": nb, "runs": runs, "prob": 1.0}]


def build_player_kernels(probs: np.ndarray, trans: TransitionModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For one player with outcome probs shape (7,), build:
    R[s] = expected runs on this PA from state s
    P[s, s'] = P(next state s' | start s)  (mass may be <1; remainder = inning end)
    E[s] = P(inning ends)
    """
    R = np.zeros(N_STATE, dtype=np.float64)
    P = np.zeros((N_STATE, N_STATE), dtype=np.float64)
    E = np.zeros(N_STATE, dtype=np.float64)
    for si in range(N_STATE):
        er = 0.0
        p_next = np.zeros(N_STATE, dtype=np.float64)
        p_end = 0.0
        for oi, p_out in enumerate(probs):
            if p_out <= 0:
                continue
            k = int(trans.n_trans[oi, si])
            for j in range(k):
                p = p_out * float(trans.next_prob[oi, si, j])
                r = float(trans.next_runs[oi, si, j])
                nxt = int(trans.next_idx[oi, si, j])
                er += p * r
                if nxt < 0:
                    p_end += p
                else:
                    p_next[nxt] += p
        # renormalize tiny drift
        mass = p_next.sum() + p_end
        if mass > 0 and abs(mass - 1.0) > 1e-8:
            p_next /= mass
            p_end /= mass
        R[si] = er
        P[si] = p_next
        E[si] = p_end
    return R, P, E
