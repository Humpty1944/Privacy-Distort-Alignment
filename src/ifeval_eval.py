from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

CheckerFn = Callable[[str, dict], bool]


@dataclass
class IFEvalInstance:
    prompt: str
    instruction_ids: List[str]
    kwargs_list: List[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.kwargs_list:
            self.kwargs_list = [{} for _ in self.instruction_ids]
        if len(self.kwargs_list) != len(self.instruction_ids):
            raise ValueError(
                f"instruction_ids/kwargs_list length mismatch "
                f"({len(self.instruction_ids)} vs {len(self.kwargs_list)})"
            )


def _loose_variants(response: str) -> List[str]:
    r = response.strip()
    lines = r.split("\n")
    candidates = [
        r,
        "\n".join(lines[1:]).strip(),
        "\n".join(lines[:-1]).strip(),
        "\n".join(lines[1:-1]).strip(),
    ]
    candidates += [c.replace("*", "") for c in candidates]

    variants = []
    seen = set()
    for i, c in enumerate(candidates):
        if c in seen or (c == "" and i != 0):
            continue
        seen.add(c)
        variants.append(c)
    return variants


def follows_instruction(
    response: str,
    instruction_id: str,
    kwargs: dict,
    checkers: Dict[str, CheckerFn],
    loose: bool = False,
) -> bool:
    checker = checkers.get(instruction_id)
    if checker is None:
        raise KeyError(
            f"no checker for instruction id {instruction_id!r}; known: {sorted(checkers)}"
        )
    if not loose:
        return bool(checker(response, kwargs))
    return any(bool(checker(v, kwargs)) for v in _loose_variants(response))


def _instruction_result(
    response: str, instruction_id: str, kwargs: dict, checkers: Dict[str, CheckerFn]
) -> Tuple[bool, bool]:
    strict = follows_instruction(response, instruction_id, kwargs, checkers)
    loose = strict or follows_instruction(response, instruction_id, kwargs, checkers, loose=True)
    return strict, loose


def evaluate_ifeval(
    records: Sequence[Tuple[IFEvalInstance, str]],
    checkers: Dict[str, CheckerFn],
) -> dict:
    n_prompts = n_inst = 0
    prompt_strict = prompt_loose = inst_strict = inst_loose = 0
    for inst, response in records:
        if not inst.instruction_ids:
            continue
        n_prompts += 1
        results = [
            _instruction_result(response, iid, kw, checkers)
            for iid, kw in zip(inst.instruction_ids, inst.kwargs_list)
        ]
        n_inst += len(results)
        inst_strict += sum(strict for strict, _ in results)
        inst_loose += sum(loose for _, loose in results)
        prompt_strict += all(strict for strict, _ in results)
        prompt_loose += all(loose for _, loose in results)

    def frac(num: int, den: int) -> float:
        return num / den if den else math.nan

    accs = {
        "prompt_level_strict_acc": frac(prompt_strict, n_prompts),
        "prompt_level_loose_acc": frac(prompt_loose, n_prompts),
        "instruction_level_strict_acc": frac(inst_strict, n_inst),
        "instruction_level_loose_acc": frac(inst_loose, n_inst),
    }
    finite = [v for v in accs.values() if not math.isnan(v)]
    return {
        **accs,
        "ifeval_overall": sum(finite) / len(finite) if finite else math.nan,
        "n_prompts": n_prompts,
        "n_instructions": n_inst,
    }


def default_checkers() -> Dict[str, CheckerFn]:
    try:
        from .ifeval_checks import CHECKERS
    except ImportError:
        from ifeval_checks import CHECKERS
    return dict(CHECKERS)
