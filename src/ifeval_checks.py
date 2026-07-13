from __future__ import annotations
import re
from typing import Callable, Dict


def check_no_comma(response: str, kwargs: dict) -> bool:
    return "," not in response


def check_lowercase(response: str, kwargs: dict) -> bool:
    letters = [c for c in response if c.isalpha()]
    return len(letters) > 0 and response == response.lower()


def check_uppercase(response: str, kwargs: dict) -> bool:
    letters = [c for c in response if c.isalpha()]
    return len(letters) > 0 and response == response.upper()


def _word_count(response: str) -> int:
    return len(response.split())


def check_number_words(response: str, kwargs: dict) -> bool:
    n = _word_count(response)
    relation, target = kwargs["relation"], kwargs["num_words"]
    if relation == "at least":
        return n >= target
    if relation == "at most":
        return n <= target
    if relation == "around":
        return abs(n - target) <= max(1, round(target * 0.1))
    return False


def check_end_phrase(response: str, kwargs: dict) -> bool:
    return response.strip().endswith(kwargs["end_phrase"].strip())


def check_forbidden_words(response: str, kwargs: dict) -> bool:
    resp_lower = response.lower()
    return not any(re.search(r"\b" + re.escape(w.lower()) + r"\b", resp_lower)
                   for w in kwargs["forbidden_words"])


def check_quotation(response: str, kwargs: dict) -> bool:
    r = response.strip()
    return len(r) >= 2 and r.startswith('"') and r.endswith('"')


def check_title(response: str, kwargs: dict) -> bool:
    return bool(re.search(r"<<[^>]+>>", response))


def check_keyword_frequency(response: str, kwargs: dict) -> bool:
    keyword, relation, target = kwargs["keyword"], kwargs["relation"], kwargs["frequency"]
    count = len(re.findall(r"\b" + re.escape(keyword.lower()) + r"\b", response.lower()))
    if relation == "at least":
        return count >= target
    if relation == "at most":
        return count <= target
    return count == target


def check_keyword_existence(response: str, kwargs: dict) -> bool:
    resp_lower = response.lower()
    return all(re.search(r"\b" + re.escape(k.lower()) + r"\b", resp_lower)
               for k in kwargs["keywords"])


CHECKERS: Dict[str, Callable[[str, dict], bool]] = {
    "punctuation:no_comma": check_no_comma,
    "change_case:english_lowercase": check_lowercase,
    "change_case:english_capital": check_uppercase,
    "length_constraints:number_words": check_number_words,
    "startend:end_checker": check_end_phrase,
    "startend:quotation": check_quotation,
    "detectable_format:title": check_title,
    "keywords:forbidden_words": check_forbidden_words,
    "keywords:frequency": check_keyword_frequency,
    "keywords:existence": check_keyword_existence,
}


LONG_OUTPUT_TYPES = {"length_constraints:number_words"}
