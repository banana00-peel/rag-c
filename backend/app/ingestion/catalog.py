import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelMatch:
    canonical: str
    aliases: tuple[str, ...]

# 型号目录
MODEL_CATALOG: tuple[ModelMatch, ...] = (
    ModelMatch("G20", ("G20", "石头G20", "石头 G20")),
    ModelMatch(
        "G10S Pro",
        ("G10S Pro", "G10SPro", "G10S", "石头G10SPro", "石头 G10S Pro", "石头G10S"),
    ),
    ModelMatch("S5", ("S5", "石头S5", "石头 S5")),
)


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\r\n", "\n").replace("\r", "\n")
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def detect_model(filename: str, text: str) -> str:
    haystack = f"{filename}\n{text[:12000]}".upper()
    matches: list[tuple[int, str]] = []
    for item in MODEL_CATALOG:
        for alias in item.aliases:
            pattern = re.escape(alias.upper()).replace(r"\ ", r"\s*")
            match = re.search(rf"(?<![A-Z0-9]){pattern}(?![A-Z0-9])", haystack)
            if match:
                matches.append((match.start(), item.canonical))
                break
    if not matches:
        return ""
    matches.sort(key=lambda item: item[0])
    return matches[0][1]


def detect_brand(filename: str, text: str) -> str:
    haystack = f"{filename}\n{text[:12000]}".lower()
    if "roborock" in haystack or "石头" in haystack:
        return "石头"
    return "石头"


def normalize_model(value: str) -> str:
    candidate = value.strip()
    if not candidate:
        return ""
    for item in MODEL_CATALOG:
        aliases = {alias.upper().replace(" ", "") for alias in item.aliases}
        if candidate.upper().replace(" ", "") in aliases:
            return item.canonical
    return candidate

