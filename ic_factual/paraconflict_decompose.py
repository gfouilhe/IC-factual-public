"""Parse and assemble ParaConflict conflict / filler / continuation prompts."""

from __future__ import annotations


def answers_list(row: dict) -> list[str]:
    ans = row["Answer"]
    if isinstance(ans, str):
        return [ans]
    return list(ans)


def canonical_category(category: str) -> str:
    if category == "Athelete Sport":
        return "Athlete Sport"
    return category


def decompose_row(row: dict) -> tuple[str, str, str]:
    """Return (conflict_clause, coherent_passage, continuation)."""
    clean = row["Clean Prompt"].strip()
    sub = row["Substitution Conflict"].strip()
    coh = row["Coherent Conflict"].strip()

    if sub.endswith(clean):
        conflict_clause = sub[: -len(clean)].strip()
    elif clean in sub:
        conflict_clause = sub[: sub.rfind(clean)].strip()
    else:
        distractor = row["Distracted Token"]
        if distractor in sub:
            end = sub.find(distractor) + len(distractor)
            conflict_clause = sub[:end].strip()
            if not conflict_clause.endswith("."):
                conflict_clause = conflict_clause.rstrip() + "."
        else:
            conflict_clause = sub

    if coh.startswith(conflict_clause):
        passage = coh[len(conflict_clause) :].strip().lstrip(". ")
    elif conflict_clause and conflict_clause in coh:
        passage = coh.replace(conflict_clause, "", 1).strip().lstrip(". ")
    else:
        passage = coh

    return conflict_clause, passage, clean


def assemble_prompt(conflict_clause: str, filler: str, continuation: str) -> str:
    parts = [conflict_clause.strip()]
    if filler:
        parts.append(filler.strip())
    parts.append(continuation.strip())
    return " ".join(p for p in parts if p)
