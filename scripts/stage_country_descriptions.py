"""Stage Wikipedia-style descriptions for ParaConflict countries to JSON.

This is the staging step for ``--filler-kind`` in
``{wiki_country, wiki_distractor_country, wiki_unrelated_country}``
of ``scripts/build_capitals_context_sweep.py``.

Workflow
--------
* Run on a node with internet (login node on the cluster, or any laptop)
  -- compute nodes do not have network access.
* Loads the World-Capital subset of ParaConflict (so the country list
  matches ``scripts/build_capitals_crossproduct.py``) to enumerate the
  ~218 (country, true-capital) pairs.
* For each country, fetches a paragraph-length description from the
  Wikipedia REST summary endpoint
  (``en.wikipedia.org/api/rest_v1/page/summary/{title}``).  Aliases and
  light title fixups are tried in turn.  Manual overrides for a small
  set of edge-case names (see ``MANUAL_TITLE_OVERRIDES``).
* Writes a JSON map ``{country: {description, capital, source_title,
  n_chars}}`` plus a top-level ``_meta`` block.  This is the file
  consumed by ``--country-descriptions`` in the build script.

Compute nodes can then read the staged JSON offline.

Dependencies
------------
Standard library only (uses ``urllib.request``).  ``ic_factual.loaders``
is used only to enumerate countries; if that import fails for some
reason we also accept ``--country-list`` / ``--input`` as a manual
fallback.

Usage
-----
::

    python scripts/stage_country_descriptions.py \
        --output assets/country_descriptions.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path


WIKI_REST_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
USER_AGENT = (
    "ic-factual-research/0.1 (https://example.invalid; offline-research)"
)
DEFAULT_TIMEOUT = 15.0
DEFAULT_MIN_CHARS = 150
DEFAULT_TARGET_CHARS = 2400  # ~600 tokens; tile_to_n in build covers L>that
DEFAULT_RETRIES = 3
DEFAULT_SLEEP_SEC = 0.0  # Wikipedia is ok with modest sustained traffic


# ParaConflict uses a few country names that don't match the canonical
# Wikipedia page title (most are fine; only a handful need overrides).
MANUAL_TITLE_OVERRIDES: dict[str, list[str]] = {
    "the United States": ["United States"],
    "the United States of America": ["United States"],
    "the United Mexican States": ["Mexico"],
    "the United Kingdom": ["United Kingdom"],
    "the United Arab Emirates": ["United Arab Emirates"],
    "the United States Virgin Islands": ["United States Virgin Islands"],
    "the Republic of Korea": ["South Korea"],
    "South Korea": ["South Korea"],
    "the Democratic People's Republic of Korea": ["North Korea"],
    "North Korea": ["North Korea"],
    "the Republic of Moldova": ["Moldova"],
    "the Russian Federation": ["Russia"],
    "the Czech Republic": ["Czech Republic"],
    "the Republic of Singapore": ["Singapore"],
    "the Country of Cura\u00e7ao": ["Cura\u00e7ao"],
    "Holy See (Vatican City State)": ["Vatican City"],
    "the Holy See": ["Vatican City"],
    "Lao People's Democratic Republic": ["Laos"],
    "Iran (Islamic Republic of)": ["Iran"],
    "Republic of the Union of Myanmar": ["Myanmar"],
    "Federated States of Micronesia": ["Federated States of Micronesia"],
    "the Federated States of Micronesia": ["Federated States of Micronesia"],
    "Timor-Leste": ["East Timor"],
    "Bolivia (Plurinational State of)": ["Bolivia"],
    "Venezuela (Bolivarian Republic of)": ["Venezuela"],
    "Brunei Darussalam": ["Brunei"],
    "Syrian Arab Republic": ["Syria"],
    "Viet Nam": ["Vietnam"],
    "C\u00f4te d'Ivoire": ["Ivory Coast", "C\u00f4te d'Ivoire"],
    "United Republic of Tanzania": ["Tanzania"],
    "Republic of the Congo": ["Republic of the Congo"],
    "Democratic Republic of the Congo": ["Democratic Republic of the Congo"],
    "the Republic of the Congo": ["Republic of the Congo"],
    "the Democratic Republic of the Congo": [
        "Democratic Republic of the Congo"
    ],
    "Eswatini": ["Eswatini"],
    "Cabo Verde": ["Cape Verde"],
    "Cape Verde": ["Cape Verde"],
    "Georgia": ["Georgia (country)"],
    "Jordan": ["Jordan", "Jordan (country)"],
    "Mexico": ["Mexico", "Mexico (country)"],
}


def _candidate_titles(country: str) -> list[str]:
    cands = list(MANUAL_TITLE_OVERRIDES.get(country, []))
    cands.append(country)
    # Strip leading "the ".
    no_the = country
    if country.lower().startswith("the "):
        no_the = country[4:]
        cands.append(no_the)
    # Strip trailing "(X)" annotations.
    stripped = re.sub(r"\s*\(.*?\)\s*$", "", country).strip()
    if stripped and stripped != country:
        cands.append(stripped)
    # Strip common formal prefixes: "Republic of X", "Country of X", etc.
    # Apply to both the original and the no-"the" version.
    prefix_patterns = [
        r"^Republic of (the )?",
        r"^Country of ",
        r"^Kingdom of ",
        r"^State of ",
        r"^Commonwealth of (the )?",
        r"^Federated States of ",
        r"^Federation of ",
        r"^Principality of ",
        r"^Sultanate of ",
    ]
    for src in (country, no_the):
        for pat in prefix_patterns:
            simplified = re.sub(pat, "", src, flags=re.IGNORECASE).strip()
            if simplified and simplified != src:
                cands.append(simplified)
    # Final fallback for ambiguous short names (e.g. "Georgia" -> a
    # disambiguation page): try "X (country)".
    for base in list(cands):
        if "(" not in base:
            cands.append(f"{base} (country)")
    # Unique, preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for c in cands:
        c = c.strip()
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _fetch_summary(
    title: str, timeout: float, retries: int
) -> tuple[str, str] | None:
    """Return (description, source_title) on success, None on hard failure.

    Hard failure = 4xx that is not transient.  Transient errors are retried.
    """
    url = WIKI_REST_URL.format(title=urllib.parse.quote(title, safe=""))
    last_err: Exception | None = None
    for attempt in range(max(1, retries)):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            last_err = e
            time.sleep(min(2.0 * (attempt + 1), 5.0))
            continue
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_err = e
            time.sleep(min(2.0 * (attempt + 1), 5.0))
            continue

        # Disambiguation pages do not have a useful extract.
        page_type = payload.get("type", "standard")
        if page_type == "disambiguation":
            return None
        extract = payload.get("extract") or ""
        source_title = payload.get("title") or title
        return extract, source_title

    if last_err is not None:
        print(f"    fetch error after {retries} retries: {last_err}", file=sys.stderr)
    return None


def _enumerate_countries() -> list[tuple[str, str]]:
    """Return [(country, true_capital), ...] from ParaConflict World Capital."""
    try:
        from ic_factual.loaders import load_paraconflict
    except Exception as e:  # pragma: no cover - environment-specific
        raise SystemExit(
            "Failed to import ic_factual.loaders.  Run inside the project "
            "container, or pass --country-list manually.  "
            f"Underlying error: {e}"
        )
    ds = load_paraconflict(split="test")
    ds = ds.filter(lambda r: r["Category"] == "World Capital")
    out: dict[str, str] = {}
    for row in ds:
        country = row["Subject"]
        ans = row["Answer"]
        if isinstance(ans, list):
            ans = ans[0] if ans else ""
        if country and country not in out and ans:
            out[country] = ans
    return sorted(out.items())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch Wikipedia summaries for the ParaConflict 'World "
            "Capital' country list and write a JSON map consumed by "
            "scripts/build_capitals_context_sweep.py "
            "(--country-descriptions)."
        )
    )
    parser.add_argument(
        "--output",
        type=str,
        default="assets/country_descriptions.json",
        help="Where to write the JSON map.",
    )
    parser.add_argument(
        "--country-list",
        type=str,
        default=None,
        help=(
            "Optional newline-separated file of '<country>\\t<capital>' rows "
            "to use instead of loading ParaConflict via ic_factual."
        ),
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=DEFAULT_MIN_CHARS,
        help=(
            "Minimum description length in chars; below this the entry is "
            "marked as missing rather than written."
        ),
    )
    parser.add_argument(
        "--target-chars",
        type=int,
        default=DEFAULT_TARGET_CHARS,
        help=(
            "Soft target description length in chars.  We do not currently "
            "fetch additional sections, but this is recorded in _meta."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help="Retries on transient HTTP / network errors.",
    )
    parser.add_argument(
        "--sleep-sec",
        type=float,
        default=DEFAULT_SLEEP_SEC,
        help="Per-request sleep to be polite to the Wikipedia API.",
    )
    parser.add_argument(
        "--max-countries",
        type=int,
        default=None,
        help="Cap on number of countries (for smoke tests).",
    )
    args = parser.parse_args()

    if args.country_list:
        cl_path = Path(args.country_list)
        pairs: list[tuple[str, str]] = []
        for line in cl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            country, _, capital = line.partition("\t")
            pairs.append((country.strip(), capital.strip()))
    else:
        pairs = _enumerate_countries()

    if args.max_countries is not None:
        pairs = pairs[: args.max_countries]
    print(f"Enumerated {len(pairs)} (country, capital) pairs")

    out_blob: dict[str, dict] = {}
    missing: list[str] = []
    n_short: int = 0
    for i, (country, capital) in enumerate(pairs):
        cands = _candidate_titles(country)
        description: str | None = None
        source_title: str | None = None
        for cand in cands:
            res = _fetch_summary(cand, args.timeout, args.retries)
            if args.sleep_sec > 0:
                time.sleep(args.sleep_sec)
            if res is None:
                continue
            extract, src_t = res
            if not extract or len(extract) < args.min_chars:
                continue
            description, source_title = extract, src_t
            break

        if description is None:
            missing.append(country)
            print(f"  [{i+1:3d}/{len(pairs)}] MISS  {country!r}")
            continue
        if len(description) < args.min_chars:
            n_short += 1
        out_blob[country] = {
            "description": description,
            "capital": capital,
            "source_title": source_title,
            "n_chars": len(description),
        }
        print(
            f"  [{i+1:3d}/{len(pairs)}] OK    {country!r:<40} "
            f"<- {source_title!r}  ({len(description)} chars)"
        )

    out_blob["_meta"] = {
        "_meta": True,
        "source": "Wikipedia REST summary",
        "url_template": WIKI_REST_URL,
        "n_countries_requested": len(pairs),
        "n_countries_ok": len(out_blob) - 1,
        "n_missing": len(missing),
        "missing_countries": missing,
        "n_short_below_min_chars": n_short,
        "min_chars": args.min_chars,
        "target_chars": args.target_chars,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_blob, indent=2, ensure_ascii=False))
    print(
        f"\nWrote {out_path} ({len(out_blob) - 1} country entries, "
        f"{len(missing)} missing)."
    )
    if missing:
        print("Missing:", missing)


if __name__ == "__main__":
    main()
