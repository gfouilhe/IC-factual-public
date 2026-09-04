"""Build context-length / position sweep variants of the capitals cross-product.

This is an extension of ``scripts/build_capitals_crossproduct.py`` that
inserts a configurable amount of filler text between the in-context
"conflict" statement and the question, optionally moves the conflict to
the front of the prompt, or repeats the conflict K times.

Each output JSONL line is a single (country, distractor) prompt in a
schema compatible with ``scripts/evaluate_capitals.py``.  Per-condition
metadata (filler length, kind, conflict position, copies) is stored in
the file's ``_meta`` header line.

Filler kinds
------------
* ``prose`` -- a fixed, deterministic English paragraph about plant cell
  biology, tiled to ``--length-tokens`` tokens.  Contains no country
  names, capital city names, or the word "capital".  Identical filler
  string is used for every prompt in the condition.

* ``capital_noise`` -- a per-prompt sequence of statements
  ``"The capital of X is Y. "`` sampled from the set of (country,
  true_capital) pairs in the input, EXCLUDING the queried country, the
  queried distractor capital, the distractor's country, and the queried
  country's true capital.  **All facts in this filler are TRUE.**

* ``capital_noise_false`` -- same as ``capital_noise`` but the (X, Y)
  pairings are deterministically *shuffled* so each statement asserts
  a FALSE pairing (e.g. ``"The capital of Japan is Lima."``).  The set
  of countries and capitals is unchanged, only the matching is wrong.
  Tests whether induction-head copying is content-agnostic (predicts
  ``capital_noise`` and ``capital_noise_false`` behave identically).

* ``varied_prose`` -- a deterministic sequence of short paragraphs
  drawn from a pool of ~10 unrelated topics (plant cells, weather,
  music theory, algebra, geology, baking, knitting, atomic physics,
  ocean tides, neural networks).  None of the paragraphs mention
  countries, cities or the word ``capital``.  The order is shuffled
  *per-prompt* using ``shuffle_seed`` and the prompt index, then the
  concatenation is tiled / truncated to ``--length-tokens`` tokens.
  Different prompts in the same condition therefore see *different*
  topic sequences, which controls for any single-topic effect of the
  ``prose`` filler.

* ``wiki_country`` -- a Wikipedia-style passage describing the
  *queried* country (loaded from a JSON map provided via
  ``--country-descriptions``).  The country's true capital is replaced
  by ``____`` so the passage does not trivially leak the memorized
  answer.  Tests whether semantically related context (which re-cues
  the queried country's facts) shifts behaviour toward memorized.

* ``wiki_distractor_country`` -- same idea but the description is of
  the *distractor's* country (the country whose true capital is the
  in-context distractor).  The distractor's true capital is replaced
  by ``____``.  Tests whether reinforcing the distractor's country
  context shifts behaviour toward in-context.

* ``wiki_unrelated_country`` -- description of a third country chosen
  deterministically from the pool but excluded from the queried and
  distractor countries.  Acts as a control for the wiki *style*: if
  the wiki_country / wiki_distractor_country effects are about the
  semantic relation rather than about register, this one should
  behave like ``prose``.

* ``wiki_scrambled`` -- same source passage as ``wiki_unrelated_country``,
  but every capitalized word (proper noun) and every 4-digit year is
  *lexically shuffled* across the passage.  Sentence structure,
  punctuation, function-word density, and overall encyclopedic cadence
  are preserved; the *factual content* is destroyed (countries, people,
  cities, dates are randomly reassigned).  Tests whether the wiki
  effect is driven by **register alone** (predicts: same P(mem) as
  wiki_unrelated_country) or by **competing factual assertions**
  (predicts: collapses back to the prose baseline).

* ``prose_facts`` -- ``varied_prose`` interleaved with true
  non-capital country factual statements (``"X borders Y."``,
  ``"X gained independence in N."``, populations, languages, geography,
  ...) at controlled density (~one fact every ``--facts-every-tokens``
  tokens of prose; default 32, which yields roughly 1/4 of the prompt
  being factual assertions for typical fact lengths).  Tests whether
  injecting factual assertions into a narrative-prose carrier
  re-engages memorized recall, even without encyclopedic register.

* ``wiki_shuffled`` -- ``wiki_unrelated_country`` description with
  its sentences shuffled into a random order before tiling.  Preserves
  all factual content and register; destroys discourse coherence.
  Predicts: tracks ``wiki_unrelated_country`` if the effect is
  per-sentence; drops toward prose if discourse cohesion matters.

Conflict and question templates
-------------------------------
Default conflict template (Yu, Merullo & Pavlick 2023 Section 3):

    The capital of {country} is {distractor}.

Conflict-template presets via ``--conflict-template``:

* ``standard``  -- the default above.
* ``negated``   -- ``"The capital of {country} is not {distractor}."``
                   Tests whether the model parses the negation or just
                   copies the most recent ``is X`` pattern.

Question-template presets via ``--question-template``:

* ``qa``         -- ``Q: What is the capital of {country}? A:`` (paper).
* ``bare``       -- ``The capital of {country} is`` (no Q/A template).
* ``answer_is``  -- ``The answer is``.
* ``possessive`` -- ``{country}'s capital is``.
* ``learned``    -- ``I learned that the capital of {country} is``.
* ``of_course``  -- ``The capital city of {country}, of course, is``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from ic_factual.config import PYTHIA_SIZES, resolve_model_size
from ic_factual.loaders import load_tokenizer


# ---------------------------------------------------------------------------
# Templates and the fixed prose filler.
# ---------------------------------------------------------------------------

CONFLICT_TEMPLATE = "The capital of {country} is {distractor}."
QUESTION_TEMPLATE = "Q: What is the capital of {country}? A:"

# Hand-picked synthetic 1-2 token city-like names that do NOT correspond
# to any real-world settlement.  Used by --distractor-kind synthetic.
# Replace the original distractor with one of these deterministically
# (hashed from country + original distractor).
SYNTHETIC_DISTRACTORS = [
    "Zorblax", "Quenmid", "Vexilon", "Drillpax", "Skythorn",
    "Vanloth", "Norvex", "Pelthar", "Sabrith", "Tarlin",
    "Velmir", "Wynmoth", "Yarnox", "Bremidor", "Calthar",
    "Drimion", "Elnara", "Falbrin", "Gartoth", "Hadrumn",
    "Iselon", "Jorbar", "Kralenn", "Lormith", "Mendrok",
    "Nivarr", "Othrund", "Phalax", "Quirix", "Rolmith",
]

CONFLICT_TEMPLATE_PRESETS: dict[str, str] = {
    "standard": CONFLICT_TEMPLATE,
    "negated": "The capital of {country} is not {distractor}.",
}

QUESTION_TEMPLATE_PRESETS: dict[str, str] = {
    "qa": QUESTION_TEMPLATE,
    "bare": "The capital of {country} is",
    "answer_is": "The answer is",
    "possessive": "{country}'s capital is",
    "learned": "I learned that the capital of {country} is",
    "of_course": "The capital city of {country}, of course, is",
}

# A deterministic English paragraph containing no country names, no city
# names, and not the word "capital".  Verified by inspection.  We tile this
# paragraph to reach the requested filler-token count.
PROSE_SNIPPET = (
    "Plant cells contain a rigid outer cell wall composed primarily of "
    "cellulose, which provides structural support and resists turgor "
    "pressure. Within the cell, organelles such as chloroplasts capture "
    "light energy and convert it into chemical energy through "
    "photosynthesis. The mitochondria perform aerobic respiration, "
    "producing adenosine triphosphate that powers cellular processes. "
    "Vacuoles store water, ions, and metabolic byproducts, while "
    "ribosomes synthesize proteins from messenger RNA transcripts. The "
    "nucleus houses the genetic material and coordinates gene expression "
    "in response to environmental cues. Endoplasmic reticulum and Golgi "
    "bodies process and traffic newly synthesized proteins to their "
    "destinations. Cytoskeletal filaments give shape to the cell and "
    "enable intracellular transport. During mitosis, chromosomes "
    "condense and segregate into two daughter nuclei. Stomatal pores in "
    "the leaf epidermis regulate gas exchange and transpiration. "
)

# Pool of topic-disjoint paragraphs used by --filler-kind varied_prose.
# Each paragraph is ~200 words / ~230-280 GPT-NeoX tokens so that even at
# N=1024 a per-prompt sample concatenates only 4-5 distinct paragraphs
# (no within-sample repetition).  The full pool is >2000 tokens.  None of
# the paragraphs mention countries, cities, or the word "capital";
# verified by inspection.  Order is shuffled deterministically per prompt
# from (shuffle_seed, prompt_index) so each prompt sees a different
# sequence -- this is the key control vs. the canonical PROSE_SNIPPET,
# where every prompt sees the same single tiled paragraph.
VARIED_PROSE_PARAGRAPHS = [
    # 0. plant cells
    "Plant cells contain a rigid outer wall composed primarily of "
    "cellulose microfibrils embedded in a matrix of pectin and "
    "hemicellulose, which provides structural support and resists "
    "turgor pressure. Within the cell, organelles such as chloroplasts "
    "capture light energy with chlorophyll molecules and convert it "
    "into chemical energy through photosynthesis, fixing carbon "
    "dioxide into carbohydrate. The mitochondria perform aerobic "
    "respiration, oxidizing glucose to produce adenosine triphosphate "
    "that powers cellular processes. Vacuoles store water, ions, and "
    "metabolic byproducts, while ribosomes synthesize proteins from "
    "messenger RNA transcripts. The nucleus houses the genetic "
    "material organized into chromatin and coordinates gene expression "
    "in response to environmental cues. Endoplasmic reticulum and "
    "Golgi bodies process and traffic newly synthesized proteins to "
    "their destinations. Cytoskeletal filaments give shape to the cell "
    "and enable intracellular transport of vesicles. During mitosis, "
    "chromosomes condense and segregate into two daughter nuclei. "
    "Stomatal pores in the leaf epidermis regulate gas exchange and "
    "transpiration in response to humidity and circadian rhythms.",
    # 1. weather
    "Atmospheric pressure varies with altitude and temperature, "
    "driving the global circulation of air masses through cells of "
    "rising and sinking motion. Cumulus clouds form when warm, moist "
    "air rises and cools to its dew point, releasing latent heat that "
    "further accelerates the updraft. Cold fronts produce sharp "
    "temperature drops and brief, intense precipitation as a wedge of "
    "dense cold air slides beneath warmer air ahead of it. Stable "
    "atmospheric layers yield clear skies and calm conditions, while "
    "unstable layers can spawn thunderstorms with hail, heavy rain, "
    "and occasional tornadoes. Jet streams meander at the boundary "
    "between cold polar air and warmer subtropical air, steering "
    "weather systems from west to east at high altitude. Water cycles "
    "continuously between the ocean, atmosphere, and land through "
    "evaporation, condensation, and precipitation. The Coriolis effect "
    "deflects winds and ocean currents, producing the characteristic "
    "counter-rotating gyres that organize storms into spirals. "
    "Forecasting models integrate observations from satellites, "
    "radiosondes, and surface stations to estimate future states of "
    "the atmosphere.",
    # 2. music theory
    "Western tonal music is built on a chromatic scale of twelve "
    "equally tempered pitches that recur cyclically across registers. "
    "Major and minor scales differ in the position of their semitone "
    "steps, producing distinct emotional colorings. A triad consists "
    "of a root, a third, and a fifth stacked above each scale degree, "
    "and the quality of the third determines whether the chord sounds "
    "major or minor. Chord progressions create a sense of tension and "
    "release through cadences that resolve onto tonic harmony, with "
    "secondary dominants and modulation enriching the tonal palette. "
    "Counterpoint weaves multiple melodic lines simultaneously "
    "according to rules that govern consonance, dissonance, and voice "
    "leading. Rhythm organizes time into recurring patterns of strong "
    "and weak pulses, often grouped into meters such as duple, triple, "
    "or compound. Dynamics, articulation, and tempo modify the "
    "expressive surface of a piece. Form provides the architectural "
    "plan, ranging from binary and ternary structures to sonata, "
    "rondo, and theme-and-variations frameworks. Texture distinguishes "
    "monophony, homophony, and polyphony as principal organizing "
    "strategies.",
    # 3. algebra
    "A polynomial expression sums terms in which variables are raised "
    "to non-negative integer powers, with each term carrying a "
    "coefficient drawn from some underlying field. The degree of the "
    "polynomial equals the largest such exponent and bounds both the "
    "maximum number of real roots and the number of turning points of "
    "the corresponding curve. Quadratics of degree two factor into "
    "linear binomials whose roots can be recovered using the "
    "discriminant, which determines whether those roots are real, "
    "repeated, or complex. Higher-order polynomials may have complex "
    "roots that always occur in conjugate pairs whenever the "
    "coefficients are real. The fundamental theorem of algebra "
    "guarantees that every nonconstant polynomial of degree n over the "
    "complex numbers admits exactly n roots counted with multiplicity. "
    "Synthetic division and long division provide algorithmic tools "
    "for factoring once a single root has been found. Ring and field "
    "extensions generalize this picture, allowing one to study "
    "polynomials whose coefficients lie in number systems beyond the "
    "ordinary integers and rationals. Group theory then describes the "
    "symmetries among the roots.",
    # 4. geology
    "Igneous rocks crystallize from molten magma as it cools either "
    "deep beneath the surface, producing coarse-grained intrusive "
    "rocks such as granite and gabbro, or after a volcanic eruption, "
    "producing fine-grained extrusive rocks such as basalt and "
    "rhyolite. Sedimentary rocks form by the gradual deposition and "
    "lithification of mineral and organic particles, often preserving "
    "fossils that record the history of life and ancient environments. "
    "Metamorphic rocks arise when heat and pressure restructure "
    "existing rock without melting it, generating foliated textures "
    "such as those of schist and gneiss. Plate tectonics drives the "
    "slow recycling of crustal material over millions of years, with "
    "mid-ocean ridges generating new oceanic crust while subduction "
    "zones consume old crust. The rock cycle links the three principal "
    "rock types into a continuous loop of formation, weathering, "
    "deposition, and transformation. Mineralogists classify minerals "
    "by chemical composition and crystal structure, using properties "
    "such as cleavage, hardness, and luster as diagnostic tools. "
    "Stratigraphic correlation across sedimentary basins allows "
    "geologists to reconstruct past climates, sea levels, and tectonic "
    "events.",
    # 5. bread baking
    "Yeast metabolizes simple sugars in dough, producing carbon "
    "dioxide that inflates a network of gluten strands developed by "
    "the kneading or folding of wheat flour and water. Long "
    "fermentation develops complex flavors as enzymes break down "
    "starches into shorter sugars and proteins into peptides and amino "
    "acids that contribute to color and aroma during baking. A "
    "pre-fermented sponge or sourdough starter can be prepared the day "
    "before to improve flavor and extensibility, harnessing wild yeast "
    "and lactic acid bacteria. Folding the dough at intervals "
    "strengthens its structure, redistributes the yeast and trapped "
    "gas, and improves the final crumb texture. Shaping creates "
    "surface tension that helps the loaf hold its form during the "
    "final proof. A hot oven sets the crust quickly while steam keeps "
    "the interior soft enough to expand fully in a short burst of oven "
    "spring. Hydration ratio, salt percentage, and fermentation time "
    "interact to determine the final crumb and crust characteristics. "
    "Temperature control during bulk fermentation governs the balance "
    "of organic acids produced by sourdough cultures, with cooler "
    "temperatures favoring lactic acid and warmer temperatures "
    "favoring acetic acid.",
    # 6. knitting
    "A knitted fabric is constructed from interlocking loops drawn "
    "through previous loops, row after row, on a pair of pointed "
    "needles or on a circular needle. Knit stitches form vertical V "
    "shapes on one face of the fabric, while purl stitches produce "
    "horizontal bumps on the same face, and combining the two yields "
    "textures such as ribbing, garter, seed stitch, and complex "
    "cables. Stockinette curls at its edges because of asymmetry in "
    "the loop tension, which is why edges are often bordered with "
    "garter or ribbed sections. Tension control determines the gauge, "
    "or the number of stitches and rows per unit length, which in turn "
    "fixes the final dimensions of the finished piece. Increases and "
    "decreases shape the fabric by adding or removing stitches at "
    "controlled points, allowing for raglan sleeves, gussets, and "
    "curved hems. Color work techniques such as stranded knitting and "
    "intarsia introduce multiple yarns into a single row, while lace "
    "patterns combine yarn-overs with paired decreases to produce open "
    "structures. Blocking with water or steam relaxes the fibers and "
    "sets the final shape, evening out small irregularities and "
    "letting the stitches bloom.",
    # 7. atomic physics
    "An atom consists of a dense positively charged nucleus of protons "
    "and neutrons surrounded by electrons confined to discrete energy "
    "levels described by quantum numbers. Electrons absorb or emit "
    "photons of light when they jump between these levels, producing "
    "the line spectra characteristic of each element. The arrangement "
    "of electrons in the outermost shell determines an element's "
    "chemical reactivity and its position in the periodic table, with "
    "noble gases occupying complete shells and alkali metals carrying "
    "a single valence electron. Quantum mechanics predicts the "
    "probability of finding an electron in a particular orbital around "
    "the nucleus, with the shapes of those orbitals determined by the "
    "angular momentum quantum number. The Pauli exclusion principle "
    "forbids two electrons from sharing the same complete set of "
    "quantum numbers, structuring the buildup of multielectron atoms. "
    "Nuclear stability depends on the balance between the strong force "
    "binding nucleons and the electromagnetic repulsion between "
    "protons, with unstable isotopes decaying through alpha, beta, or "
    "gamma emission. Spin-orbit coupling and relativistic corrections "
    "produce fine and hyperfine splittings observable in "
    "high-resolution spectroscopy.",
    # 8. ocean tides
    "Tides are produced by the gravitational pull of the moon and the "
    "sun acting on large bodies of water across the rotating planet. "
    "Two tidal bulges form on opposite sides of the planet because the "
    "gravitational gradient stretches the ocean both toward and away "
    "from the attracting body, and these bulges rotate as the earth "
    "spins, producing roughly two high tides per day in most coastal "
    "regions. The amplitude of the tide depends on coastline geometry, "
    "water depth, the alignment of the sun and moon, and resonance "
    "properties of the basin in which the tide propagates. Spring "
    "tides occur near new and full moon phases, when the lunar and "
    "solar pulls add together, while neap tides occur near the quarter "
    "phases, when the two pulls partially cancel. Tidal currents flow "
    "into and out of estuaries, driving substantial mixing and "
    "sediment transport. The energy dissipated by tidal friction "
    "slowly transfers angular momentum from the rotating earth to the "
    "moon, lengthening the day and gradually pushing the moon to a "
    "higher orbit. Long-period tidal constituents arise from "
    "astronomical cycles spanning weeks to decades.",
    # 9. neural networks
    "An artificial neural network is composed of layers of simple "
    "computational units that transform an input vector by applying a "
    "weighted sum followed by a nonlinear activation function such as "
    "the rectified linear unit, hyperbolic tangent, or sigmoid. "
    "Training adjusts the weights to minimize a loss function via "
    "gradient descent, with the gradients computed efficiently by "
    "backpropagation through the chain rule. Deeper networks can "
    "represent more complex functions but require more data, careful "
    "initialization, and regularization techniques such as dropout, "
    "weight decay, and batch normalization to avoid overfitting. "
    "Modern variants include convolutional networks for grid-"
    "structured signals, recurrent networks for sequences, and "
    "self-attention architectures that compare every position in a "
    "sequence with every other position. Optimization algorithms such "
    "as stochastic gradient descent, momentum, and adaptive methods "
    "navigate the highly nonconvex loss landscape during training. "
    "Generalization beyond the training distribution remains a central "
    "theoretical and practical challenge. Embedding layers map "
    "discrete tokens into continuous vector spaces where geometric "
    "relations capture semantic similarity. Modular layer composition "
    "lets engineers build very large models from a small set of "
    "reusable primitives.",
]

NOISE_TEMPLATE = "The capital of {country} is {city}. "

WIKI_REDACTION_PLACEHOLDER = "____"


# Hand-curated pool of true non-capital country fact statements used by
# --filler-kind prose_facts.  All assertions are factually correct (as of
# 2024) and *deliberately* avoid the {country, capital} relation -- they
# are about borders, independence years, populations, languages,
# continents, currencies, geography, etc.  We sample these
# deterministically (seed from the prompt index) and interleave them
# with varied_prose paragraphs.  None of the statements mention any of
# the queried/distractor *capitals* (verified by inspection), so the
# filler cannot trivially leak the answer.
NON_CAPITAL_COUNTRY_FACTS: list[str] = [
    # Borders
    "France shares a border with Germany.",
    "Spain shares a border with Portugal.",
    "Brazil shares a border with Argentina.",
    "Egypt shares a border with Libya.",
    "India shares a border with Bangladesh.",
    "Canada shares a border with the United States.",
    "Mexico shares a border with Guatemala.",
    "Vietnam shares a border with Laos.",
    "Ukraine shares a border with Poland.",
    "Sweden shares a border with Norway.",
    "Switzerland shares a border with Italy.",
    "Turkey shares a border with Greece.",
    "Iran shares a border with Iraq.",
    "Colombia shares a border with Ecuador.",
    "Saudi Arabia shares a border with Yemen.",
    # Independence years
    "Brazil gained independence in 1822.",
    "India gained independence in 1947.",
    "Algeria gained independence in 1962.",
    "Indonesia gained independence in 1945.",
    "Mexico gained independence in 1821.",
    "Vietnam gained independence in 1945.",
    "Nigeria gained independence in 1960.",
    "Kenya gained independence in 1963.",
    "Pakistan gained independence in 1947.",
    "Zimbabwe gained independence in 1980.",
    # Populations (approximate, rounded)
    "Japan has a population of about 125 million.",
    "Germany has a population of about 84 million.",
    "Iran has a population of about 88 million.",
    "Ethiopia has a population of about 120 million.",
    "Egypt has a population of about 110 million.",
    "Turkey has a population of about 85 million.",
    "Argentina has a population of about 46 million.",
    "Canada has a population of about 40 million.",
    "Australia has a population of about 26 million.",
    "Bangladesh has a population of about 170 million.",
    # Continents and regions
    "Mongolia is located in central Asia.",
    "Bolivia is located in South America.",
    "Senegal is located in West Africa.",
    "Finland is located in northern Europe.",
    "Cambodia is located in Southeast Asia.",
    "Uruguay is located in South America.",
    "Madagascar is located off the southeast coast of Africa.",
    "Iceland is located in the North Atlantic Ocean.",
    "Tunisia is located in North Africa.",
    "Sri Lanka is located in South Asia.",
    # Languages
    "Brazil's official language is Portuguese.",
    "Switzerland recognizes German, French, Italian, and Romansh as official languages.",
    "Egypt's official language is Arabic.",
    "Mongolia's official language is Mongolian.",
    "Indonesia's official language is Indonesian.",
    "Ethiopia uses Amharic as a working language.",
    "Belgium recognizes Dutch, French, and German as official languages.",
    "Singapore recognizes English, Malay, Mandarin, and Tamil as official languages.",
    # Currencies
    "Japan uses the yen as its currency.",
    "Switzerland uses the Swiss franc as its currency.",
    "Brazil uses the real as its currency.",
    "South Africa uses the rand as its currency.",
    "India uses the rupee as its currency.",
    "Turkey uses the lira as its currency.",
    "Sweden uses the krona as its currency.",
    "Mexico uses the peso as its currency.",
    # Geography and natural features
    "Russia is the world's largest country by area.",
    "Egypt is home to the Nile river.",
    "Argentina contains part of the Andes mountains.",
    "Nepal contains part of the Himalayan mountain range.",
    "Tanzania is home to Mount Kilimanjaro.",
    "Brazil contains most of the Amazon rainforest.",
    "Canada borders three different oceans.",
    "Indonesia is composed of over seventeen thousand islands.",
    "Norway is known for its fjords along the western coast.",
    "Mongolia contains the Gobi desert.",
    "The Sahara covers much of northern Africa.",
    "Greenland is the world's largest island.",
    # Membership and governance
    "Germany is a member of the European Union.",
    "Sweden joined the European Union in 1995.",
    "Japan is a constitutional monarchy.",
    "Norway is not a member of the European Union.",
    "Switzerland is famous for its neutrality in foreign affairs.",
    "Canada is a member of the Commonwealth of Nations.",
    "Brazil is a federal republic.",
    "India is a parliamentary democracy.",
    # Misc culture
    "Japan is known for cherry blossom season in spring.",
    "Italy is widely recognized for its Renaissance art.",
    "Norway awards the Nobel Peace Prize each year.",
    "Brazil hosts the Carnival festival each year.",
    "China invented papermaking and gunpowder.",
    "Ethiopia is one of the oldest independent nations in Africa.",
    "Egypt is famous for its ancient pyramids.",
    "Greece is considered the birthplace of Western philosophy.",
    "Mongolia was once at the heart of the largest contiguous land empire in history.",
    "Russia spans eleven time zones.",
]


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _encode(text: str, tokenizer) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def _decode(ids: list[int], tokenizer) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True)


def _tile_to_n_tokens(text: str, n: int, tokenizer) -> str:
    """Tile ``text`` and truncate to exactly ``n`` tokens (or return '')."""
    if n <= 0:
        return ""
    base_ids = _encode(text, tokenizer)
    if not base_ids:
        return ""
    ids: list[int] = []
    while len(ids) < n:
        ids.extend(base_ids)
    return _decode(ids[:n], tokenizer)


def _split_filler_in_half(filler: str, tokenizer) -> tuple[str, str]:
    if not filler:
        return "", ""
    ids = _encode(filler, tokenizer)
    half = len(ids) // 2
    left = _decode(ids[:half], tokenizer)
    right = _decode(ids[half:], tokenizer)
    return left, right


def _build_capital_noise_filler(
    n_tokens: int,
    tokenizer,
    pool: list[tuple[str, str]],
    exclude_countries: set[str],
    exclude_capitals: set[str],
    seed: int,
    falsify: bool = False,
) -> str:
    """Generate ``n_tokens`` worth of synthetic ``"The capital of X is Y."``
    statements, omitting any pair touching ``exclude_countries`` or
    ``exclude_capitals``.  Order is shuffled deterministically from ``seed``.

    If ``falsify`` is True, the (country, capital) pairings are
    deterministically permuted so that every statement asserts a FALSE
    pairing (using the same country set and capital set as the true pool,
    just mismatched).  Order is still shuffled.
    """
    if n_tokens <= 0:
        return ""
    rng = random.Random(seed)
    candidates = [
        (c, k)
        for c, k in pool
        if c not in exclude_countries and k not in exclude_capitals
    ]
    if not candidates:
        return ""
    if falsify and len(candidates) > 1:
        countries = [c for c, _ in candidates]
        capitals = [k for _, k in candidates]
        capitals_shuffled = list(capitals)
        attempts = 0
        while attempts < 32:
            rng.shuffle(capitals_shuffled)
            if all(a != b for a, b in zip(capitals, capitals_shuffled)):
                break
            attempts += 1
        candidates = list(zip(countries, capitals_shuffled))
    rng.shuffle(candidates)

    out_text_parts: list[str] = []
    out_token_count = 0
    i = 0
    safety_budget = max(8 * n_tokens, 256)
    while out_token_count < n_tokens and safety_budget > 0:
        c, k = candidates[i % len(candidates)]
        stmt = NOISE_TEMPLATE.format(country=c, city=k)
        out_text_parts.append(stmt)
        out_token_count += len(_encode(stmt, tokenizer))
        i += 1
        safety_budget -= 1
        if i % len(candidates) == 0:
            rng.shuffle(candidates)

    text = "".join(out_text_parts)
    ids = _encode(text, tokenizer)[:n_tokens]
    return _decode(ids, tokenizer)


def _build_varied_prose_filler(
    n_tokens: int,
    tokenizer,
    seed: int,
) -> str:
    """Concatenate a deterministic shuffle of ``VARIED_PROSE_PARAGRAPHS``
    and tile / truncate to exactly ``n_tokens`` tokens.

    The order is shuffled per-prompt using ``seed``, so different prompts
    in the same condition see different paragraph sequences (controls
    for any single-topic effect of the canonical ``prose`` filler).
    """
    if n_tokens <= 0:
        return ""
    rng = random.Random(seed)
    paragraphs = list(VARIED_PROSE_PARAGRAPHS)
    rng.shuffle(paragraphs)

    parts: list[str] = []
    out_token_count = 0
    i = 0
    safety_budget = max(8 * n_tokens, 256)
    while out_token_count < n_tokens and safety_budget > 0:
        para = paragraphs[i % len(paragraphs)]
        if parts and not parts[-1].endswith(" "):
            parts.append(" ")
        parts.append(para)
        out_token_count += len(_encode(para, tokenizer))
        i += 1
        safety_budget -= 1
        if i % len(paragraphs) == 0:
            rng.shuffle(paragraphs)

    text = "".join(parts)
    ids = _encode(text, tokenizer)[:n_tokens]
    return _decode(ids, tokenizer)


def _redact_capital(passage: str, capital: str) -> str:
    """Replace whole-word, case-insensitive occurrences of ``capital``
    in ``passage`` with ``WIKI_REDACTION_PLACEHOLDER``.  Multi-word
    capitals (e.g. ``"San Juan"``) are matched as a single unit.
    """
    import re

    if not passage or not capital:
        return passage
    pattern = re.compile(
        r"(?<!\w)" + re.escape(capital) + r"(?!\w)",
        flags=re.IGNORECASE,
    )
    return pattern.sub(WIKI_REDACTION_PLACEHOLDER, passage)


def _build_wiki_filler(
    n_tokens: int,
    tokenizer,
    description: str,
    redact_capital: str,
) -> str:
    """Tile / truncate a wiki description to exactly ``n_tokens`` tokens.

    The provided ``redact_capital`` (true capital of the country whose
    description this is) is removed first to avoid leaking the answer.
    """
    if n_tokens <= 0:
        return ""
    if not description:
        return ""
    redacted = _redact_capital(description, redact_capital)
    base_ids = _encode(redacted, tokenizer)
    if not base_ids:
        return ""
    ids: list[int] = []
    while len(ids) < n_tokens:
        ids.extend(base_ids)
    return _decode(ids[:n_tokens], tokenizer)


def _scramble_proper_nouns_and_dates(text: str, seed: int) -> str:
    """Lexically shuffle capitalized words (proper nouns) and 4-digit years
    within ``text``.  Preserves all function words, punctuation, and
    syntactic structure; destroys the factual *identity* of named
    entities and dates.

    A "capitalized word" is any contiguous run of letters of length >= 2
    that starts with an uppercase letter.  Sentence-initial capitalized
    function words ("The", "A", "It", ...) are excluded by a small
    stoplist so we don't shuffle them around and break sentence starts.
    Single-letter tokens ("A", "I") are likewise left alone.

    4-digit years (1500-2099) are detected separately and shuffled
    within their own pool, so dates remain plausible 4-digit numbers
    but no longer point to true events.
    """
    import re

    if not text:
        return text
    rng = random.Random(seed)

    common_caps = {
        "The", "A", "An", "It", "Its", "This", "That", "These", "Those",
        "He", "She", "They", "We", "I", "You", "Their", "His", "Her",
        "And", "Or", "But", "If", "When", "While", "After", "Before",
        "However", "Although", "Because", "Since", "Until", "Though",
        "In", "On", "At", "By", "For", "From", "Of", "To", "With",
        "Many", "Most", "Some", "All", "Both", "Each", "Several",
        "Today", "Now", "Then",
    }

    cap_pattern = re.compile(r"\b[A-Z][A-Za-z]{1,}\b")
    matches = list(cap_pattern.finditer(text))
    cap_indices: list[int] = []
    cap_values: list[str] = []
    for i, m in enumerate(matches):
        w = m.group(0)
        if w in common_caps:
            continue
        cap_indices.append(i)
        cap_values.append(w)
    if cap_values:
        shuffled_caps = list(cap_values)
        rng.shuffle(shuffled_caps)
        replacement_for_match: dict[int, str] = {
            cap_indices[k]: shuffled_caps[k] for k in range(len(cap_indices))
        }

        out_chunks: list[str] = []
        cursor = 0
        for i, m in enumerate(matches):
            out_chunks.append(text[cursor : m.start()])
            if i in replacement_for_match:
                out_chunks.append(replacement_for_match[i])
            else:
                out_chunks.append(m.group(0))
            cursor = m.end()
        out_chunks.append(text[cursor:])
        text = "".join(out_chunks)

    year_pattern = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2})\b")
    year_matches = list(year_pattern.finditer(text))
    if year_matches:
        years = [m.group(0) for m in year_matches]
        shuffled_years = list(years)
        rng.shuffle(shuffled_years)
        out_chunks = []
        cursor = 0
        for k, m in enumerate(year_matches):
            out_chunks.append(text[cursor : m.start()])
            out_chunks.append(shuffled_years[k])
            cursor = m.end()
        out_chunks.append(text[cursor:])
        text = "".join(out_chunks)

    return text


def _shuffle_sentences(text: str, seed: int) -> str:
    """Split ``text`` on sentence-final punctuation and shuffle the
    resulting sentences deterministically.  Punctuation is preserved
    on the sentences that originally carried it.
    """
    import re

    if not text:
        return text
    rng = random.Random(seed)
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    parts = [p for p in parts if p]
    if len(parts) <= 1:
        return text
    rng.shuffle(parts)
    return " ".join(parts)


def _build_wiki_scrambled_filler(
    n_tokens: int,
    tokenizer,
    description: str,
    redact_capital: str,
    seed: int,
) -> str:
    """Scramble proper nouns / years in a wiki description, then tile
    to ``n_tokens`` tokens.

    Capitalized-word and year shuffling happens *before* redacting the
    capital so the capital placeholder ``____`` is unaffected.  In
    practice the wiki source already had its true capital substring
    redacted to ``____`` upstream, so the scrambler operates on a
    capital-free passage.
    """
    if n_tokens <= 0 or not description:
        return ""
    redacted = _redact_capital(description, redact_capital)
    scrambled = _scramble_proper_nouns_and_dates(redacted, seed=seed)
    base_ids = _encode(scrambled, tokenizer)
    if not base_ids:
        return ""
    ids: list[int] = []
    while len(ids) < n_tokens:
        ids.extend(base_ids)
    return _decode(ids[:n_tokens], tokenizer)


def _build_wiki_shuffled_filler(
    n_tokens: int,
    tokenizer,
    description: str,
    redact_capital: str,
    seed: int,
) -> str:
    """Sentence-shuffle a wiki description, then tile to ``n_tokens`` tokens."""
    if n_tokens <= 0 or not description:
        return ""
    redacted = _redact_capital(description, redact_capital)
    shuffled = _shuffle_sentences(redacted, seed=seed)
    base_ids = _encode(shuffled, tokenizer)
    if not base_ids:
        return ""
    ids: list[int] = []
    while len(ids) < n_tokens:
        ids.extend(base_ids)
    return _decode(ids[:n_tokens], tokenizer)


def _build_prose_facts_filler(
    n_tokens: int,
    tokenizer,
    seed: int,
    facts_every_tokens: int,
    exclude_capitals: set[str],
) -> str:
    """Interleave a deterministic shuffle of ``VARIED_PROSE_PARAGRAPHS``
    with true non-capital country fact statements at controlled density.

    We split the prose into sentences and emit them one at a time;
    every ``facts_every_tokens`` *prose* tokens accumulated we insert
    one randomly-drawn fact sentence between two prose sentences,
    then resume.  This keeps the fact density well-defined even at
    short ``n_tokens`` where a single paragraph would otherwise be
    truncated before any fact had been injected.

    Facts that mention any of ``exclude_capitals`` (case-insensitive
    whole-word match) are dropped from the candidate pool so the
    filler cannot leak the queried answer or the in-context distractor.
    """
    if n_tokens <= 0:
        return ""
    import re

    rng = random.Random(seed)

    paragraphs = list(VARIED_PROSE_PARAGRAPHS)
    rng.shuffle(paragraphs)

    sent_split = re.compile(r"(?<=[.!?])\s+")
    prose_sentences: list[str] = []
    for para in paragraphs:
        prose_sentences.extend(s for s in sent_split.split(para.strip()) if s)

    facts_pool = list(NON_CAPITAL_COUNTRY_FACTS)
    if exclude_capitals:
        cleaned_pool: list[str] = []
        excl_patterns = [
            re.compile(r"(?<!\w)" + re.escape(c) + r"(?!\w)", flags=re.IGNORECASE)
            for c in exclude_capitals
            if c
        ]
        for fact in facts_pool:
            if not any(p.search(fact) for p in excl_patterns):
                cleaned_pool.append(fact)
        facts_pool = cleaned_pool
    if not facts_pool:
        facts_pool = list(NON_CAPITAL_COUNTRY_FACTS)
    rng.shuffle(facts_pool)

    parts: list[str] = []
    prose_token_buf = 0
    out_token_count = 0
    sent_idx = 0
    fact_idx = 0
    safety_budget = max(8 * n_tokens, 256)
    facts_every = max(int(facts_every_tokens), 8)

    def _append(s: str) -> int:
        nonlocal out_token_count
        if parts and not parts[-1].endswith(" "):
            parts.append(" ")
        parts.append(s)
        tok = len(_encode(s, tokenizer))
        out_token_count += tok
        return tok

    while out_token_count < n_tokens and safety_budget > 0:
        sent = prose_sentences[sent_idx % len(prose_sentences)]
        sent_idx += 1
        added = _append(sent)
        prose_token_buf += added
        if prose_token_buf >= facts_every and out_token_count < n_tokens:
            fact = facts_pool[fact_idx % len(facts_pool)]
            fact_idx += 1
            _append(fact)
            prose_token_buf = 0
        safety_budget -= 1

    text = "".join(parts)
    ids = _encode(text, tokenizer)[:n_tokens]
    return _decode(ids, tokenizer)


def assemble_prompt(
    country: str,
    distractor: str,
    filler: str,
    position: str,
    n_copies: int,
    tokenizer,
    question_template: str = QUESTION_TEMPLATE,
    conflict_template: str = CONFLICT_TEMPLATE,
) -> str:
    """Construct the final prompt for one (country, distractor) record."""
    conflict = conflict_template.format(country=country, distractor=distractor)
    conflict_block = " ".join([conflict] * max(int(n_copies), 1))
    question = question_template.format(country=country)

    if not filler:
        return f"{conflict_block} {question}"

    if position == "tail":
        return f"{conflict_block} {filler} {question}"
    if position == "front":
        return f"{filler} {conflict_block} {question}"
    if position == "middle":
        left, right = _split_filler_in_half(filler, tokenizer)
        return f"{left} {conflict_block} {right} {question}"
    raise ValueError(
        f"Unknown --conflict-position: {position!r} "
        f"(expected one of front, middle, tail)"
    )


# ---------------------------------------------------------------------------
# IO.
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> tuple[dict, list[dict]]:
    meta: dict = {}
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("_meta"):
                meta = obj
            else:
                records.append(obj)
    return meta, records


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a context-length sweep variant of the capitals "
            "cross-product (one JSONL = one condition)."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Base cross-product JSONL from build_capitals_crossproduct.py.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Where to write the new JSONL.",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default=None,
        choices=list(PYTHIA_SIZES),
        help=(
            "Pythia size whose tokenizer we use to measure the filler "
            "length.  All Pythia sizes share the same GPT-NeoX tokenizer "
            "so this only matters for offline file resolution.  Ignored "
            "when --tokenizer-path is given."
        ),
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help=(
            "Explicit tokenizer Hub id or local snapshot path (e.g. a "
            "Qwen3 directory).  When set, this tokenizer measures the "
            "filler length instead of the Pythia GPT-NeoX tokenizer.  "
            "Required when building sweeps for a non-Pythia model family "
            "whose tokenization (and therefore L_rel) differs.  Mutually "
            "exclusive with --model-size."
        ),
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=0,
        help=(
            "Target model's context-window size in tokens (e.g. 2048 for "
            "Pythia, 32768 for Qwen3).  Used only to record the relative "
            "input length L_rel = prompt_tokens / context_window in the "
            "output _meta header so downstream analysis can plot the "
            "Veseli et al. (2025) relative-length axis.  0 = unknown."
        ),
    )
    parser.add_argument(
        "--length-tokens",
        type=int,
        default=0,
        help=(
            "Target number of filler tokens between the conflict statement "
            "and the question.  0 reproduces the original paper template."
        ),
    )
    parser.add_argument(
        "--filler-kind",
        type=str,
        default="prose",
        choices=(
            "prose",
            "capital_noise",
            "capital_noise_false",
            "varied_prose",
            "wiki_country",
            "wiki_distractor_country",
            "wiki_unrelated_country",
            "wiki_scrambled",
            "wiki_shuffled",
            "prose_facts",
        ),
        help=(
            "prose: deterministic plant-biology paragraph (same for all "
            "prompts).  capital_noise: per-prompt 'The capital of X is Y.' "
            "statements drawn from the input pool, excluding the queried "
            "country/capital pair (TRUE pairings).  capital_noise_false: "
            "same set of (country, capital) names but deterministically "
            "shuffled so each statement asserts a FALSE pairing.  "
            "varied_prose: per-prompt deterministic shuffle of ~10 "
            "topic-disjoint paragraphs (no country/city names).  "
            "wiki_country / wiki_distractor_country / "
            "wiki_unrelated_country: Wikipedia-style passage about the "
            "queried country / distractor's country / a third unrelated "
            "country (capital of that country redacted).  Requires "
            "--country-descriptions.  wiki_scrambled: same source as "
            "wiki_unrelated_country, but every capitalized word and "
            "every 4-digit year is lexically shuffled (preserves "
            "register, destroys facts).  wiki_shuffled: "
            "wiki_unrelated_country with sentences shuffled (destroys "
            "discourse coherence).  prose_facts: varied_prose with true "
            "non-capital country facts injected every "
            "--facts-every-tokens tokens (preserves narrative register, "
            "adds factual assertions)."
        ),
    )
    parser.add_argument(
        "--facts-every-tokens",
        type=int,
        default=32,
        help=(
            "For --filler-kind prose_facts: insert one non-capital "
            "country fact (X borders Y, X gained independence in N, "
            "...) every N tokens of prose.  Default 32 yields roughly "
            "1 fact per ~32 tokens; lower means denser, higher means "
            "sparser.  Ignored for other filler kinds."
        ),
    )
    parser.add_argument(
        "--country-descriptions",
        type=str,
        default=None,
        help=(
            "Path to JSON map of {country: {description: str}} produced "
            "by scripts/stage_country_descriptions.py.  Required for "
            "--filler-kind in {wiki_country, wiki_distractor_country, "
            "wiki_unrelated_country}.  Records whose required country "
            "key is missing from the map are skipped (with a count "
            "logged at the end)."
        ),
    )
    parser.add_argument(
        "--conflict-template",
        type=str,
        default="standard",
        help=(
            "Conflict-statement template.  Preset name "
            "(standard | negated) or a raw template string containing "
            "{country} and {distractor}."
        ),
    )
    parser.add_argument(
        "--distractor-kind",
        type=str,
        default="real",
        choices=("real", "synthetic"),
        help=(
            "real (default): use the original distractor capital from the "
            "input record.  synthetic: replace the distractor with a "
            "hand-picked non-real city-like name (Zorblax, Quenmid, ...).  "
            "Assignment is deterministic per (country, original_distractor)."
        ),
    )
    parser.add_argument(
        "--conflict-position",
        type=str,
        default="tail",
        choices=("tail", "front", "middle"),
        help=(
            "Where to place the conflict statement relative to the filler. "
            "'tail' (default): conflict|filler|Q.  'front': "
            "filler|conflict|Q.  'middle': filler/2|conflict|filler/2|Q."
        ),
    )
    parser.add_argument(
        "--n-copies",
        type=int,
        default=1,
        help="Number of times to repeat the conflict statement (>=1).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap the number of (country, distractor) prompts (smoke tests).",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=0,
        help=(
            "Seed for shuffling the input before --max-samples truncation, "
            "and for the per-prompt noise sampler."
        ),
    )
    parser.add_argument(
        "--question-template",
        type=str,
        default="qa",
        help=(
            "Question template appended after the conflict statement and "
            "filler.  Accepts a preset name (qa | bare | answer_is) or a "
            "raw template string containing {country}.  Default 'qa' = "
            "'Q: What is the capital of {country}? A:' (paper template).  "
            "'bare' = 'The capital of {country} is' strips the Q/A template."
        ),
    )
    args = parser.parse_args()

    if args.question_template in QUESTION_TEMPLATE_PRESETS:
        question_template = QUESTION_TEMPLATE_PRESETS[args.question_template]
        question_template_name = args.question_template
    else:
        if "{country}" not in args.question_template:
            raise SystemExit(
                "--question-template must be a preset or contain {country}; "
                f"got {args.question_template!r}"
            )
        question_template = args.question_template
        question_template_name = "custom"

    if args.conflict_template in CONFLICT_TEMPLATE_PRESETS:
        conflict_template = CONFLICT_TEMPLATE_PRESETS[args.conflict_template]
        conflict_template_name = args.conflict_template
    else:
        if "{country}" not in args.conflict_template or "{distractor}" not in args.conflict_template:
            raise SystemExit(
                "--conflict-template must be a preset or contain "
                "{country} and {distractor}; "
                f"got {args.conflict_template!r}"
            )
        conflict_template = args.conflict_template
        conflict_template_name = "custom"

    if args.n_copies < 1:
        raise SystemExit("--n-copies must be >= 1")

    if args.tokenizer_path is not None:
        if args.model_size is not None:
            raise SystemExit(
                "--tokenizer-path and --model-size are mutually exclusive"
            )
        size = None
        tokenizer_label = args.tokenizer_path
        tokenizer = load_tokenizer(args.tokenizer_path)
    else:
        size = resolve_model_size(args.model_size)
        tokenizer_label = f"pythia-{size}"
        tokenizer = load_tokenizer(model_size=size)

    in_path = Path(args.input)
    in_meta, records = _load_jsonl(in_path)
    print(f"Loaded {len(records)} prompts from {in_path}")

    if args.shuffle_seed is not None:
        random.Random(args.shuffle_seed).shuffle(records)
    if args.max_samples is not None:
        records = records[: args.max_samples]
        print(f"Subsampled to {len(records)} prompts (seed={args.shuffle_seed})")

    # Build the (country, capital) pool for the capital_noise filler.
    capital_pool: dict[str, str] = {}
    for r in records:
        country = r["country"]
        true_capital = r["answers"][0] if r.get("answers") else None
        if true_capital:
            capital_pool.setdefault(country, true_capital)
        dc = r.get("distractor_country")
        dist = r.get("distractor")
        if dc and dist:
            capital_pool.setdefault(dc, dist)
    pool_pairs = sorted(capital_pool.items())
    print(f"Capital pool: {len(pool_pairs)} (country, capital) pairs")

    # For prose, the filler is identical across all prompts.
    if args.filler_kind == "prose":
        shared_prose = _tile_to_n_tokens(PROSE_SNIPPET, args.length_tokens, tokenizer)
        shared_prose_tokens = len(_encode(shared_prose, tokenizer))
        print(
            f"Prose filler: target={args.length_tokens} tokens, "
            f"actual={shared_prose_tokens} tokens."
        )

    # For wiki_*, load the country -> description map once.
    wiki_descriptions: dict[str, str] = {}
    wiki_country_capital: dict[str, str] = {}
    if args.filler_kind in (
        "wiki_country", "wiki_distractor_country", "wiki_unrelated_country",
        "wiki_scrambled", "wiki_shuffled",
    ):
        if not args.country_descriptions:
            raise SystemExit(
                f"--filler-kind={args.filler_kind} requires "
                "--country-descriptions <path/to/country_descriptions.json> "
                "(produced by scripts/stage_country_descriptions.py)"
            )
        desc_path = Path(args.country_descriptions)
        if not desc_path.is_file():
            raise SystemExit(
                f"--country-descriptions file not found: {desc_path}"
            )
        with open(desc_path, "r", encoding="utf-8") as f:
            blob = json.load(f)
        # Accept either {country: "desc"} or {country: {description, ...}}.
        for k, v in blob.items():
            if k.startswith("_"):
                continue
            if isinstance(v, str):
                wiki_descriptions[k] = v
            elif isinstance(v, dict) and "description" in v:
                wiki_descriptions[k] = v["description"]
                if "capital" in v:
                    wiki_country_capital[k] = v["capital"]
        # Fall back to the per-record true_capital / distractor when the
        # JSON does not explicitly list a capital for this country.
        all_wiki_countries_sorted = sorted(wiki_descriptions)
        print(
            f"Loaded {len(wiki_descriptions)} country descriptions from "
            f"{desc_path}"
        )

    out_records: list[dict] = []
    actual_filler_lens: list[int] = []
    prompt_token_lens: list[int] = []
    skipped_no_wiki: int = 0
    for idx, rec in enumerate(records):
        country = rec["country"]
        original_distractor = rec["distractor"]
        distractor_country = rec.get("distractor_country", "") or ""
        true_capital = rec["answers"][0] if rec.get("answers") else ""

        if args.distractor_kind == "synthetic":
            key = f"{country}|{original_distractor}".encode("utf-8")
            h = int(hashlib.sha256(key).hexdigest()[:8], 16) % len(SYNTHETIC_DISTRACTORS)
            distractor = SYNTHETIC_DISTRACTORS[h]
            distractor_country = ""
        else:
            distractor = original_distractor

        if args.filler_kind == "prose":
            filler = shared_prose
        elif args.filler_kind == "varied_prose":
            filler = _build_varied_prose_filler(
                n_tokens=args.length_tokens,
                tokenizer=tokenizer,
                seed=int(args.shuffle_seed) * 100003 + idx,
            )
        elif args.filler_kind == "prose_facts":
            filler = _build_prose_facts_filler(
                n_tokens=args.length_tokens,
                tokenizer=tokenizer,
                seed=int(args.shuffle_seed) * 100003 + idx,
                facts_every_tokens=args.facts_every_tokens,
                exclude_capitals={true_capital, distractor},
            )
        elif args.filler_kind in (
            "wiki_country", "wiki_distractor_country",
            "wiki_unrelated_country", "wiki_scrambled", "wiki_shuffled",
        ):
            if args.filler_kind == "wiki_country":
                wiki_target = country
                redact = true_capital
            elif args.filler_kind == "wiki_distractor_country":
                wiki_target = distractor_country
                redact = distractor
            else:
                # Pick a deterministic third country, excluding queried
                # and distractor countries.  Same selection rule as
                # wiki_unrelated_country -- wiki_scrambled and
                # wiki_shuffled use it too so they share the same
                # source-passage distribution as the unrelated baseline.
                rng_local = random.Random(
                    int(args.shuffle_seed) * 100003 + idx
                )
                excluded = {country, distractor_country}
                pool_unrelated = [
                    c for c in all_wiki_countries_sorted if c not in excluded
                ]
                if not pool_unrelated:
                    skipped_no_wiki += 1
                    continue
                wiki_target = rng_local.choice(pool_unrelated)
                redact = wiki_country_capital.get(wiki_target, "")
            description = wiki_descriptions.get(wiki_target, "")
            if not description:
                skipped_no_wiki += 1
                continue
            if args.filler_kind == "wiki_scrambled":
                filler = _build_wiki_scrambled_filler(
                    n_tokens=args.length_tokens,
                    tokenizer=tokenizer,
                    description=description,
                    redact_capital=redact,
                    seed=int(args.shuffle_seed) * 100003 + idx,
                )
            elif args.filler_kind == "wiki_shuffled":
                filler = _build_wiki_shuffled_filler(
                    n_tokens=args.length_tokens,
                    tokenizer=tokenizer,
                    description=description,
                    redact_capital=redact,
                    seed=int(args.shuffle_seed) * 100003 + idx,
                )
            else:
                filler = _build_wiki_filler(
                    n_tokens=args.length_tokens,
                    tokenizer=tokenizer,
                    description=description,
                    redact_capital=redact,
                )
        else:
            exclude_countries = {country, distractor_country}
            exclude_capitals = {true_capital, distractor}
            filler = _build_capital_noise_filler(
                n_tokens=args.length_tokens,
                tokenizer=tokenizer,
                pool=pool_pairs,
                exclude_countries=exclude_countries,
                exclude_capitals=exclude_capitals,
                seed=int(args.shuffle_seed) * 100003 + idx,
                falsify=(args.filler_kind == "capital_noise_false"),
            )

        prompt = assemble_prompt(
            country=country,
            distractor=distractor,
            filler=filler,
            position=args.conflict_position,
            n_copies=args.n_copies,
            tokenizer=tokenizer,
            question_template=question_template,
            conflict_template=conflict_template,
        )

        filler_tokens = len(_encode(filler, tokenizer)) if filler else 0
        prompt_tokens = len(_encode(prompt, tokenizer))
        actual_filler_lens.append(filler_tokens)
        prompt_token_lens.append(prompt_tokens)

        out_rec = dict(rec)
        out_rec["prompt"] = prompt
        out_rec["context_len_tokens"] = filler_tokens
        out_rec["prompt_tokens"] = prompt_tokens
        out_rec["filler_kind"] = args.filler_kind
        out_rec["conflict_position"] = args.conflict_position
        out_rec["n_copies"] = args.n_copies
        out_rec["conflict_template_name"] = conflict_template_name
        out_rec["distractor"] = distractor
        out_rec["distractor_kind"] = args.distractor_kind
        if args.distractor_kind == "synthetic":
            out_rec["original_distractor"] = original_distractor
            out_rec["distractor_country"] = distractor_country
        out_records.append(out_rec)

    def _median(xs: list[int]) -> int:
        if not xs:
            return 0
        s = sorted(xs)
        return int(s[len(s) // 2])

    median_prompt_tokens = _median(prompt_token_lens)
    max_prompt_tokens = max(prompt_token_lens) if prompt_token_lens else 0
    context_window = int(args.context_window) if args.context_window else 0
    # L_rel = relative input length (Veseli et al. 2025): prompt length as a
    # fraction of the model's context window.  Recorded for both the target
    # filler length and the realised median/max prompt length so downstream
    # plots can use the relative-length axis without re-tokenizing.
    if context_window > 0:
        l_rel_target = args.length_tokens / context_window
        l_rel_median_prompt = median_prompt_tokens / context_window
        l_rel_max_prompt = max_prompt_tokens / context_window
    else:
        l_rel_target = None
        l_rel_median_prompt = None
        l_rel_max_prompt = None

    meta = {
        "_meta": True,
        "source_input": str(in_path),
        "n_prompts": len(out_records),
        "length_tokens_target": args.length_tokens,
        "filler_kind": args.filler_kind,
        "facts_every_tokens": (
            args.facts_every_tokens if args.filler_kind == "prose_facts" else None
        ),
        "conflict_position": args.conflict_position,
        "n_copies": args.n_copies,
        "question_template": question_template,
        "question_template_name": question_template_name,
        "conflict_template": conflict_template,
        "conflict_template_name": conflict_template_name,
        "distractor_kind": args.distractor_kind,
        "tokenizer_model_size": size,
        "tokenizer_label": tokenizer_label,
        "context_window": context_window,
        "l_rel_target": l_rel_target,
        "l_rel_median_prompt": l_rel_median_prompt,
        "l_rel_max_prompt": l_rel_max_prompt,
        "max_samples": args.max_samples,
        "shuffle_seed": args.shuffle_seed,
        "median_filler_tokens": _median(actual_filler_lens),
        "median_prompt_tokens": median_prompt_tokens,
        "max_prompt_tokens": max_prompt_tokens,
        "input_meta": in_meta,
        "country_descriptions_path": (
            str(args.country_descriptions) if args.country_descriptions else None
        ),
        "skipped_no_wiki": skipped_no_wiki,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(meta) + "\n")
        for r in out_records:
            f.write(json.dumps(r) + "\n")
    print(
        f"Wrote {len(out_records)} prompts to {out_path} "
        f"(median filler={meta['median_filler_tokens']} tok, "
        f"median prompt={meta['median_prompt_tokens']} tok, "
        f"max prompt={meta['max_prompt_tokens']} tok)"
    )
    if skipped_no_wiki:
        print(
            f"  warning: skipped {skipped_no_wiki} prompts because the "
            "required country description was missing from "
            f"{args.country_descriptions}"
        )


if __name__ == "__main__":
    main()
