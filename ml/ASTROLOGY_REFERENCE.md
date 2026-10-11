# Astrology reference — design input for the zodiac rubric

The learned zodiac-to-music mapping failed. 1,325 scraped playlists produced 11.6%
balanced accuracy against a 8.33% chance baseline — 1.39×, real but useless (`README.md`).
Strangers' playlist titles are not a measurement of anything, so we stopped trying to
*learn* the mapping and started trying to *design* it.

This document is the design input. It is not evidence about people. It exists so that the
rubric the team builds next is traceable to a source rather than to whoever was typing.

---

## How to read this

Every substantive claim carries one of these labels. **Nothing in this document is
allowed to appear without one.** The single most common failure mode in writing about
astrology is sliding from "the tradition says" to "it is the case", and the labels exist
to make that slide impossible to perform by accident.

| label | means |
|---|---|
| **[TRADITION]** | A documented claim of the astrological tradition. Design input. Says nothing about whether it is true. |
| **[STUDIES]** | An empirical finding from the research literature, with an N and a result. |
| **[CONTESTED]** | Sources genuinely disagree. Both positions recorded. |
| **[UNFALSIFIABLE]** | Cannot be tested even in principle as stated. Usable as flavour, never as a premise. |
| **[FOLK]** | Popular / editorial convention with no cited basis. Useful for knowing audience expectations, nothing else. |

Two further rules for anyone editing this file:

1. **A claim with no URL gets deleted, not fixed.** There is no "I'm pretty sure" tier.
2. **Where a number is widely repeated but could not be traced to its primary source,
   it is marked as such.** See §2.1 for a live example — the single most-cited number
   in this entire literature appears in three different versions.

---

# Part 0 — A provenance problem to settle before anything else

Before documenting "what the tradition says about Aries", we have to be honest about
which tradition, and how old it actually is.

**[STUDIES] Sun-sign astrology — one sign per person, derived from the Sun's position
alone — is a 20th-century newspaper product, not an ancient doctrine.** Its origins are
"unclear", but it "was largely popularized by horoscopes which began appearing in English
newspapers in the 1930s", and by 1937 the astrologer R.H. Naylor was writing a regular
column, *Your Stars*, in the *Sunday Express* "which featured horoscopes based on the 12
star signs"
([Wikipedia, Sun sign astrology](https://en.wikipedia.org/wiki/Sun_sign_astrology)).

**[TRADITION] Classical and traditional Western astrology does not work this way.** It
reads a whole chart — the Sun plus "the other six 'planets'", their signs, their mutual
angles and their house placements (ibid.). Reducing that to a single Sun sign is a
simplification introduced for mass-media distribution.

### Why this matters to our design, concretely

Our product is a **sun-sign** product. So:

- We **cannot** claim the twelve personality portraits in §1.7 are what "astrologers have
  said for millennia". They are largely a 1930s-onward popular synthesis. Saying otherwise
  in the app copy would be a false historical claim, and an easily checkable one.
- We **can** legitimately ground the rubric in elements, modalities, polarity and
  rulerships (§1.2–§1.6). Those *are* genuinely ancient, they *are* attached to signs
  rather than to whole charts, and they are attested in primary sources — Ptolemy's
  *Tetrabiblos*, 2nd century CE
  ([Tetrabiblos Book I, Chs. 11–12, 17–18](https://penelope.uchicago.edu/Thayer/E/Roman/Texts/Ptolemy/Tetrabiblos/1B*.html)).

This is a point in favour of the plan, not against it. The two-axis derivation is the part
of the tradition with real documentary depth. The twelve hand-written personality blurbs
are the part with none.

### One more thing the app must never imply

**[STUDIES] The tropical signs no longer correspond to the constellations they are named
for.** Because of the precession of the equinoxes, the tropical zodiac — anchored to the
March equinox, not to the stars — has drifted roughly 24° from the sidereal positions, at
about 1° every 72 years, which is nearly a full sign. The Sun at tropical 0° Aries now
sits against the stars of Pisces
([Wikipedia, Western astrology](https://en.wikipedia.org/wiki/Western_astrology);
[Stars in astrology](https://en.wikipedia.org/wiki/Stars_in_astrology)).

Tropical astrology is internally consistent about this — it was never claiming to track
the constellations, only the seasons (ibid.). But any app copy saying "the stars were in
Aries when you were born" is simply false. Say *sign*, never *constellation*, and never
"where the stars actually were".

---

# Part 1 — The actual structure of the tradition

## 1.1 The headline structural result

**The assumption in the brief holds. Element × modality produces exactly twelve unique
combinations, one per sign, with no collisions and no gaps.** Verified sign by sign in
§1.4. Two independent reference sources state the bijection explicitly, and it also
follows from arithmetic (§1.4.1).

There is one caveat, and it is a real one rather than a pedantic one: **the clean
three-modality scheme is itself partly a modern tidying of a four-category classical
scheme.** Ptolemy splits what we now call "cardinal" into two separate groups. See §1.3.2.
That does not break the 4×3 derivation — it means the derivation is grounded in modern
astrological convention, which is well documented, rather than in Ptolemy directly.

## 1.2 The four elements (triplicities)

**[TRADITION]** Four elements, three signs each, spaced 120° apart on the ecliptic —
hence "triplicity" / "trigon" / "triangle"
([Wikipedia, Astrology and the classical elements](https://en.wikipedia.org/wiki/Astrology_and_the_classical_elements)).

Attested in primary source: Ptolemy, *Tetrabiblos* I.18, describes exactly these four
"triangles"
([Tetrabiblos I.18](https://penelope.uchicago.edu/Thayer/E/Roman/Texts/Ptolemy/Tetrabiblos/1B*.html)).

| element | signs | classical qualities (Manilius) | temperament terms (modern reference) |
|---|---|---|---|
| **Fire** | Aries, Leo, Sagittarius | "hot, dry, ardent" | "spontaneous, reactive, and assertive"; "pride, passion, courage, creativity, and impulse" |
| **Earth** | Taurus, Virgo, Capricorn | "heavy, cold, dry" | "reliable, practical, and sensual"; they "collect and build", seek solid results |
| **Air** | Gemini, Libra, Aquarius | "light, hot, wet" | "intellectual, detached, and fair-minded"; "humanitarian, intellectual, social, and communicative" |
| **Water** | Cancer, Scorpio, Pisces | "cold, wet, soft" | "fluid, emotional, and intuitive"; "rely more on their senses than logic"; "nurturing, perceptive, psychic, and understanding" |

Classical qualities quoted from Manilius via
[Wikipedia, Astrology and the classical elements](https://en.wikipedia.org/wiki/Astrology_and_the_classical_elements).
Temperament terms quoted from
[Cafe Astrology, The Elements in Astrology](https://cafeastrology.com/natal/elements-astrology.html).

### 1.2.1 [CONTESTED] A seasonal element scheme that may or may not conflict

Two Wikipedia articles describe the ancient seasonal treatment of triplicities
differently, and **we could not resolve it**:

- [Astrology and the classical elements](https://en.wikipedia.org/wiki/Astrology_and_the_classical_elements)
  states that "in ancient astrology, triplicities were more of a seasonal nature",
  assigning **spring–air, summer–fire, autumn–earth, winter–water**. Taken literally that
  is a *different element assignment* from the standard one (spring is Aries/Taurus/Gemini,
  which under the standard scheme are Fire/Earth/Air).
- [Triplicity](https://en.wikipedia.org/wiki/Triplicity) describes the seasonal scheme
  instead as a parallel *grouping by season* (Spring = Aries–Taurus–Gemini, etc.) that sits
  alongside the elemental grouping rather than replacing it, and notes that
  "neither are given much attention by modern astrologers". It states there is no evidence
  ancient sources assigned elements differently.

**Action for the rubric:** use the standard elemental assignment in §1.2. It is the one
attested in Ptolemy I.18, the one every modern reference uses, and the one the audience
expects. Do not cite the seasonal scheme at all until someone reads a primary source on
it. Recorded here only so nobody rediscovers the discrepancy and assumes it was missed.

Also note: the seasonal scheme inverts in the Southern Hemisphere, whereas the elemental
scheme does not ([Triplicity](https://en.wikipedia.org/wiki/Triplicity)). A global product
has an obvious reason to prefer the non-inverting one.

### 1.2.2 [TRADITION] Triplicity rulers — a layer we are deliberately not using

Each triplicity has day/night/participating planetary rulers, and these differ between
authorities. The Dorothean system gives Fire: Sun (day), Jupiter (night), Saturn
(participating); Earth: Venus, Moon, Mars; Air: Saturn, Mercury, Jupiter; Water: Venus,
Mars, Moon. **Ptolemy then modified the Water triplicity, making Mars ruler for both day
and night charts, and William Lilly concurred**
([Triplicity](https://en.wikipedia.org/wiki/Triplicity)). Ptolemy I.18 as transmitted gives
the Water triangle to Mars with Moon and Venus as co-rulers
([Tetrabiblos I.18](https://penelope.uchicago.edu/Thayer/E/Roman/Texts/Ptolemy/Tetrabiblos/1B*.html)).

This is recorded as an example of how quickly the tradition stops agreeing with itself once
you go one layer below the headline structure. We are not using triplicity rulers. One
disagreement at this level of specificity is enough to make the layer unusable as a
design axis.

## 1.3 The three modalities (quadruplicities)

**[TRADITION]** Three modalities, four signs each, spaced 90° apart — hence
"quadruplicity". Also called *modes* or *qualities*.

| modality | signs | traditional meaning | seasonal logic |
|---|---|---|---|
| **Cardinal** | Aries, Cancer, Libra, Capricorn | initiating, "active, involved, ambitious and opportunistic" | each begins a season — the two equinoxes (Aries, Libra) and two solstices (Cancer, Capricorn) |
| **Fixed** | Taurus, Leo, Scorpio, Aquarius | consolidating, stabilising, resistant to change | each sits mid-season, following a cardinal sign |
| **Mutable** | Gemini, Virgo, Sagittarius, Pisces | adapting, transitional, dispersing | each ends a season, immediately before the next cardinal sign |

Groupings and the "active, involved, ambitious and opportunistic" phrasing from
[Cafe Astrology](https://cafeastrology.com/natal/elements-astrology.html) and
[Astrology42, Modalities](https://www.astrology42.com/modalities/).

### 1.3.1 [TRADITION] Older terminology

Ptolemy, *Tetrabiblos* I.11, classifies the signs as:

- **Solstitial** (tropical): Cancer, Capricorn — where "the sun turns… and reverses his
  latitudinal progress"
- **Equinoctial**: Aries, Libra — where "the sun makes the nights exactly equal to the days"
- **Solid** (fixed): Taurus, Leo, Scorpio, Aquarius — "following the solstitial and
  equinoctial signs"
- **Bicorporeal** (double-bodied): Gemini, Virgo, Sagittarius, Pisces — positioned
  "between the solid and the solstitial and equinoctial signs"

([Tetrabiblos I.11](https://penelope.uchicago.edu/Thayer/E/Roman/Texts/Ptolemy/Tetrabiblos/1B*.html))

The modern names *movable / fixed / common* also appear in older English-language sources
for cardinal / fixed / mutable.

### 1.3.2 [CONTESTED] The classical scheme has four categories, not three

This is worth stating plainly because it is the one place where the brief's assumed
structure is a modern simplification rather than an ancient one.

Ptolemy's I.11 classification is **2 + 2 + 4 + 4 = 12**. Solstitial (2) and equinoctial (2)
are *separate categories* for him. Modern astrology merges them into a single
four-member "cardinal" group, producing **4 + 4 + 4 = 12**.

**What this does and does not break:**

- It does **not** break the 4×3 derivation. The modern three-modality scheme is
  universally used and consistently documented across modern references, and it is the
  scheme the audience knows.
- It **does** mean we should describe the modality axis as *modern astrological convention*,
  not as "what Ptolemy said". Ptolemy's equinoctial/solstitial distinction is a genuinely
  different cut.
- It also means the cardinal group is the least internally coherent of the three on
  classical grounds: Aries and Libra (equinoxes) and Cancer and Capricorn (solstices) were
  originally two different things. If the rubric produces weaker or mushier descriptions
  for cardinal signs than for fixed and mutable ones, this is the likely reason, and it is
  predictable in advance rather than a bug.

## 1.4 The 4 × 3 table — verified sign by sign

**[TRADITION]** Each cell is occupied exactly once. Each sign appears exactly once.

| | **Cardinal** | **Fixed** | **Mutable** |
|---|---|---|---|
| **Fire** | Aries | Leo | Sagittarius |
| **Earth** | Capricorn | Taurus | Virgo |
| **Air** | Libra | Aquarius | Gemini |
| **Water** | Cancer | Scorpio | Pisces |

Transposed as a per-sign check, in zodiacal order:

| # | sign | element | modality | polarity |
|---|---|---|---|---|
| 1 | Aries | Fire | Cardinal | positive |
| 2 | Taurus | Earth | Fixed | negative |
| 3 | Gemini | Air | Mutable | positive |
| 4 | Cancer | Water | Cardinal | negative |
| 5 | Leo | Fire | Fixed | positive |
| 6 | Virgo | Earth | Mutable | negative |
| 7 | Libra | Air | Cardinal | positive |
| 8 | Scorpio | Water | Fixed | negative |
| 9 | Sagittarius | Fire | Mutable | positive |
| 10 | Capricorn | Earth | Cardinal | negative |
| 11 | Aquarius | Air | Fixed | positive |
| 12 | Pisces | Water | Mutable | negative |

**12 signs, 12 distinct (element, modality) pairs, 12 cells filled. No collisions, no gaps.**

Two reference sources state the bijection directly:

- "each zodiac sign shares an element and quality (or mode) with other signs, but none
  share both" — [Cafe Astrology](https://cafeastrology.com/natal/elements-astrology.html)
- "For each mode, there is one of each of the astrological elements. For each element,
  there is one of each astrological quality." —
  [Astrology42](https://www.astrology42.com/modalities/)

### 1.4.1 Why it is forced, not coincidental

Worth understanding, because it tells us the structure is load-bearing rather than
decorative. Walking the zodiac in order:

- element cycles with period **4** (Fire, Earth, Air, Water, repeat)
- modality cycles with period **3** (Cardinal, Fixed, Mutable, repeat)

Since gcd(3, 4) = 1, the joint cycle has period lcm(3, 4) = **12** — so over twelve
consecutive signs every (element, modality) pair occurs exactly once. This is the Chinese
Remainder Theorem; a sign's position mod 12 is uniquely recoverable from its element
(position mod 4) and its modality (position mod 3).

**Design consequence, and it is the important one:** the twelve categories are genuinely
*derivable* from two axes of 4 and 3 levels. A rubric needs to specify **4 + 3 = 7**
things, not 12, and the twelve outputs follow by composition. Any sign description that
cannot be reconstructed from its element and its modality is an unsourced addition, and
should be visible as such in review. That is the auditability property the brief asked for,
and the arithmetic above is what guarantees it.

## 1.5 Ruling planets

### 1.5.1 [TRADITION] Classical rulerships (seven planets), with Ptolemy's rationale

Ptolemy, *Tetrabiblos* I.17, derives these geometrically. Cancer and Leo are "the most
northern" and closest to the zenith, so they were assigned "to the luminaries, as houses,
Leo… to the sun and Cancer… to the moon". The other five planets then get two signs each,
assigned outward by aspect from the luminaries
([Tetrabiblos I.17](https://penelope.uchicago.edu/Thayer/E/Roman/Texts/Ptolemy/Tetrabiblos/1B*.html)):

| planet | signs | Ptolemy's stated reason |
|---|---|---|
| Sun | Leo | northernmost, nearest zenith |
| Moon | Cancer | as above |
| Mercury | Gemini, Virgo | adjacent to the luminaries' houses — "he never is farther removed from the sun than one sign" |
| Venus | Taurus, Libra | sextile to the luminaries — "never more than two signs removed from the sun" |
| Mars | Aries, Scorpio | quartile (square) to the luminaries |
| Jupiter | Sagittarius, Pisces | trine to the luminaries |
| Saturn | Capricorn, Aquarius | opposite the luminaries' houses |

This is a tidy, symmetric, *internally motivated* scheme: the Sun and Moon take the two
signs nearest the zenith, and the remaining planets fan outward in order of their maximum
distance from the Sun. Whatever else is true of it, it is not arbitrary in construction.

### 1.5.2 [CONTESTED] Modern rulerships

After the discovery of Uranus (1781), Neptune (1846) and Pluto (1930), modern astrology
reassigned three signs:

| sign | classical ruler | modern ruler |
|---|---|---|
| Scorpio | Mars | Pluto |
| Aquarius | Saturn | Uranus |
| Pisces | Jupiter | Neptune |

Traditional-revival astrologers reject these reassignments and keep the seven-planet
scheme; most popular astrology uses the modern one, often citing both as "ruler" and
"co-ruler". **This is a live disagreement within the field, not a settled fact.**

**Action for the rubric:** if rulerships are used at all, use the **classical** scheme.
Three reasons: it is attested in a primary source with a stated derivation (§1.5.1); it is
not contested; and it does not require us to take a position on whether Pluto's 2006
reclassification as a dwarf planet affects its astrological rulership, a question with no
non-arbitrary answer. **[UNFALSIFIABLE]** — flagged because someone will ask.

## 1.6 Polarity

**[TRADITION]** The signs alternate in polarity around the zodiac, odd signs one way and
even signs the other.

Attested in primary source: Ptolemy, *Tetrabiblos* I.12, assigns "six of the signs to the
masculine and diurnal nature and an equal number to the feminine and nocturnal",
alternating — masculine: Aries, Gemini, Leo, Libra, Sagittarius, Aquarius; feminine:
Taurus, Cancer, Virgo, Scorpio, Capricorn, Pisces
([Tetrabiblos I.12](https://penelope.uchicago.edu/Thayer/E/Roman/Texts/Ptolemy/Tetrabiblos/1B*.html)).

Ptolemy I.18 independently describes the Fire and Air triangles as masculine and the Earth
triangle as feminine (ibid.), which gives the mapping onto elements:

| polarity | also called | elements | signs |
|---|---|---|---|
| positive | masculine, diurnal, yang, active | **Fire + Air** | Aries, Gemini, Leo, Libra, Sagittarius, Aquarius |
| negative | feminine, nocturnal, yin, receptive | **Earth + Water** | Taurus, Cancer, Virgo, Scorpio, Capricorn, Pisces |

### 1.6.1 Polarity is not an independent axis — do not treat it as one

**Polarity is fully determined by element.** Fire and Air are positive; Earth and Water are
negative. It carries no information the element axis does not already carry.

Including it in the rubric as a third axis would be double-counting — it would silently
weight the element axis twice. Use it only as *vocabulary* for describing element groupings
(e.g. the shared "outward-directed" quality of Fire and Air), never as an input.

### 1.6.2 Terminology

The "masculine/feminine" labels are the historical ones and are what the primary sources
use. Modern references increasingly prefer **positive/negative**, **active/receptive**, or
**yang/yin**, since the older terms carry gender implications the modern usage does not
intend. **For user-facing copy, use active/receptive.** It is accurate to the tradition's
actual content, and avoids importing a gender claim into a music app that has no reason to
make one.

## 1.7 The twelve signs — and why this section is the weak one

**[TRADITION] / [FOLK], and the distinction matters more here than anywhere else.**

Per §0, the detailed modern personality portraits of the twelve sun signs are substantially
a 20th-century popular synthesis. They are also where reference sources diverge most. Below
is the consensus shape, with the divergence recorded.

Element and modality are repeated in each row because, per §1.4.1, **the traits should be
reconstructible from them.** Where a listed trait does not follow from the two axes, that is
flagged.

| sign | el. | mod. | most consistently attributed | notes and divergences |
|---|---|---|---|---|
| Aries | Fire | Card. | initiating, assertive, impulsive, competitive, direct | Follows cleanly from cardinal + fire. |
| Taurus | Earth | Fixed | steady, sensual, patient, possessive, change-averse | Follows cleanly. "Indulgent" vs "disciplined" split across sources. |
| Gemini | Air | Mut. | verbal, curious, quick, inconsistent, dual | Follows cleanly. "Dual/two-faced" derives from the Twins imagery, not the axes. |
| Cancer | Water | Card. | nurturing, protective, home-oriented, moody | "Initiating" (cardinal) sits awkwardly with "retreating", a recurring tension. |
| Leo | Fire | Fixed | expressive, proud, generous, theatrical, needs recognition | Follows cleanly. |
| Virgo | Earth | Mut. | analytical, exacting, service-oriented, critical | "Analytical" is attributed to Mercury rulership rather than to Earth + mutable. |
| Libra | Air | Card. | relational, aesthetic, diplomatic, indecisive | "Indecisive" directly contradicts cardinal = initiating. See below. |
| Scorpio | Water | Fixed | intense, private, controlled, investigative, loyal | Follows cleanly. The most *favourably* described sign in popular sources despite "dark" framing. |
| Sagittarius | Fire | Mut. | expansive, exploratory, blunt, restless, philosophical | Follows cleanly. |
| Capricorn | Earth | Card. | ambitious, disciplined, structural, reserved, long-horizon | Follows cleanly; the best-behaved cell. |
| Aquarius | Air | Fixed | inventive, detached, contrarian, systems-minded, humanitarian | "Inventive/unpredictable" contradicts fixed = stable. Often attributed to Uranus (modern rulership) instead. |
| Pisces | Water | Mut. | empathic, dreamy, boundaryless, artistic, escapist | Follows cleanly. |

### 1.7.1 The disagreements, which are the actually informative part

The brief asked for these explicitly, because the disagreement is itself a measure of how
firm any of this is. Four distinct kinds:

**(a) Direct internal contradictions with the structure.** Libra is Cardinal (initiating)
and also canonically "indecisive". Aquarius is Fixed (stable, resistant) and also
canonically "unpredictable, revolutionary". Cancer is Cardinal and also canonically
"retreating, shell-dwelling". In each case the sign's folk personality fights its own
structural position. These are the three cells where a derived rubric will most visibly
diverge from audience expectation — **know about it in advance and decide deliberately
which to follow.**

**(b) Traits assigned to multiple signs.** "Loyal" is attributed to Taurus, Cancer, Leo,
Scorpio and Capricorn depending on the source. "Creative" to Pisces, Leo, Libra and
Aquarius. "Intelligent" to Virgo, Aquarius, Gemini and Scorpio. The trait vocabulary is not
partitioned across the signs, so trait lists do not discriminate between signs — which is
one mechanistic reason why *any* classifier trained on them, ours included, has so little
to work with.

**(c) Same sign described as X and not-X.** Taurus as "indulgent" and "disciplined";
Scorpio as "intensely loyal" and "vengeful"; Pisces as "deeply empathic" and "escapist,
self-absorbed". Note that each pair is a flattering/unflattering version of one underlying
disposition — which is precisely the Barnum construction described in §2.1, and lets a
reader confirm the description whichever way they actually are.

**(d) Date boundaries disagree.** Sources differ by a day or two on sign cusp dates, since
the Sun's ingress into a sign shifts year to year and by time zone. Any fixed date table is
wrong for some users near a boundary. **Action: either compute the ingress properly, or
state that boundary dates are approximate. Do not silently hard-code a date table and
present the result as certain.** This is the same class of error as `BIAS.md` #6 —
displaying more precision than the input supports.

### 1.7.2 [UNFALSIFIABLE] Things to use as flavour and never as a premise

Recorded so they are not mistaken for claims: that signs "rule" body parts; that planetary
"energies" or "vibrations" influence temperament; that signs correspond to musical keys or
frequencies (asserted by e.g.
[Gongs Unlimited, Zodiac Tone Circles](https://gongs-unlimited.com/blogs/unlimited-blog/zodiac-tone-circles-the-signs-and-sound-healing),
with no stated basis). These are not false so much as not formulated in a way that could be
checked. They may appear as decoration. They may not appear as reasons.

## 1.8 Prior art on zodiac-to-music mappings: essentially none

Worth stating because it sets expectations for the rubric's status.

**[STUDIES]** We found **no peer-reviewed work mapping zodiac signs to musical attributes
or audio features.** The literature on music and individual differences exists (§2.7) but
does not touch astrology.

**[FOLK]** What exists is editorial. The most-cited example maps elements to music taste
as follows: Fire signs "prefer upbeat and energizing music"; Earth signs "like music
that's more organic-sounding, sensual, and soulful"; Air signs "love thoughtful lyrics but
can still jam out to futuristic-sounding techno"; Water signs "prefer music they can lose
themselves in, as they enjoy more artistic, emotional, and romantic tunes"
([Bustle, Neal & Kahn, 24 Sep 2017, upd. 20 Feb 2024](https://www.bustle.com/life/how-your-zodiac-sign-can-affect-your-taste-in-music-2438088)).
The article's only cited authority is an interview with one astrologer; it presents no
evidence.

### 1.8.1 The useful finding hiding in the folk mapping

The Bustle per-sign list is **already approximately element × modality**, apparently
without intending to be:

- Cardinal Fire (Aries) → "high-energy dance beats and self-confident anthems" — initiating
- Fixed Fire (Leo) → "theatrical, fun, passionate, and dramatic" — sustained display
- Mutable Fire (Sagittarius) → "high-energy, eclectic styles across multiple genres" — roving

Same fire, differentiated along exactly the modality axis. The same pattern holds for the
Air signs (Gemini "fast-paced, lyrically-driven… creative wordplay"; Aquarius "futuristic,
experimental, electronic"; Libra "lyrical love songs").

**This is the strongest practical argument for the two-axis design.** A rubric derived from
element × modality will land close to what the audience already expects, because the folk
convention is an unwitting instance of the same structure. We get auditability *and*
recognisability from the same construction, rather than trading one for the other.

---

# Part 2 — The psychology, which is the part that actually matters

Part 1 is where the content comes from. **Part 2 is why the product works at all.** None
of the mechanisms below require astrology to be true. That is the point, and it is what
makes the design honest rather than merely lucky.

## 2.1 The Forer / Barnum effect

**[STUDIES]** The founding study: Bertram R. Forer, "The fallacy of personal validation: a
classroom demonstration of gullibility", *Journal of Abnormal and Social Psychology*,
44(1), 118–123, January 1949
([Ovid record](https://www.ovid.com/journals/tjoaas/fulltext/00032378-194901000-00014~the-fallacy-of-personal-validation-a-classroom-demonstration)).

Design: **39 psychology students** completed Forer's "Diagnostic Interest Blank", then each
received what they believed was an individual personality interpretation. All 39 received
**the same 13-statement sketch**, which Forer had assembled **from a newsstand astrology
book**. Students rated its accuracy on a **0–5 scale**
([Wikipedia, Barnum effect](https://en.wikipedia.org/wiki/Barnum_effect)).

Representative statements, verbatim:

> "You have a great need for other people to like and admire you."

> "At times you are extroverted, affable, sociable, while at other times you are
> introverted, wary, reserved."

(ibid.) Note the construction of the second: it asserts both poles, so it cannot be wrong.
This is item (c) in §1.7.1 — the same device the sign portraits use.

### 2.1.1 [CONTESTED] The most-cited number in this literature has three versions

The brief asked us to get this right, so: **we could not pin the mean rating to a single
value across secondary sources.**

| reported mean | source |
|---|---|
| **4.26** / 5 | [search-surfaced secondary summaries citing Forer 1949](https://www.ovid.com/journals/tjoaas/fulltext/00032378-194901000-00014~the-fallacy-of-personal-validation-a-classroom-demonstration) |
| **4.30** / 5 | [Wikipedia, Barnum effect](https://en.wikipedia.org/wiki/Barnum_effect) |
| **4.2** /5, with "more than 40% gave it the top mark of 5 out of 5" | [Wikipedia, Astrology and science](https://en.wikipedia.org/wiki/Astrology_and_science) |

**4.26 is the figure most commonly quoted.** We did not obtain the 1949 paper itself, so we
cannot say which is the paper's own number. **Cite it as "approximately 4.3 out of 5
(N = 39)" and do not reproduce a two-decimal figure we have not verified.** The spread
between 4.2, 4.26 and 4.30 is secondary-source drift, and reproducing it uncritically is
exactly the failure this document exists to prevent.

The *direction and size* of the effect are not in doubt: a generic sketch given to 39
people was rated above 4 out of 5 for personal accuracy by all of them.

### 2.1.2 The name

**[STUDIES]** Paul Meehl coined "Barnum effect" in "Wanted — A Good Cookbook",
*American Psychologist*, 11(6), 1956, comparing vague test descriptions to P.T. Barnum's
showmanship ([Wikipedia, Barnum effect](https://en.wikipedia.org/wiki/Barnum_effect)).

### 2.1.3 Moderators — i.e. the design dials

**[STUDIES]** The effect is not constant. It is strengthened by:

- **perceived personalisation** of the feedback
- **source authority** (high-status evaluator)
- **positive trait emphasis** over negative
- **vague phrasing** — hedges like "at times" that let the reader supply the specifics

([Wikipedia, Barnum effect](https://en.wikipedia.org/wiki/Barnum_effect); review
literature: Dickson & Kelly, "The 'Barnum Effect' in Personality Assessment: A Review of
the Literature", *Psychological Reports*,
[record](https://www.researchgate.net/publication/232554639_The_'Barnum_Effect'_in_Personality_Assessment_A_Review_of_the_Literature))

**Read that list again as a list of things we could do, and then note that it is also the
list in §3.** Every single moderator is a lever for making a statement feel more true than
it is. "Perceived personalisation" and "source authority" in particular are the two that
turn entertainment into deception, because they are claims about *provenance* rather than
about content. This is the most important paragraph in Part 2 for design purposes.

### 2.1.4 [STUDIES] The effect is explicitly understood as an ethics problem

Barnum statements are "personality descriptions intentionally crafted to be broad enough
for widespread endorsement as self-relevant". The effect has been used *as a teaching tool
about research deception* — students given bogus personality feedback accept it, which is
the demonstration ([Beins, "Using the Barnum Effect To Teach Ethics in Research", 1991,
ERIC ED340747](https://files.eric.ed.gov/fulltext/ED340747.pdf)). Legal and ethical
consequences arise "when such pseudoscientific concepts are marketed in the commercial
market… as valid diagnostic instruments" (ibid.).

We are in the commercial market. §3 is the constraint that follows.

## 2.2 Subjective validation and confirmation bias

**[STUDIES]** Related but distinct from the Barnum effect:

- **Barnum effect** — a *statement* is constructed to be broadly applicable, so almost
  anyone endorses it.
- **Subjective validation** — the *reader* actively supplies meaning, treating a statement
  as accurate because it has personal significance to them. The term comes out of work on
  cold reading (Ray Hyman; Marks & Kammann, *The Psychology of the Psychic*).
- **Confirmation bias** — the reader then preferentially notices, recalls and seeks
  instances that fit, and discounts those that do not. The standard review is Nickerson,
  "Confirmation Bias: A Ubiquitous Phenomenon in Many Guises", *Review of General
  Psychology*, 1998.

The three compose: a statement engineered to fit anyone (Barnum), interpreted into personal
specificity by the reader (subjective validation), then confirmed by selective recall over
following days (confirmation bias). A daily horoscope exploits all three by construction —
it is re-read daily, against fresh events that can be fitted to it.

**Note for us:** a music recommendation has a property a horoscope does not — **you can
check it immediately.** You press play and either like the song or not. That is a real
constraint on how far subjective validation can carry the product, and also a genuine
honesty advantage: our output is falsifiable by the user in about fifteen seconds. We
should lean on that rather than away from it.

## 2.3 Identity and belonging — the strongest mechanism, and the one that needs no deception

This is the section to build the product on.

**[STUDIES] Arbitrary category membership is sufficient to produce real in-group
identification.** The minimal group paradigm — Tajfel, Billig, Bundy & Flament,
*European Journal of Social Psychology*, 1971 — randomly assigned participants to groups on
trivial criteria (preferences between paintings, coin tosses), with anonymity and no
personal stake, and found that "even arbitrary distinctions between groups, such as
preferences for certain paintings, or the color of their shirts, can trigger a tendency to
favor one's own group at the expense of others"
([Wikipedia, Minimal group paradigm](https://en.wikipedia.org/wiki/Minimal_group_paradigm)).
Participants even chose maximum in-group/out-group *differentiation* over maximum absolute
in-group gain. Tajfel developed social identity theory to explain it, proposing that
in-group favouritism in these conditions is "the only way in which to achieve positive
distinctiveness" (ibid.).

**Why this is the load-bearing finding for Lovify:**

Sun signs are an arbitrary 12-way partition of people. The minimal group literature says
that is *exactly enough* to generate identification, in-group affinity, and the pleasure of
"us Scorpios". **The social reality of the category does not depend on its predictive
validity.** It depends only on people knowing which bucket they are in — and with sun
signs, essentially everyone already does, for free, with no onboarding.

So the honest version of the product's value proposition is:

> Sun signs are a pre-installed, universally known, socially meaningful way to divide
> people into twelve groups. We use them as an organising device for music discovery. The
> grouping is arbitrary; the sense of belonging is real; the music is real.

Nothing in that sentence is false, and it is a complete account of why the feature can work.
**No claim of predictive accuracy is needed anywhere in it.** That is worth noticing: the
mechanism that actually drives engagement is the one that requires no deception at all.

## 2.4 Narrative identity

**[STUDIES]** Dan McAdams' life-story model of identity holds that people construct
identity as an internalised, evolving narrative, and draw on culturally available
frameworks to do it. Astrology supplies a ready-made vocabulary of character types,
tensions and arcs — a low-effort scaffold for self-interpretation.

This explains a pattern the Barnum effect alone does not: people engage with sign
descriptions *they disagree with*, and argue about them. That is narrative work (using a
framework to articulate who you are, including by contrast), not simple credulity. It
also predicts that a sign description which is *specific and slightly provocative*
generates more engagement than one which is bland and agreeable — because it gives the user
something to push against.

Design note: this is a legitimate way to get engagement without Barnum engineering. A
description specific enough to be *wrong* about some users is more honest than one vague
enough to be right about all of them — and per the above, probably more engaging too. The
incentives here point the same way as the ethics, which is unusual and worth exploiting.

## 2.5 Who engages, and when

### 2.5.1 [STUDIES] Prevalence and demographics

Pew Research Center, surveying **9,593 U.S. adults, 21–27 October 2024** (published 21 May
2025)
([Pew](https://www.pewresearch.org/religion/2025/05/21/3-in-10-americans-consult-astrology-tarot-cards-or-fortune-tellers/)):

| finding | figure |
|---|---|
| Believe in astrology ("the position of the stars and planets can affect people's lives") | **27%** (2017: 29% — stable) |
| Consult astrology/horoscope, tarot or a fortune teller at least yearly | **30%** |
| Consult astrology or horoscopes specifically, at least yearly | **28%** |
| Engage monthly / weekly | 14% / 6% |
| Believe — women 18–49 | **43%** |
| Believe — women 50+ / men 18–49 / men 50+ | 27% / 20% / 16% |
| Consult at least yearly — women 18–49 | **46%** |
| Consult at least yearly — LGBT adults | **54%** (LGBT women 63%, LGBT men 40%) |

The skew is strong and directly relevant: **young women are roughly 2.7× as likely as older
men to believe**, and the LGBT engagement rate is nearly double the national one. For a
university project this is also a sampling warning — our survey respondents will be drawn
from a population where engagement is well above base rate.

### 2.5.2 [STUDIES] Motivation — the finding that most constrains §3

From the same survey:

| finding | figure |
|---|---|
| Engage mostly **"just for fun"** | **20%** of all U.S. adults |
| Engage mostly because they "believe the practices give them helpful insights" | **10%** |
| Rely **"a lot"** on these practices for major life decisions | **1%** |
| Rely "a little" | 5% |
| LGBT adults relying at least a little for major life decisions | 21% |

([Pew](https://www.pewresearch.org/religion/2025/05/21/3-in-10-americans-consult-astrology-tarot-cards-or-fortune-tellers/))

**Two conclusions, pulling in opposite directions, and both matter:**

1. **The dominant mode is play, by 2:1.** Most users already hold astrology at arm's
   length. A product framed as entertainment is not fighting its audience's expectations —
   it is matching them. This is permission to be playful.
2. **But a minority does rely on it, and it is not negligible.** 10% engage for "helpful
   insights", 6% rely at least a little on major decisions, 21% among LGBT adults. **Any
   design decision that would be fine for the 20% but harmful for the 6% must be evaluated
   against the 6%.** They are the ones for whom a confident-sounding claim does damage.

### 2.5.3 [STUDIES] Engagement under stress and uncertainty

The classic finding is Lillqvist & Lindeman (1998), "Belief in astrology as a strategy for
self-verification and coping with negative life-events", *European Psychologist* — astrology
belief associated with coping with negative life events. Related: Keinan (2002) on magical
thinking increasing under stress; Padgett & Jorgenson (1982) on economic threat and
superstition/astrology interest in interwar Germany.

**[STUDIES]** Clinically, the concern is documented: a NIMHANS review reports that while
"the human mind seeks coping mechanisms during stress and uncertainty", obsessive astrology
use is associated with mental health problems, and that taking readings too literally is
linked to increased isolation, depression, anxiety and aggravation of underlying
psychiatric conditions. A specific mechanism named is a shift toward **external locus of
control** — coming to believe outcomes are governed by outside forces rather than one's own
behaviour, which "can disempower people"
([The South First, summarising the NIMHANS review](https://thesouthfirst.com/health/obsession-to-astrology-use-can-lead-to-mental-health-problems-nimhans-review/)).
We accessed this via a news summary rather than the review itself — **treat as directional,
not precise.**

This is the single most important input to §3. It says the people most drawn to astrology
at any given moment are disproportionately those under stress, and that the specific harm
pathway is displaced agency. **A product that is engineered to feel most insightful exactly
when the user is most uncertain is targeting that vulnerability, whether or not anyone
intended it.**

## 2.6 The empirical record on astrology's validity: uniformly null

**[STUDIES]** This section is for the survey write-up. We pre-registered H0 in
`SURVEY.md`, expecting a null; the literature is the reason that is the right prior.

### 2.6.1 Carlson 1985, the double-blind test

Shawn Carlson, "A double-blind test of astrology", *Nature* 318(6045), 419–425, 1985
([Nature](https://www.nature.com/articles/318419a0) — paywalled;
[ADS record](https://ui.adsabs.harvard.edu/abs/1985Natur.318..419C)).

Design: **28 astrologers** matched **over 100 natal charts** to personality profiles from
the California Psychological Inventory, double-blind. Result: "predictions based on natal
astrology were no better than chance", and the test "clearly refutes the astrological
hypothesis"
([Wikipedia, Astrology and science](https://en.wikipedia.org/wiki/Astrology_and_science)).

**[CONTESTED] The critique must be reported.** Astrology-aligned authors argue the study
was flawed — "unfairly skewed design, disregard for its own stated criteria of evaluation,
irrelevant groupings of data, and rejection of unexpected results" — and that reanalysis
supports astrology
([Currey, "Support for Astrology from the Carlson Double-blind Experiment", via Astrology
News Service](https://astrologynewsservice.com/research/support-for-astrology-from-the-carlson-double-blind-experiment/);
[PDF copy](https://www.talmudology.com/s/support-for-astrology-from-carlson-experiment.pdf)).

**Our position, stated honestly:** we have not read the critique in full and cannot
adjudicate the statistics. We do not need to. **The null conclusion does not rest on
Carlson.** §2.6.2–§2.6.4 are independent, much larger, and point the same way. If Carlson
were withdrawn tomorrow the overall picture would not move. Citing Carlson *alone* would be
a weak argument, and we should not do it.

### 2.6.2 Dean & Kelly — time twins and astrologer reliability

Dean, G. & Kelly, I.W., "Is Astrology Relevant to Consciousness and Psi?", *Journal of
Consciousness Studies*, 10(6–7), 175–198, 2003.

| study | N | result |
|---|---|---|
| "Time twins" | **2,011 sets** of people born within 5 minutes of each other, on 100+ cognitive, behavioural and physical variables | **no effect** |
| Astrologer matching | **45 experienced astrologers**, **160 subjects** selected for extreme EPQ personality traits | astrologers "performed much worse than merely basing decisions off the individuals' ages, and much worse than 45 control subjects who did not use birth charts" |
| Inter-astrologer agreement | 25 studies | agreement **~0.1** |

([Wikipedia, Astrology and science](https://en.wikipedia.org/wiki/Astrology_and_science))

The time-twins design is the strongest available test: if charts carry information, people
born minutes apart at the same place should resemble each other. They do not. And the
reliability finding is arguably more damaging than the validity finding — **at an agreement
of ~0.1, astrologers do not even agree with each other**, which caps validity regardless of
whether any underlying effect exists.

### 2.6.3 Large-N studies

| study | N | result |
|---|---|---|
| Hartmann, Reuter & Nyborg (2006), "The relationship between date of birth and individual differences in personality and general intelligence: A large-scale study", *Personality and Individual Differences* 40(7), 1349–1362 | large-scale | no meaningful relationship between birth date and personality or intelligence |
| Voas (census analysis) | **>20 million** individuals, England and Wales — whether star signs corresponded to marriage arrangements | **no effect** |
| Horoscope-matching studies | 10 studies; participants picked the horoscope they felt was accurate | **no better than chance** |

([Wikipedia, Astrology and science](https://en.wikipedia.org/wiki/Astrology_and_science))

The Voas analysis matters most for scale. At N > 20 million, effects far too small to be of
any practical use would be detectable. Nothing was found.

### 2.6.4 [STUDIES] Self-attribution — the one mechanism that produces real correlations

This is the most important study type for our survey, because it is the one thing that
could make our survey come out *non-null for the wrong reason*.

Early work (Mayo, White & Eysenck, 1978) reported an extraversion/zodiac relationship —
"odd" signs more extraverted. Subsequent work showed the effect **appears only in people who
know what their sun sign is supposed to mean**: "individuals with knowledge of astrology
tend to describe their personalities in terms of traits compatible with their sun signs"
([Wikipedia, Astrology and science](https://en.wikipedia.org/wiki/Astrology_and_science),
citing Wunder, E. (2003), "Self-attribution, sun-sign traits, and the alleged role of
favourableness as a moderator variable", *Personality and Individual Differences* 35(8),
1783–1789).

So the correlation is real but **circular**: people learn their sign's description and
then self-report in line with it. Astrology is not describing them; it is *instructing*
them, and they comply on the questionnaire.

**Direct consequence for `SURVEY.md`, and it is actionable:** if our survey finds
above-chance prediction of sun sign from music taste, **self-attribution is the leading
explanation, not astrology.** The survey should therefore measure **astrology
knowledge/belief** as a covariate, and report accuracy split by it. If any signal appears
only among believers, we have replicated Wunder, not discovered anything. This is cheap to
add — one or two questions — and it is the difference between an interpretable null and an
uninterpretable weak positive. **Recommend adding it before the survey goes out.**

### 2.6.5 [CONTESTED] The Gauquelin "Mars effect"

The strongest positive claim astrology's defenders cite. Michel Gauquelin reported a
correlation between Mars' diurnal position and athletic eminence in **570 birth charts**.
A replication by seven French scientists "found no statistical evidence" and attributed the
original to "selective bias on Gauquelin's part"
([Benski et al., *The "Mars effect": a French test of over 1,000 sports champions*,
Prometheus Books, 1995, via Wikipedia](https://en.wikipedia.org/wiki/Astrology_and_science)).

Reported here because an honest document reports the best counter-case. It failed
independent replication; it does not change the picture.

### 2.6.6 The 1975 scientists' statement, and why we cite it carefully

"Objections to Astrology: A Statement by 186 Leading Scientists", *The Humanist*,
September/October 1975, by Bart Bok, Lawrence Jerome and Paul Kurtz, endorsed by 186
scientists including 18–19 Nobel laureates
([text](https://www.astrology.co.uk/tests/objections.html)).

**[CONTESTED] — on its merits as an argument, not its conclusion.** Carl Sagan declined to
sign, objecting that the statement's tone was authoritarian
([ibid.](https://www.astrology.co.uk/tests/objections.html);
[discussion](https://dayology.com/2017/03/12/carl-sagan-the-good-skeptic/)). Sagan did not
think astrology worked; he thought a list of credentialled signatories was the wrong
instrument. He was right, and it is a useful discipline for us: **an appeal to 186
authorities is weaker than one census study with N > 20 million.** Cite §2.6.2–§2.6.4. Do
not lead with the manifesto.

## 2.7 The missing link nobody has mentioned: music taste ↔ personality is *also* weak

This is the finding most likely to be overlooked, and it changes how the result should be
framed.

The product implicitly assumes a chain: **sun sign → personality → music taste.** The
literature on the second link is independent of astrology, and it is not strong either.

**[STUDIES]** Schäfer, T. & Mehlhorn, C. (2017), "Can personality traits predict musical
style preferences? A meta-analysis", *Personality and Individual Differences* 116, 265–273,
[doi:10.1016/j.paid.2017.04.061](https://doi.org/10.1016/j.paid.2017.04.061). Meta-analysis
of **28 studies**. Finding: correlations were inconsistent across studies and even the
significant ones were small, **|r| between .10 and .21**; most weighted mean correlations
were near zero. Largest effects for openness to experience. The authors conclude that
**"personality is a poor predictor of musical taste"**
([summary and figures](https://research.gold.ac.uk/29269/1/PSY_Ruth_2020.pdf)).

For balance, the opposite framing: Rentfrow & Gosling (2003), "The Do Re Mi's of Everyday
Life", *Journal of Personality and Social Psychology* 84(6), 1236–1256,
[doi:10.1037/0022-3514.84.6.1236](https://doi.org/10.1037/0022-3514.84.6.1236), established
four music-preference dimensions over **3,500+ individuals** related to personality,
self-views and cognitive abilities
([record](https://researchers.dellmed.utexas.edu/en/publications/the-do-re-mis-of-everyday-life-the-structure-and-personality-corr)).
The five-factor MUSIC model replicates well
([Chee, Leung & de Vries, 2024, *Musicae Scientiae*](https://journals.sagepub.com/doi/10.1177/10298649231167488)).
And Spotify's own research reports that listening behaviour predicts Big Five traits "on par
or better than prior big data personality studies", with emotional stability and
conscientiousness best detected
([Spotify Research](https://research.atspotify.com/just-the-way-you-are-music-listening-and-personality/);
underlying paper *Social Psychological and Personality Science*,
[doi:10.1177/1948550620923228](https://doi.org/10.1177/1948550620923228)).

**Reconciling these:** the *structure* of music preference is robust and replicable, and
behavioural listening data predicts personality better than self-reported genre
preferences do. But the preference–personality correlations are small, in the .10–.21 range.

**The chain argument, which is the honest framing of our whole result:**

| link | best available estimate | source |
|---|---|---|
| sun sign → personality | **≈ 0** (null at N > 20M; null in 2,011 time-twin sets) | §2.6.2–§2.6.3 |
| personality → music taste | |r| ≈ .10–.21, "a poor predictor" | Schäfer & Mehlhorn 2017 |
| **composition** | **≈ 0** | — |

The chain is broken at the **first** link, not the second. Our 1.39× result is therefore
exactly what the literature predicts, and the honest explanation of the residual signal is
not astrology at all — it is that **playlist curators share conventions** (the §1.8 folk
mapping, plus era and genre fashions). We measured the curators, as `README.md` already
says. The literature explains why that was the only thing available to measure.

**This is a better result than a weak positive would have been.** It is a prediction the
literature made in advance, and we confirmed it with a control.

## 2.8 Why the modern astrology apps succeeded

**[STUDIES]** Co–Star: launched **17 October 2017**; founders Banu Guler, Anna Kopp, Ben
Weitzman. **>20 million downloads** as of April 2021, with growth concentrated among young
women in the U.S. Funding: **$750k** pre-seed (Female Founders Fund, early 2018);
**$5.2M** seed (Maveron, Aspect, 14W, 2019); **$15M** Series A led by Spark Capital (April
2021). The Android launch in January 2020 came off a **120,000-person waitlist**
([Wikipedia, Co–Star](https://en.wikipedia.org/wiki/Co%E2%80%93Star), citing Axios
14 Apr 2021, Business Insider, Newsweek 3 Aug 2019, Vanity Fair 22 May 2019, NYT, Gizmodo
17 Jan 2020).

Design choices the reporting credits:

1. **Real ephemeris data as the credibility layer.** The app analyses "publicly accessible
   NASA JPL data" to compute transits (ibid.). The astronomy is genuinely accurate; the
   *interpretation* is not thereby validated. **This is precisely the move §3 forbids
   overstating** — accurate inputs do not make outputs accurate, and NASA framing invites
   users to assume they do.
2. **Human-written text, algorithmically assembled.** The system maps "human-written
   snippets of text to planetary movements to display personalized content" (ibid.) —
   writers produce a library, the engine selects. Editorially controlled, cheap to operate,
   and feels personalised. Guler described changing "a lot of the content that our AI uses
   to generate the notifications"
   ([Jezebel, Maria Sherman, 2 Apr 2019](https://www.jezebel.com/why-are-co-stars-daily-notifications-so-rude-1833747335)).
3. **Blunt, unflattering tone.** The notifications became a cultural meme precisely for
   being harsh — a deliberate break from wellness-app positivity. Note this **inverts** the
   Barnum moderator in §2.1.3 (favourable statements are more readily endorsed): Co–Star
   traded endorsement for *memorability and shareability*. Guler conceded some notifications
   "are still too vague or go a bit too far" (Jezebel, ibid.).
4. **Social comparison as the retention loop.** Charts "can be compared with those of other
   users" ([Wikipedia](https://en.wikipedia.org/wiki/Co%E2%80%93Star)) — the friend-grid
   and compatibility view. This is the §2.3 mechanism operationalised, and it is the
   feature closest to what Lovify does.

**[CONTESTED] The criticism of exactly these choices.** The harsh-notification strategy drew
sustained criticism: notifications flagged as "alarming without context" — e.g. telling a
user their mental-health sector is "in crisis" with no explanation or follow-up — and users
publicly reported deleting the app over the effect on their mental health
([Jezebel](https://www.jezebel.com/why-are-co-stars-daily-notifications-so-rude-1833747335);
[Crystal Crush, "Delete Co-Star"](https://www.crystalcrushmagazine.com/magick/4-15-2021/198f0um5tl368gc7ted1oauyd7ca21)).

A widely-repeated quote has Co–Star's founder admitting to "playing with users' emotional
states" and saying "especially when you're good… we will troll you a little bit." **We could
not verify this in the Jezebel piece it is usually attributed to, and we are not citing it.**
Recorded as unverified so nobody re-imports it from a secondary summary.

**What we take from Co–Star:** mechanism (2) and feature (4) are directly transferable and
ethically clean. The NASA-data framing (1) is a credibility-transfer move we should not
copy. The harsh-tone strategy (3) worked commercially and produced documented harm, and
interacts badly with §2.5.3 — a blunt notification about your "crisis" reaches a population
disproportionately composed of people under stress.

### 2.8.1 One operational constraint

**[STUDIES]** Apple's App Review Guidelines treat fortune-telling as a saturated category:
such apps are "well established on the App Store and Apple will not accept new submissions
unless they offer a meaningfully different or improved experience" (Guideline 4.3(b)), and
**an "entertainment purposes" label does not by itself satisfy this**
([App Review Guidelines](https://developer.apple.com/app-store/review/guidelines/); see
also [TechCrunch on Struck's repeated rejections,
20 Jul 2020](https://techcrunch.com/2020/07/20/after-numerous-rejections-strucks-dating-app-for-the-co-star-crowd-hits-the-app-store/)).

Two implications. First, the music functionality has to be the substance, not the zodiac
layer — which is also the honest ordering. Second, and more important for §3: **the
entertainment disclaimer is not a compliance checkbox that discharges our obligations.**
It does not satisfy Apple, and it does not satisfy the ethical constraint either.

---

# Part 3 — The ethical line

We are deliberately building something designed to feel personally meaningful while knowing
it has no predictive validity. That is a defensible thing to build. It is also one step
away from something indefensible, and the step is short.

The distinction is not about tone, and it is not about disclaimers. **It is about claims
regarding provenance and accuracy.** Playful content with honest provenance is fine. The
identical content with implied validation is not.

## 3.1 The principle, in one line

> **Be as playful as you like about the content. Never be playful about where it comes from
> or how well it works.**

Everything below follows from that.

## 3.2 What the product may do

| allowed | why it is allowed |
|---|---|
| Frame the zodiac layer as a **playful organising device** for music discovery | True, and matches how 20% of U.S. adults ("just for fun") already engage ([Pew](https://www.pewresearch.org/religion/2025/05/21/3-in-10-americans-consult-astrology-tarot-cards-or-fortune-tellers/)) |
| Write **specific, opinionated, even unflattering** sign descriptions | Specificity is the opposite of Barnum construction (§2.1.3). A description that can be wrong about you is more honest than one that cannot. |
| Use "us Scorpios" **in-group framing** and friend comparison | The §2.3 mechanism needs no validity claim. Arbitrary categories produce real belonging ([Tajfel et al. 1971](https://en.wikipedia.org/wiki/Minimal_group_paradigm)) |
| **Document the rubric publicly** — elements, modalities, and the derivation | Turns the design into a disclosed method rather than a hidden oracle. This is the single strongest ethical move available to us. |
| Say the result is **about music, not about the person** | Already the `BIAS.md` #7 position. Keep it. |
| Let users **override** their sign or ignore the layer entirely | Keeps the feature a lens, not a verdict |
| State plainly that **sun signs do not predict music taste**, and link the survey | Costs nothing, and is the thing that makes everything else above honest |

## 3.3 What the product must not do

Each row names the specific mechanism being abused, so the constraint can be checked rather
than interpreted.

| forbidden | mechanism abused | source |
|---|---|---|
| Claim or imply the matching is **validated, accurate, or predictive** | — | The empirical record is null (§2.6). The claim would simply be false. ASA: "Claims that psychic or spiritual readings are truthful or accurate are likely to be considered problematic" ([ASA](https://www.asa.org.uk/advice-online/psychics-spiritualists-fortune-tellers-astrologers-and-clairvoyants.html)) |
| **Borrow credibility from real astronomy** — ephemeris/NASA framing presented as supporting the interpretation | Barnum moderator: *source authority* (§2.1.3) | §2.8 item 1. Accurate inputs, unvalidated outputs. |
| Dress the output in **science signalling** — accuracy percentages, confidence scores, "our model determined" | Barnum moderator: *perceived personalisation* + *source authority* | §2.1.3; §2.1.4 on marketing pseudoscience "as valid diagnostic instruments" ([ERIC ED340747](https://files.eric.ed.gov/fulltext/ED340747.pdf)) |
| Write **Barnum statements engineered to be mistaken for insight** — both-poles hedging ("at times you are X, at other times not-X"), universal needs stated as personal findings | The Barnum effect itself (§2.1) | Forer's sketch was literally assembled from an astrology book ([Wikipedia](https://en.wikipedia.org/wiki/Barnum_effect)) — the device is known and documented |
| **Target vulnerable moments** — notifications keyed to inactivity, late-night use, breakup signals, or any inferred distress | §2.5.3: engagement rises under stress; harm pathway is external locus of control | [NIMHANS review summary](https://thesouthfirst.com/health/obsession-to-astrology-use-can-lead-to-mental-health-problems-nimhans-review/) |
| Send **unsolicited negative personal assertions** ("your mental health sector is in crisis") | Same as above, compounded by no context or follow-up | Documented harm from exactly this ([Jezebel](https://www.jezebel.com/why-are-co-stars-daily-notifications-so-rude-1833747335); [Crystal Crush](https://www.crystalcrushmagazine.com/magick/4-15-2021/198f0um5tl368gc7ted1oauyd7ca21)) |
| **Advise decisions** — about relationships, compatibility-as-verdict, or anything consequential | 6% of adults rely at least a little on such content for major decisions; 21% of LGBT adults (§2.5.2) | [Pew](https://www.pewresearch.org/religion/2025/05/21/3-in-10-americans-consult-astrology-tarot-cards-or-fortune-tellers/) |
| **Promise outcomes** — better matches, improved love life, compatibility "guaranteed" | — | ASA: marketers must not "make promises they cannot keep… or exploit the credulity of naive or susceptible people" ([ASA](https://www.asa.org.uk/advice-online/psychics-spiritualists-fortune-tellers-astrologers-and-clairvoyants.html)) |
| **Monetise the ambiguity** — paywall the "real" reading, imply the free one is withheld truth | Dark pattern: "coercing, steering, or deceiving users into making decisions that, if fully informed… they might not make" (Mathur et al. 2019) | [FTC, *Bringing Dark Patterns to Light*, Sept 2022](https://www.ftc.gov/reports/bringing-dark-patterns-light); [Mathur et al. taxonomy](https://deceptive.design/book/contents/chapter-13/) |
| Treat **"for entertainment purposes only"** as discharging any of the above | — | It does not even satisfy Apple's review guideline (§2.8.1), let alone the ethical constraint. A disclaimer no one reads does not license a claim everyone sees. |

## 3.4 The two tests to apply in review

Both are cheap enough to run on any proposed copy or feature.

**1. The provenance test.** Rewrite the feature's claim in plain language with the
mechanism disclosed:

> "We sorted music into twelve buckets using a rubric we wrote, derived from the traditional
> elements and modalities. Your sign picks a bucket. Sun signs do not predict music taste —
> we checked."

If the feature still makes sense and still appeals, it is fine. **If it only works while the
user believes something stronger, it is deceptive** — and the thing it needs them to believe
is the thing we know to be false.

**2. The stress test.** Ask who receives this at their worst moment. §2.5.3 says a
disproportionate share of any astrology audience is under stress at the time of use. A
notification that reads as fun to a bored user on a Tuesday and as a verdict to a user
having a bad week is **not two different features — it is one feature, and it must be
evaluated on the second reading.**

## 3.5 Why this is not a cost

The ethical constraints and the engagement mechanisms point the same way, which is worth
stating because it is unusual:

- The strongest engagement mechanism available (§2.3, in-group identity) **requires no
  validity claim at all.** Arbitrary categories work *because* they are categories, not
  because they are correct.
- The most engaging content style (§2.4, specific and arguable) is **less** Barnum-like, not
  more. Vagueness buys endorsement and loses interest.
- The most defensible framing (a disclosed rubric) is also **the most interesting thing we
  have to talk about** — the derivation in §1.4.1 is a better story than "our model says
  73%".

The deceptive version of this product is not even the more successful one. It is just the
lazier one.

---

# Appendix A — Source quality, and what we could not verify

Stated so that the next person knows which load-bearing claims rest on thin support.

### Accessed as primary or near-primary
- Ptolemy, *Tetrabiblos* Book I (Chs. 11, 12, 17, 18) — Loeb translation, full text
  ([Penelope/UChicago](https://penelope.uchicago.edu/Thayer/E/Roman/Texts/Ptolemy/Tetrabiblos/1B*.html)).
  Carries all of §1.2, §1.3.1, §1.5.1, §1.6.
- Pew Research Center, Oct 2024 survey, N = 9,593, published May 2025 — full report
  ([Pew](https://www.pewresearch.org/religion/2025/05/21/3-in-10-americans-consult-astrology-tarot-cards-or-fortune-tellers/)).
  Strongest single source in the document.
- ASA guidance on psychic/astrology advertising
  ([ASA](https://www.asa.org.uk/advice-online/psychics-spiritualists-fortune-tellers-astrologers-and-clairvoyants.html)).
- Apple App Review Guidelines
  ([Apple](https://developer.apple.com/app-store/review/guidelines/)).
- Spotify Research post on music listening and personality
  ([Spotify](https://research.atspotify.com/just-the-way-you-are-music-listening-and-personality/)).

### Accessed via encyclopaedic secondary sources with references attached
Carlson 1985, Dean & Kelly 2003, Hartmann et al. 2006, Voas, Wunder 2003, Benski et al.
1995, Tajfel et al. 1971, Forer 1949, Meehl 1956 — all via
[Wikipedia, Astrology and science](https://en.wikipedia.org/wiki/Astrology_and_science),
[Barnum effect](https://en.wikipedia.org/wiki/Barnum_effect) and
[Minimal group paradigm](https://en.wikipedia.org/wiki/Minimal_group_paradigm), which cite
the primary literature. **Adequate for a design document; upgrade to primary sources before
any of this appears in the final report.** Priority order for that upgrade: Dean & Kelly
2003 (carries the most weight in §2.6), Wunder 2003 (carries the §2.6.4 recommendation),
Schäfer & Mehlhorn 2017 (carries §2.7).

### Could not verify — do not cite without further work
- **Forer's exact mean accuracy rating.** Three values in circulation (4.2, 4.26, 4.30).
  Primary paper not obtained. See §2.1.1. Use "approximately 4.3 out of 5 (N = 39)".
- **The Co–Star founder "playing with users' emotional states" / "troll you a little bit"
  quote.** Widely repeated, not found in the Jezebel article it is attributed to. See §2.8.
- **Carlson's precise reported success rates** and the detail of the Ertel/Currey
  reanalyses. Nature is paywalled; the critique was read only in astrology-aligned
  summary. §2.6.1 deliberately does not rest on Carlson.
- **The ancient seasonal element scheme** (spring–air, summer–fire, …). Two Wikipedia
  articles conflict. See §1.2.1. Not used.
- **Carlson's N** is given as 28 astrologers / 100+ charts by the secondary source;
  not confirmed against the paper.
- **The NIMHANS review** was read only via a news summary
  ([The South First](https://thesouthfirst.com/health/obsession-to-astrology-use-can-lead-to-mental-health-problems-nimhans-review/)).
  §2.5.3 is directionally supported but not precisely sourced — and it carries real weight
  in §3, so it is the most important upgrade on this list.

### Deliberately not used
Triplicity rulers (§1.2.2 — authorities disagree); modern planetary rulerships (§1.5.2 —
contested); polarity as an independent axis (§1.6.1 — collinear with element); sign/body and
sign/frequency correspondences (§1.7.2 — unfalsifiable); the 1975 scientists' statement as a
primary argument (§2.6.6 — an appeal to authority where we have census-scale data).

---

# Appendix B — What the rubric should take from this

Not a rubric. A list of constraints the rubric has to satisfy, each pointing at its section.

1. **Specify 7 things, not 12.** Four element profiles and three modality profiles; derive
   the twelve by composition. §1.4.1.
2. **Every sign description must be reconstructible from its two axes.** Anything that is
   not is an unsourced addition and should be visible as one in review. §1.4.1, §1.7.
3. **Expect trouble in three cells.** Libra (cardinal but "indecisive"), Aquarius (fixed but
   "revolutionary"), Cancer (cardinal but "retreating"). Decide deliberately whether to
   follow the structure or the folk expectation, and write the decision down. §1.7.1(a).
4. **Expect cardinal signs to come out mushiest.** The cardinal group is a modern merge of
   Ptolemy's two distinct equinoctial and solstitial categories. §1.3.2.
5. **Use classical rulerships or none.** §1.5.2.
6. **Do not use polarity as an input.** It is determined by element. §1.6.1.
7. **Say "sign", never "constellation".** §0.
8. **Compute sign boundaries or admit they are approximate.** §1.7.1(d).
9. **Add an astrology-knowledge/belief covariate to the survey.** Without it, a weak
   positive is uninterpretable — self-attribution and astrology predict the same data.
   §2.6.4. *This is the one item here that changes another deliverable, and it is cheap.*
10. **Expect the rubric to match folk expectations anyway**, because the folk mapping is
    already approximately element × modality. §1.8.1.
11. **Run both §3.4 tests on every piece of user-facing copy** before it ships.
