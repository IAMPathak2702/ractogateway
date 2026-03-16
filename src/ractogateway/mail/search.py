"""Search helpers for RactoMail."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone

from ._models import DateRange, FuzzyMatch, FuzzySearch

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_BOOLEAN_TOKEN_RE = re.compile(r'"[^"]+"|\(|\)|\bAND\b|\bOR\b|\bNOT\b|[^\s()]+', re.IGNORECASE)
_NEAR_RE = re.compile(
    r'(?P<left>"[^"]+"|\S+)\s+NEAR/(?P<distance>\d+)\s+(?P<right>"[^"]+"|\S+)',
    re.IGNORECASE,
)
_AMOUNT_RE = re.compile(r"(?:rs\.?\s*)?([0-9][0-9,]*(?:\.[0-9]{1,2})?)", re.IGNORECASE)
_HOLIDAYS: dict[str, dict[int, date]] = {
    "diwali": {
        2023: date(2023, 11, 12),
        2024: date(2024, 11, 1),
        2025: date(2025, 10, 20),
        2026: date(2026, 11, 8),
    },
    "holi": {
        2023: date(2023, 3, 8),
        2024: date(2024, 3, 25),
        2025: date(2025, 3, 14),
        2026: date(2026, 3, 3),
    },
}


def normalize_search_text(text: str) -> str:
    """Normalize text for approximate matching."""
    lowered = text.casefold()
    collapsed = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", collapsed).strip()


def tokenize(text: str) -> list[str]:
    """Tokenize into lowercase alphanumeric terms."""
    return [token.casefold() for token in _TOKEN_RE.findall(text)]


def _normalize_fuzzy_token(token: str, config: FuzzySearch) -> str:
    base = normalize_search_text(token).replace(" ", "")
    if not config.indian_names and not config.transliteration_fuzzy:
        return base
    return re.sub(r"(.)\1+", r"\1", base)


def levenshtein_distance(left: str, right: str) -> int:
    """Compute Levenshtein edit distance."""
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            insertion = current[right_index - 1] + 1
            deletion = previous[right_index] + 1
            substitution = previous[right_index - 1] + (left_char != right_char)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def find_fuzzy_matches(
    query_terms: list[str],
    corpus_terms: list[str],
    config: FuzzySearch,
) -> list[FuzzyMatch]:
    """Find fuzzy matches from query terms against a message corpus."""
    if not config.enabled:
        return []

    corpus_counter = Counter(_normalize_fuzzy_token(term, config) for term in corpus_terms)
    matches: list[FuzzyMatch] = []
    for term in query_terms:
        normalized_term = _normalize_fuzzy_token(term, config)
        if not normalized_term:
            continue
        best_match: tuple[str, int] | None = None
        for corpus_term in corpus_counter:
            distance = levenshtein_distance(normalized_term, corpus_term)
            if distance > config.max_distance:
                continue
            if best_match is None or distance < best_match[1]:
                best_match = (corpus_term, distance)
        if best_match is not None:
            matches.append(
                FuzzyMatch(
                    original=term,
                    matched_to=best_match[0],
                    distance=best_match[1],
                    match_count=corpus_counter[best_match[0]],
                )
            )
    return matches


def extract_amounts(text: str) -> list[float]:
    """Extract approximate amounts from body or attachment text."""
    amounts: list[float] = []
    for raw_value in _AMOUNT_RE.findall(text):
        cleaned = raw_value.replace(",", "")
        try:
            amounts.append(float(cleaned))
        except ValueError:
            continue
    return amounts


def _strip_boolean_term(term: str) -> str:
    cleaned = term.strip().strip('"')
    if ":" in cleaned:
        cleaned = cleaned.split(":", maxsplit=1)[1]
    return cleaned.replace("*", "")


def _token_sequence(text: str) -> list[str]:
    return tokenize(text)


def _contains_phrase(text: str, phrase: str) -> bool:
    return normalize_search_text(phrase) in normalize_search_text(text)


def _near_match(text: str, left: str, right: str, distance: int) -> bool:
    sequence = _token_sequence(text)
    left_terms = tokenize(left)
    right_terms = tokenize(right)
    if not left_terms or not right_terms:
        return False
    left_term = left_terms[0]
    right_term = right_terms[0]
    left_positions = [index for index, term in enumerate(sequence) if term == left_term]
    right_positions = [index for index, term in enumerate(sequence) if term == right_term]
    return any(abs(lp - rp) <= distance for lp in left_positions for rp in right_positions)


@dataclass
class _BooleanParser:
    tokens: list[str]
    index: int = 0

    def parse(self, text: str) -> bool:
        result = self._parse_expression(text)
        return result

    def _parse_expression(self, text: str) -> bool:
        value = self._parse_term(text)
        while self._peek_upper() == "OR":
            self.index += 1
            next_value = self._parse_term(text)
            value = value or next_value
        return value

    def _parse_term(self, text: str) -> bool:
        value = self._parse_factor(text)
        while self._peek_upper() == "AND":
            self.index += 1
            next_value = self._parse_factor(text)
            value = value and next_value
        return value

    def _parse_factor(self, text: str) -> bool:
        if self._peek_upper() == "NOT":
            self.index += 1
            return not self._parse_factor(text)
        return self._parse_atom(text)

    def _parse_atom(self, text: str) -> bool:
        piece = self._peek()
        if piece is None:
            return False
        if piece == "(":
            self.index += 1
            value = self._parse_expression(text)
            if self._peek() == ")":
                self.index += 1
            return value
        self.index += 1
        return _term_matches(text, piece)

    def _peek(self) -> str | None:
        if self.index >= len(self.tokens):
            return None
        return self.tokens[self.index]

    def _peek_upper(self) -> str | None:
        token = self._peek()
        return token.upper() if token is not None else None


def _term_matches(text: str, token: str) -> bool:
    term = _strip_boolean_term(token)
    if not term:
        return False
    if " " in term:
        return _contains_phrase(text, term)
    return term.casefold() in tokenize(text)


def evaluate_boolean_query(text: str, query: str) -> bool:
    """Evaluate a lightweight boolean query against text."""
    near_match = _NEAR_RE.fullmatch(query.strip())
    if near_match is not None:
        return _near_match(
            text,
            near_match.group("left"),
            near_match.group("right"),
            int(near_match.group("distance")),
        )

    tokens = _BOOLEAN_TOKEN_RE.findall(query)
    if not tokens:
        return False

    parser = _BooleanParser(tokens=tokens)
    return parser.parse(text)


class IndianCalendar:
    """Resolve a few common Indian date references into concrete ranges."""

    def __init__(self, *, fiscal_year_start: str = "April") -> None:
        self.fiscal_year_start = fiscal_year_start

    def resolve(self, reference: str, *, now: datetime | None = None) -> DateRange | None:
        """Resolve a natural-language Indian business date expression."""
        current = now or datetime.now(tz=timezone.utc)
        lowered = reference.casefold().strip()
        resolved: DateRange | None = None

        fy_match = re.search(r"\bq([1-4])\s*fy\s*(\d{4})\b", lowered)
        if fy_match is not None:
            quarter = int(fy_match.group(1))
            fiscal_year_end = int(fy_match.group(2))
            fiscal_year_start = fiscal_year_end - 1
            quarter_starts = {
                1: date(fiscal_year_start, 4, 1),
                2: date(fiscal_year_start, 7, 1),
                3: date(fiscal_year_start, 10, 1),
                4: date(fiscal_year_end, 1, 1),
            }
            quarter_ends = {
                1: date(fiscal_year_start, 6, 30),
                2: date(fiscal_year_start, 9, 30),
                3: date(fiscal_year_start, 12, 31),
                4: date(fiscal_year_end, 3, 31),
            }
            resolved = _date_range(
                f"Q{quarter} FY{fiscal_year_end}",
                quarter_starts[quarter],
                quarter_ends[quarter],
            )
        elif "week before diwali" in lowered:
            year = _extract_year(lowered, current.year)
            diwali = _HOLIDAYS["diwali"].get(year)
            if diwali is None:
                return None
            resolved = _date_range(
                f"week before Diwali {year}",
                diwali - timedelta(days=13),
                diwali - timedelta(days=7),
            )
        elif "last diwali" in lowered:
            year = current.year
            diwali = _HOLIDAYS["diwali"].get(year)
            if diwali is not None and datetime.combine(
                diwali,
                time.max,
                tzinfo=timezone.utc,
            ) > current:
                year -= 1
            diwali = _HOLIDAYS["diwali"].get(year)
            if diwali is None:
                return None
            resolved = _date_range(f"Diwali {year}", diwali, diwali)
        elif "after holi" in lowered and "financial year end" in lowered:
            year = _extract_year(lowered, current.year)
            holi = _HOLIDAYS["holi"].get(year)
            if holi is None:
                return None
            resolved = _date_range(
                f"after Holi before FY end {year}",
                holi + timedelta(days=1),
                date(year, 3, 31),
            )
        elif "before budget" in lowered:
            year = _extract_year(lowered, current.year)
            resolved = _date_range(
                f"before budget {year}",
                date(year, 1, 1),
                date(year, 1, 31),
            )
        elif "gst filing week" in lowered:
            start = date(current.year, current.month, 18)
            end = date(current.year, current.month, 21)
            resolved = _date_range("GST filing week", start, end)
        elif "year end rush" in lowered:
            resolved = _date_range(
                f"year end rush {current.year}",
                date(current.year, 3, 15),
                date(current.year, 3, 31),
            )
        elif re.search(r"\blast\s+week\b", lowered):
            start_of_week = date(current.year, current.month, current.day) - timedelta(
                days=current.weekday() + 7
            )
            resolved = _date_range(
                "last week",
                start_of_week,
                start_of_week + timedelta(days=6),
            )
        elif re.search(r"\bthis\s+week\b", lowered):
            start_of_week = date(current.year, current.month, current.day) - timedelta(
                days=current.weekday()
            )
            resolved = _date_range(
                "this week",
                start_of_week,
                start_of_week + timedelta(days=6),
            )
        elif re.search(r"\blast\s+month\b", lowered):
            month = current.month - 1 if current.month > 1 else 12
            year = current.year if current.month > 1 else current.year - 1
            import calendar  # noqa: PLC0415
            last_day = calendar.monthrange(year, month)[1]
            resolved = _date_range(
                "last month",
                date(year, month, 1),
                date(year, month, last_day),
            )
        elif re.search(r"\bthis\s+month\b", lowered):
            import calendar  # noqa: PLC0415
            last_day = calendar.monthrange(current.year, current.month)[1]
            resolved = _date_range(
                "this month",
                date(current.year, current.month, 1),
                date(current.year, current.month, last_day),
            )

        return resolved


def _extract_year(text: str, fallback: int) -> int:
    year_match = re.search(r"\b(20\d{2})\b", text)
    if year_match is None:
        return fallback
    return int(year_match.group(1))


def _date_range(label: str, start_date: date, end_date: date) -> DateRange:
    return DateRange(
        label=label,
        start=datetime.combine(start_date, time.min, tzinfo=timezone.utc),
        end=datetime.combine(end_date, time.max, tzinfo=timezone.utc),
    )


# Indian currency / number normalization helpers

_LAKH_RE = re.compile(r"([0-9,.]+)\s*(?:lakh|lac)", re.IGNORECASE)
_CRORE_RE = re.compile(r"([0-9,.]+)\s*crore", re.IGNORECASE)
_RUPEE_SYMBOLS = re.compile(r"(?:₹|rs\.?|inr)\s*", re.IGNORECASE)

_HINDI_DIGIT_MAP = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩",
    "0123456789",
)
_DEVANAGARI_DIGIT_MAP = str.maketrans(
    "०१२३४५६७८९",
    "0123456789",
)


def normalize_indian_currency(text: str) -> str:
    """Normalize Indian currency expressions to plain numbers."""
    result = _RUPEE_SYMBOLS.sub("", text)
    result = _LAKH_RE.sub(
        lambda m: str(float(m.group(1).replace(",", "")) * 100_000),
        result,
    )
    result = _CRORE_RE.sub(
        lambda m: str(float(m.group(1).replace(",", "")) * 10_000_000),
        result,
    )
    return result


def normalize_devanagari_digits(text: str) -> str:
    """Replace Devanagari/Arabic-Indic digit characters with ASCII digits."""
    return text.translate(_DEVANAGARI_DIGIT_MAP).translate(_HINDI_DIGIT_MAP)


def detect_language_mixing(text: str) -> bool:
    """Detect if the text mixes Devanagari script with Latin characters."""
    has_devanagari = bool(re.search(r"[\u0900-\u097F]", text))
    has_latin = bool(re.search(r"[A-Za-z]", text))
    return has_devanagari and has_latin
