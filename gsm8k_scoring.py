"""Versioned offline numeric-answer scoring. No model or dataset dependencies."""

from dataclasses import dataclass
from fractions import Fraction
import re

PROTOCOL_VERSION = 'numeric-v2'
DEFAULT_TOLERANCE = Fraction('0.0001')
_NUMBER = r'[-+]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
_CANDIDATE = re.compile(rf'(?<![\w.,/]){_NUMBER}(?:\s*/\s*{_NUMBER})?(?![\w.,/])')
_VALID_NUMBER = re.compile(r'[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\Z')
_ANSWER = re.compile(r'<answer\s*>(.*?)</answer\s*>', re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class Extraction:
    value: Fraction | None
    source: str
    token: str | None = None
    error: str | None = None


def extract_answer(text: str | None) -> Extraction:
    """Last complete XML answer, otherwise last #### suffix, otherwise last token.

    An explicit answer block with no valid answer fails rather than borrowing a
    number from reasoning. Only completion text should be supplied, never prompts.
    """
    if not isinstance(text, str) or not text.strip():
        return Extraction(None, 'missing', error='empty_completion')
    text = text.replace('\u2212', '-')
    blocks = _ANSWER.findall(text)
    if blocks:
        segment, source = blocks[-1], 'xml'
    elif '####' in text:
        segment, source = text.rsplit('####', 1)[-1], 'hash'
    else:
        if '<answer' in text.lower() or '</answer' in text.lower():
            return Extraction(None, 'xml', error='unclosed_or_malformed_answer_tag')
        segment, source = text, 'fallback'
    matches = list(_CANDIDATE.finditer(segment))
    if not matches:
        return Extraction(None, source, error='no_numeric_answer')
    match = matches[-1]
    token = match.group()
    # Mixed fractions are ambiguous under last-token extraction; fail explicitly.
    if '/' in token and re.search(r'\d\s+$', segment[:match.start()]):
        return Extraction(None, source, token, 'unsupported_mixed_fraction')
    parts = [part.strip() for part in token.split('/')]
    if not all(_VALID_NUMBER.fullmatch(part) for part in parts):
        return Extraction(None, source, token, 'invalid_numeric_format')
    # Limit pathological exponents/long numeric strings before exact arithmetic.
    if len(token) > 200 or any(abs(int(m.group(1))) > 1000 for m in re.finditer(r'[eE]([-+]?\d+)', token)):
        return Extraction(None, source, token, 'numeric_size_limit')
    try:
        value = Fraction(parts[0].replace(',', ''))
        if len(parts) == 2:
            value /= Fraction(parts[1].replace(',', ''))
    except (ValueError, ZeroDivisionError):
        return Extraction(None, source, token, 'invalid_fraction')
    return Extraction(value, source, token)


def score_answer(prediction: str | None, gold: str, tolerance=DEFAULT_TOLERANCE) -> dict:
    """Use strict absolute difference < tolerance, matching the final notebook rule."""
    tolerance = Fraction(str(tolerance)) if not isinstance(tolerance, Fraction) else tolerance
    if tolerance <= 0:
        raise ValueError('tolerance must be positive')
    predicted = extract_answer(prediction)
    expected = extract_answer(gold)
    if expected.value is None:
        raise ValueError(f'Unparseable gold answer: {expected.error}')
    return {
        'correct': predicted.value is not None and abs(predicted.value - expected.value) < tolerance,
        'prediction_value': str(predicted.value) if predicted.value is not None else None,
        'gold_value': str(expected.value),
        'extraction_source': predicted.source,
        'extraction_error': predicted.error,
        'extracted_token': predicted.token,
    }
