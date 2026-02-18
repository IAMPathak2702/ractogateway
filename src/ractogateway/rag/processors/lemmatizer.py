"""Lemmatization processor — uses NLTK WordNetLemmatizer (lazy import).

Install with:  pip install ractogateway[rag-nlp]

Note: Lemmatization changes the surface form of text and can degrade embedding
quality for neural models (which were trained on unmodified text).  Use this
processor only when building keyword-index pipelines or when explicitly
required for your retrieval strategy.
"""

from __future__ import annotations

from typing import Any


def _require_nltk_lemmatizer() -> tuple[Any, Any]:
    try:
        import nltk  # noqa: PLC0415
        from nltk.stem import WordNetLemmatizer  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Lemmatizer requires the 'nltk' package. "
            "Install it with:  pip install ractogateway[rag-nlp]"
        ) from exc

    for resource in ("wordnet", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"):
        try:
            nltk.data.find(f"corpora/{resource}" if "tagger" not in resource else f"taggers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

    return nltk, WordNetLemmatizer


from ractogateway.rag.processors.base import BaseProcessor


def _get_wordnet_pos(tag: str) -> str:
    """Map Penn Treebank POS tag to WordNet POS tag."""
    if tag.startswith("J"):
        return "a"  # adjective
    if tag.startswith("V"):
        return "v"  # verb
    if tag.startswith("R"):
        return "r"  # adverb
    return "n"  # noun (default)


class Lemmatizer(BaseProcessor):
    """Reduce words to their base (lemma) form using NLTK WordNet.

    Parameters
    ----------
    use_pos_tagging:
        If ``True``, use POS tagging to improve lemmatization accuracy.
        Slightly slower but produces better results.
    """

    def __init__(self, use_pos_tagging: bool = True) -> None:
        self.use_pos_tagging = use_pos_tagging
        self._nltk: Any = None
        self._lemmatizer: Any = None

    def _init(self) -> None:
        if self._lemmatizer is None:
            nltk, WordNetLemmatizer = _require_nltk_lemmatizer()
            self._nltk = nltk
            self._lemmatizer = WordNetLemmatizer()

    def process(self, text: str) -> str:
        self._init()
        tokens = self._nltk.word_tokenize(text)

        if self.use_pos_tagging:
            tagged = self._nltk.pos_tag(tokens)
            lemmas = [
                self._lemmatizer.lemmatize(word, _get_wordnet_pos(pos))
                for word, pos in tagged
            ]
        else:
            lemmas = [self._lemmatizer.lemmatize(t) for t in tokens]

        return " ".join(lemmas)
