"""Extractive data statistics for summarization (coverage, density, compression, novel n-grams).

Adapted from summ_eval.data_stats_metric / summ_eval.data_stats_utils (SummEval, MIT License,
https://github.com/Yale-LILY/SummEval), which in turn uses the Newsroom fragments code
(https://github.com/lil-lab/newsroom/blob/master/newsroom/analyze/fragments.py).

Vendored so HELM no longer depends on the summ-eval package, whose transitive dependencies
(blanc, pytorch-pretrained-bert, moverscore) pin torch<2.0.
"""

from collections import Counter
from typing import Any, Dict, Iterable, List, NamedTuple, Sequence, Tuple, Union


def _normalize(tokens: Sequence[Any], case: bool = False) -> List[str]:
    """Lowercases (unless case=True) and turns tokens into distinct words."""
    return [str(t) if case else str(t).lower() for t in tokens]


class _Match(NamedTuple):
    summary: int
    text: int
    length: int


class Fragments:
    def __init__(self, summary: Union[str, Sequence[Any]], text: Union[str, Sequence[Any]], case: bool = False):
        self.summary: Sequence[Any] = summary.split() if isinstance(summary, str) else summary
        self.text: Sequence[Any] = text.split() if isinstance(text, str) else text
        self._matches: List[_Match] = []
        self._match(_normalize(self.summary, case), _normalize(self.text, case))

    def overlaps(self) -> List[_Match]:
        return self._matches

    def coverage(self) -> float:
        numerator = sum(o.length for o in self.overlaps())
        denominator = len(self.summary)
        return numerator / denominator if denominator else 0

    def density(self) -> float:
        numerator = sum(o.length**2 for o in self.overlaps())
        denominator = len(self.summary)
        return numerator / denominator if denominator else 0

    def compression(self) -> float:
        return len(self.text) / len(self.summary) if len(self.summary) else 0

    def _match(self, a: List[str], b: List[str]) -> None:
        a_start = b_start = 0
        while a_start < len(a):
            best_match = None
            best_match_length = 0
            while b_start < len(b):
                if a[a_start] == b[b_start]:
                    a_end = a_start
                    b_end = b_start
                    while a_end < len(a) and b_end < len(b) and b[b_end] == a[a_end]:
                        b_end += 1
                        a_end += 1
                    length = a_end - a_start
                    if length > best_match_length:
                        best_match = _Match(a_start, b_start, length)
                        best_match_length = length
                    b_start = b_end
                else:
                    b_start += 1
            b_start = 0
            if best_match:
                if best_match_length > 0:
                    self._matches.append(best_match)
                a_start += best_match_length
            else:
                a_start += 1


def _find_ngrams(input_list: Sequence[Any], n: int) -> Iterable[Tuple[Any, ...]]:
    return zip(*[input_list[i:] for i in range(n)])


class DataStatsMetric:
    def __init__(self, n_gram: int = 3, case: bool = False, tokenize: bool = True):
        self.n_gram = n_gram
        self.case = case
        self.tokenize = tokenize
        self._en: Any = None
        if tokenize:
            import spacy

            self._en = spacy.load("en_core_web_sm")

    def evaluate_example(self, summary: Any, input_text: Any) -> Dict[str, float]:
        if self.tokenize:
            disable = ["tagger", "parser", "ner", "textcat"]
            input_text = [tok.text for tok in self._en(input_text, disable=disable)]
            summary = [tok.text for tok in self._en(summary, disable=disable)]
        fragments = Fragments(summary, input_text, case=self.case)
        score_dict: Dict[str, float] = {
            "coverage": fragments.coverage(),
            "density": fragments.density(),
            "compression": fragments.compression(),
        }
        tokenized_summary = fragments.summary
        tokenized_text = fragments.text
        score_dict["summary_length"] = len(tokenized_summary)
        for i in range(1, self.n_gram + 1):
            input_ngrams = list(_find_ngrams(tokenized_text, i))
            summ_ngrams = list(_find_ngrams(tokenized_summary, i))
            input_ngrams_set = set(input_ngrams)
            summ_ngrams_set = set(summ_ngrams)
            intersect = summ_ngrams_set.intersection(input_ngrams_set)
            if not summ_ngrams_set:
                continue
            score_dict[f"percentage_novel_{i}-gram"] = (len(summ_ngrams_set) - len(intersect)) / float(
                len(summ_ngrams_set)
            )
            ngram_counter: Counter = Counter(summ_ngrams)
            repeated = [key for key, val in ngram_counter.items() if val > 1]
            score_dict[f"percentage_repeated_{i}-gram_in_summ"] = len(repeated) / float(len(summ_ngrams_set))
        return score_dict
