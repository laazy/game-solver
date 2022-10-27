from collections import Counter
from functools import reduce, wraps
from typing import Dict, Iterable, List, Tuple, Callable

POKERS_KINDS = '3456789TJQKA2SB'


CardPlaySet = Iterable[Tuple[str, str]]
SuitFunc = Callable[[str], CardPlaySet]
CompareFunc = Callable[[str, str], bool]


SUIT_FUNC: List[Tuple[str, SuitFunc]] = []
COMPARE_FUNC: Dict[str, CompareFunc] = {}


def _sorted_poker(pokers: str):
    return sorted(pokers, key=POKERS_KINDS.find)


def _suit(func):
    SUIT_FUNC.append((func.__name__, func))
    COMPARE_FUNC[func.__name__] = _default_compare
    return func


def _compare(func):
    COMPARE_FUNC[func.__name__.replace('_compare', '')] = func
    return func


def is_bigger(name: str, p1: str, p2: str):
    return COMPARE_FUNC[name](p1, p2)


def _pair_n(pokers: str, n) -> CardPlaySet:
    pokers = ''.join(_sorted_poker(pokers))
    for p, c in Counter(pokers).items():
        if c >= n:
            yield p*n, pokers.replace(p, '', n)


def _default_compare(p1: str, p2: str) -> bool:
    return POKERS_KINDS.find(p1[0]) > POKERS_KINDS.find(p2[0])


@_suit
def pair2(pokers: str) -> CardPlaySet:
    yield from _pair_n(pokers, 2)


@_suit
def pair3(pokers: str) -> CardPlaySet:
    yield from _pair_n(pokers, 3)


@_suit
def pair4(pokers: str) -> CardPlaySet:
    yield from _pair_n(pokers, 4)


@_suit
def straight(pokers: str) -> CardPlaySet:
    above_number = 4
    allowing = '3456789TJQK'
    kinds = ''.join(_sorted_poker(set(pokers)))
    if len(kinds) <= above_number:
        return
    for begin in range(len(kinds) - above_number):
        ai = allowing.find(kinds[begin])
        for end in range(begin+above_number, len(kinds)):
            aj = allowing[end-begin+ai]
            if aj != kinds[end]:
                break
            ans = kinds[begin: end+1]
            yield ans, reduce(lambda x, y: x.replace(y, '', 1), ans, pokers)


@_suit
def single(pokers: str) -> CardPlaySet:
    for c in _sorted_poker(set(pokers)):
        yield c, pokers.replace(c, '', 1)


@_suit
def skip(pokers: str) -> CardPlaySet:
    yield '', pokers


@_compare
def skip_compare(p1: str, p2: str) -> bool:
    return p1 != ''


if __name__ == "__main__":
    print(list(straight('33456789AAB')))
