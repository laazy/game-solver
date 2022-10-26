from collections import Counter
from functools import reduce, wraps
from typing import Dict, Iterable, Tuple, Callable

POKERS_KINDS = '23456789TJQKASB'

CardPlaySet = Iterable[Tuple[str, str]]
SuitFunc = Callable[[str], CardPlaySet]
CompareFunc = Callable[[str, str], bool]


SUIT_FUNC: Dict[str, SuitFunc] = {}
COMPARE_FUNC: Dict[str, CompareFunc] = {}


def suit(func):
    SUIT_FUNC[func.__name__] = func
    COMPARE_FUNC[func.__name__] = _default_compare
    return func


def compare(func):
    COMPARE_FUNC[func.__name__.replace('_compare', '')] = func
    return func


def _pair_n(pokers: str, n) -> CardPlaySet:
    pokers = ''.join(sorted(pokers))
    for p, c in Counter(pokers).items():
        if c >= n:
            yield p*n, pokers.replace(p, '', n)


def _default_compare(p1: str, p2: str) -> bool:
    return POKERS_KINDS.find(p1) > POKERS_KINDS.find(p2)


@suit
def pair2(pokers: str) -> CardPlaySet:
    yield from _pair_n(pokers, 2)


@suit
def pair3(pokers: str) -> CardPlaySet:
    yield from _pair_n(pokers, 3)


@suit
def pair4(pokers: str) -> CardPlaySet:
    yield from _pair_n(pokers, 4)


@suit
def straight(pokers: str) -> CardPlaySet:
    above_number = 4
    allowing = '3456789TJQK'
    kinds = ''.join(sorted(set(pokers), key=allowing.find))
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


@suit
def single(pokers: str) -> CardPlaySet:
    for c in pokers:
        yield c, pokers.replace(c, '', 1)


@suit
def skip(pokers: str) -> CardPlaySet:
    yield '', pokers


@compare
def skip_compare(p1: str, p2: str) -> bool:
    return p2 != ''
