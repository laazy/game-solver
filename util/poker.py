from collections import Counter
from functools import reduce
from typing import Iterable, Tuple

POKERS_KINDS = '23456789TJQKASB'


def pairn(pokers: str, n) -> Iterable[Tuple[str, str]]:
    pokers = ''.join(sorted(pokers))
    for p, c in Counter(pokers).items():
        if c >= n:
            yield p*n, pokers.replace(p, '', n)


def pair2(pokers: str) -> Iterable[Tuple[str, str]]:
    yield from pairn(pokers, 2)


def pair3(pokers: str) -> Iterable[Tuple[str, str]]:
    yield from pairn(pokers, 3)


def pair4(pokers: str) -> Iterable[Tuple[str, str]]:
    yield from pairn(pokers, 4)


def straight(pokers: str) -> Iterable[Tuple[str, str]]:
    above_number = 4
    allowing = '23456789TJQK'
    kinds = ''.join(sorted(set(pokers)))
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


def single(pokers: str) -> Iterable[Tuple[str, str]]:
    for c in pokers:
        yield c, pokers.replace(c, '', 1)


def skip(pokers: str) -> Iterable[Tuple[str, str]]:
    yield '', pokers
