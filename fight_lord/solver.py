from __future__ import annotations
from operator import methodcaller
from typing import Iterable, List, NamedTuple
from dataclasses import dataclass, field

import poker

# TODO: need to find a better name


class Suit(NamedTuple):
    cards: str
    name: str


@dataclass
class Node:
    first: str
    second: str
    depth: int
    last_suit: Suit
    leaves: List[Node] = field(default_factory=list)
    who_win: bool = None


class Solver:
    RULE = {
        i: [(i, j), ('skip', poker.skip)] for i, j in poker.SUIT_FUNC
    }
    RULE['skip'] = [i for i in poker.SUIT_FUNC if i[0] != 'skip']

    def __init__(self, first: str, second: str):
        self.tree = Node(first, second, 0, Suit('', poker.skip.__name__))
        self.counter = 10000

    @classmethod
    def iter_all_suit(cls, all_cards: str, last: Suit) -> Iterable[Suit, str]:
        for name, func in cls.RULE[last.name]:
            for cards, residue_cards in func(all_cards):
                if not (cards and last.cards) or poker.is_bigger(last.name, cards, last.cards):
                    yield Suit(cards, name), residue_cards

    def build_tree(self, node: Node):
        if self.counter <= 0:
            return
        self.counter -= 1
        hand, rival = node.first, node.second
        if node.depth % 2 != 0:
            hand, rival = rival, hand
        if not rival:
            return
        for suit, res_cards in self.iter_all_suit(hand, node.last_suit):
            if node.depth % 2 == 0:
                leaf = Node(res_cards, rival, node.depth+1, suit)
            else:
                leaf = Node(rival, res_cards, node.depth+1, suit)
            node.leaves.append(leaf)
        for idx, i in enumerate(node.leaves):
            if node.depth < 4:
                print(f'run {idx+1}/{len(node.leaves)}, in depth: {node.depth}')                
            self.build_tree(i)

    @staticmethod
    def who_win(node: Node) -> bool:
        return {
            (True, False): 'first',
            (False, True): 'second',
            (False, False): None,
        }[node.first == '', node.second == '']

    @classmethod
    def infer_win(cls, node: Node):
        res = cls.who_win(node)
        if res is not None:
            node.who_win = res
            return
        for i in node.leaves:
            cls.infer_win(i)
        winner = {i.who_win for i in node.leaves}
        if len(winner) >= 2:
            node.who_win = 'no idea'
        else:
            node.who_win = winner.pop()

    @classmethod
    def write_tree(cls, node: Node) -> Iterable[List[str]]:
        if not node.leaves:
            yield ['']
        for i in node.leaves:
            for j in cls.write_tree(i):
                yield [node.last_suit.cards] + j

    def solve(self):
        self.build_tree(self.tree)
        import pickle
        with open('./tree.pic', 'wb') as f:
            pickle.dump(self.tree, f)
        with open('tree.txt', 'w') as f:
            for i in self.write_tree(self.tree):
                f.write(','.join(i[1:]) + '\n')
        self.infer_win(self.tree)
        with open('./win_tree.pic', 'wb') as f:
            pickle.dump(self.tree, f)


if __name__ == "__main__":
    Solver('33456789AAB', '456789T22').solve()
