from __future__ import annotations
from typing import Iterable, List, NamedTuple

import poker

# TODO: need to find a better name


class Suit(NamedTuple):
    cards: str
    name: str


class Node(NamedTuple):
    first: str
    second: str
    depth: int
    last_suit: Suit
    leaves: List[Node] = []
    who_win: bool = None


class Solver:
    def __init__(self, first: str, second: str):
        self.tree = Node(first, second, 0, Suit('', poker.skip.__name__))

    @staticmethod
    def iter_all_suit(all_cards: str, last: Suit) -> Iterable[Suit, str]:
        for name, func in poker.SUIT_FUNC:
            # if last.name != 'skip' and name != last.name:
            #     continue
            for cards, residue_cards in func(all_cards):
                if poker.is_bigger(last.name, cards, last.cards):
                    yield Suit(cards, name), residue_cards

    @classmethod
    def build_tree(cls, node: Node):
        hand = node.first if node.depth % 2 == 0 else node.second
        for suit, res_cards in cls.iter_all_suit(hand, node.last_suit):
            cards_pair = (res_cards, node.second) if node.depth % 2 == 0 else (node.first, res_cards)
            node.leaves.append(Node(*cards_pair, node.depth+1, suit))
        for i in node.leaves:
            cls.build_tree(i)

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

    def solve(self):
        self.build_tree(self.tree)
        import pickle
        with open('./tree.pic', 'rb') as f:
            pickle.dump(self.tree, f)
        self.infer_win(self.tree)
        with open('./win_tree.pic', 'rb') as f:
            pickle.dump(self.tree, f)


if __name__ == "__main__":
    s = Solver('33456789AAB', '456789T22')
    s.solve()
    # print(list(s.iter_all_suit('456', Suit('4', 'single'))))
