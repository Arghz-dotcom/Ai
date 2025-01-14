import random
import time
import math
from copy import deepcopy

from MtcsConnect4.connectState import ConnectState
from meta import GameMeta, MCTSMeta


class Node:
    def __init__(self, move: int|None, parent: 'Node'):
        self.move: int|None = move
        self.parent: 'Node' = parent
        self.N: int = 0
        """ N simulations in that given state """
        self.Q: int = 0
        """ Q amount of wins the agent has out of the N simulations in that given state """ 
        self.children: dict[int, 'Node']  = {}
        self.outcome: int = GameMeta.PLAYERS['none']

    def add_children(self, children: list['Node']) -> None:
        """ Add children to node """
        for child in children:
            self.children[child.move] = child

    def has_children(self) -> bool:
        return len(self.children) > 0

    def uct_value(self, explore: float = MCTSMeta.EXPLORATION) -> float:
        """ Upper Confidence Bounds for Trees """

        if self.N == 0:
            # we prioritize nodes that are not explored, uct is infinite if node is not yet explored
            return 0 if explore == 0 else GameMeta.INF
        else:
            return self.Q / self.N + explore * math.sqrt(math.log(self.parent.N) / self.N)
        


class MCTS:
    def __init__(self, state=ConnectState()):
        self.root_state: ConnectState = deepcopy(state)
        self.root: Node = Node(None, None)
        self.run_time: float = 0
        """ Elapsed time for computation in seconds """
        self.node_count: int = 0
        self.num_rollouts: int = 0

    def select_node(self) -> tuple[Node, ConnectState]:
        """
        select a node in the tree to expand and simulate upon.
        We select the leaf node with the maximum value that we define.
        If the leaf node has not been explored upon, we will select that node; otherwise, 
        we will add all possible actions as its children into the tree and randomly select one node.
        """

        node = self.root
        state = deepcopy(self.root_state)

        while node.has_children():
            # select node with max uct
            children = node.children.values()
            max_value: float = max(children, key=lambda n: n.uct_value()).uct_value()
            # select nodes with the highest UCT value
            max_nodes: list[Node] = [n for n in children if n.uct_value() == max_value]

            # randomly select on to expand upon
            node: Node = random.choice(max_nodes)
            state.move(node.move)

            if node.N == 0: # select leaf node if it has not been explored upon otherwise we go deeper
                return node, state

        # Here leaf node, not yet discovered children
        if self.can_expand(node, state): # determines if the state is a terminal state (game over)
            node: Node = random.choice(list(node.children.values()))
            state.move(node.move)

        return node, state

    def can_expand(self, parent: Node, state: ConnectState) -> bool:
        """ After the given node is selected, we want to add all possible actions as children of the selected node """

        if state.game_over():
            return False

        children = [Node(move, parent) for move in state.get_legal_moves()]
        parent.add_children(children)

        return True

    def roll_out(self, state: ConnectState) -> int:
        """
        simulating through a random game starting from the given state in the input
        Return: winner of the simulated game to update the probability of each node
        """

        while not state.game_over():
            state.move(random.choice(state.get_legal_moves()))

        return state.get_outcome()

    def back_propagate(self, node: Node, turn: int, outcome: int) -> None:
        """
        propagate the winner of the simulated game through all of the ancestors of the selected node.
        We go through all of its ancestors because the selected node's state came from the parents
        and contributes to the overall "goodness" of the parent states
        """

        # For the current player, not the next player
        reward = 0 if outcome == turn else 1

        while node is not None:
            node.N += 1
            node.Q += reward
            node = node.parent
            if outcome == GameMeta.OUTCOMES['draw']: # we count it as a loss for every state
                reward = 0
            else:
                reward = 1 - reward # alternates between 0 and 1 because each alternate depth represents different player turns

    def search(self, time_limit: int):
        """
        Combining all the phases in order, we will select the node to simulate,
        then perform rollout on that node and then backpropagate the results onto its parent nodes.
        We will repeat these steps until time_limit is reached

        args:
          time_limit: duration of calculation in seconds
        """

        start_time = time.process_time()

        self.num_rollouts = 0
        while time.process_time() - start_time < time_limit:
            node, state = self.select_node()
            outcome = self.roll_out(state)
            self.back_propagate(node, state.to_play, outcome)
            self.num_rollouts += 1 # for calculating statistics

        self.run_time = time.process_time() - start_time

    def best_move(self) -> int:
        """
        For each node, we choose the action that leads to the state with the most N.
        It is important to note that we do not choose the highest Q/N
        because it could come from a relatively unexplored node.
        """

        if self.root_state.game_over():
            return -1

        # choose node with best N
        max_value = max(self.root.children.values(), key=lambda n: n.N).N
        max_nodes = [n for n in self.root.children.values() if n.N == max_value]
        best_child = random.choice(max_nodes)

        return best_child.move

    def move(self, move: int):
        if move in self.root.children:
            self.root_state.move(move)
            self.root = self.root.children[move]
            return

        self.root_state.move(move)
        self.root = Node(None, None)

    def statistics(self) -> tuple[int, float]:
        return self.num_rollouts, self.run_time
