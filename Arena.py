import numpy as np
from Game import Game
from pytorch_classification.utils import Bar, AverageMeter
import time

class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """
    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2) or 0 for stalemate
        """
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0
        while self.game.getGameStatus(board, curPlayer) == Game.IN_PROGRESS:
            it+=1
            if verbose:
                assert(self.display)
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(board)
            action = players[curPlayer+1](self.game.getCanonicalForm(board, curPlayer))

            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)

            if valids[action] == 0:
                print(action)
                assert valids[action] > 0
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
        if verbose:
            assert(self.display)
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameStatus(board, 1)))
            self.display(board)
        return self.game.getGameStatus(board, 1)

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  stalemate
        """
        eps_time = AverageMeter()
        bar = Bar('Arena.playGames', max=num)
        end = time.time()
        eps = 0
        maxeps = int(num)

        template = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:} | Score: {one}/{two}/{draw}'

        player_one_first = True
        scoring = {self.game.WON_PLAYER1: 0, self.game.WON_PLAYER2: 0, self.game.STALEMATE: 0}

        for _ in range(maxeps):
            result = self.playGame(verbose=verbose)

            if result != self.game.STALEMATE and not player_one_first:
                result *= -1 # Flip if player one is not going first.
            scoring[result] += 1

            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix  = template.format(eps=eps+1, maxeps=maxeps, et=eps_time.avg, total=bar.elapsed_td, eta=bar.eta_td,
                                          one=scoring[self.game.WON_PLAYER1], two=scoring[self.game.WON_PLAYER2],
                                          draw=scoring[self.game.STALEMATE])
            bar.next()

            # Swap players' order (toggling who plays first).
            player_one_first = not player_one_first
            self.player1, self.player2 = self.player2, self.player1
        bar.finish()

        return tuple(scoring.values())
