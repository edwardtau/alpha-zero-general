import unittest

from Coach import Coach

from othello.OthelloGame import OthelloGame
from othello.pytorch.NNet import NNetWrapper as OthelloPytorchNNet
from othello.tensorflow.NNet import NNetWrapper as OthelloTensorflowNNet
from othello.keras.NNet import NNetWrapper as OthelloKerasNNet

from tictactoe.TicTacToeGame import TicTacToeGame
from tictactoe.keras.NNet import NNetWrapper as TicTacToeKerasNNet

from connect4.Connect4Game import Connect4Game
from connect4.tensorflow.NNet import NNetWrapper as Connect4TensorflowNNet

from gobang.GobangGame import GobangGame
from gobang.keras.NNet import NNetWrapper as GobangKerasNNet
from gobang.tensorflow.NNet import NNetWrapper as GobangTensorflowNNet

from rts.RTSGame import RTSGame
from rts.keras.NNet import NNetWrapper as RTSKerasNNet

from utils import *

args = dotdict({
    'numIters': 1,
    'numEps': 2,                # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 3,          # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('', ''),
    'numItersForTrainExamplesHistory': 20,

})


class TestAllTrainingGames(unittest.TestCase):

    @staticmethod
    def execute_game_training(game, neural_net):
        nnet = neural_net(game)

        if args.load_model:
            nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

        c = Coach(game, nnet, args)
        if args.load_model:
            print("Load trainExamples from file")
            c.loadTrainExamples()
        c.learn()

    def test_training_othello_pytorch(self):
        self.execute_game_training(OthelloGame(6), OthelloPytorchNNet)

    def test_training_othello_tensorflow(self):
        self.execute_game_training(OthelloGame(6), OthelloTensorflowNNet)

    def test_training_othello_keras(self):
        self.execute_game_training(OthelloGame(6), OthelloKerasNNet)

    def test_training_tictactoe_keras(self):
        self.execute_game_training(TicTacToeGame(), TicTacToeKerasNNet)

    def test_training_connect4_tensorflow(self):
        self.execute_game_training(Connect4Game(), Connect4TensorflowNNet)

    def test_training_gobang_tensorflow(self):
        self.execute_game_training(GobangGame(), GobangTensorflowNNet)

    def test_training_gobang_keras(self):
        self.execute_game_training(GobangGame(), GobangKerasNNet)

    def test_training_rts_keras(self):
        self.execute_game_training(RTSGame(), RTSKerasNNet)


if __name__ == '__main__':
    unittest.main()

