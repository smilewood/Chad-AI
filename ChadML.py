import time
import pickle
import Chad as gm
import ChadNN as nn
import numpy as np
import torch


# python multiprocessing package

def epsilonGreedy(epsilon, net, q, game, verbose=False, conNet=True):
    validMoves = game.moves()
    gameState = game.networkFormat()

    for move in validMoves:
        if game.tryMove(move).gameOver():
            return move, 10

    if np.random.uniform() < epsilon:
        # Random Move
        move = int(np.random.uniform(0, len(validMoves)))
        moveChoice = validMoves[move]
        Q = net.use(list(gameState) + list(moveChoice[0]) + list(moveChoice[1]))[0]
    else:
        # Greedy Move
        ls = list(gameState)
        gr = game.conNetFormat()

        Qs = np.array([q.get((gameState, m),
                             net.use(ls + list(m[0]) + list(m[1])) if not
                             conNet else net.cUse(gr, np.array(list(m[0]) + list(m[1])))) for m in validMoves])
        # print(Qs)
        moveChoice = validMoves[np.argmax(Qs)]
        Q = max(Qs)
    return moveChoice, Q


def reinforceAllMoves(q, moves, rho):
    for i in reversed(range(1, len(moves))):
        q[moves[i - 1]] += rho * (q[moves[i]] - q[moves[i - 1]])


def trainQ(maxGames, e, decay, rho, whiteNet, blackNet, verbose=False, whiteRandom=False, blackRandom=False):
    trainQStartTime = time.time()
    print("Training {} games, epsilon starting at {}".format(maxGames, e))
    QWhite = {}
    QBlack = {}
    outcomes = np.zeros(maxGames)
    steps = [None] * maxGames
    epsilonDecayRate = decay
    epsilon = e
    for gameNumber in range(maxGames):
        if verbose and gameNumber % (maxGames / 5) == 0:
            print("")
            print("Game:", gameNumber)

        epsilon *= epsilonDecayRate
        step = 0
        activeGame = gm.newGame()
        done = False

        whiteMoves = []
        blackMoves = []

        while not done:
            step += 1

            gameTurnWhite = activeGame.networkFormat()
            WhiteMove, Wq = epsilonGreedy(epsilon if not whiteRandom else 1, whiteNet, QWhite, activeGame)
            whiteMoves.append((gameTurnWhite, WhiteMove))
            activeGame.move(WhiteMove)
            if (gameTurnWhite, WhiteMove) not in QWhite:
                QWhite[(gameTurnWhite, WhiteMove)] = Wq

            if activeGame.gameOver():
                # White has won. This cannot happen on the first move, so BlackMove will be initialized.
                QWhite[(gameTurnWhite, WhiteMove)] = 10 - (.05 * step)
                QBlack[(gameTurnBlack, BlackMove)] = rho * (-10 - QBlack[(gameTurnBlack, BlackMove)])
                reinforceAllMoves(QWhite, whiteMoves, rho)
                reinforceAllMoves(QBlack, blackMoves, rho)
                outcomes[gameNumber] = 1
                done = True
            else:  # Game is not over, black players turn
                gameTurnBlack = activeGame.networkFormat()
                BlackMove, Bq = epsilonGreedy(epsilon if not blackRandom else 1, blackNet, QBlack, activeGame)
                blackMoves.append((gameTurnBlack, BlackMove))
                activeGame.move(BlackMove)
                if (gameTurnBlack, BlackMove) not in QBlack:
                    QBlack[(gameTurnBlack, BlackMove)] = Bq

                if activeGame.gameOver():  # Black has won
                    QBlack[(gameTurnBlack, BlackMove)] = 10 - (.05 * step)
                    QWhite[(gameTurnWhite, WhiteMove)] = rho * (-10 - QWhite[(gameTurnWhite, WhiteMove)])
                    reinforceAllMoves(QWhite, whiteMoves, rho)
                    reinforceAllMoves(QBlack, blackMoves, rho)
                    outcomes[gameNumber] = -1
                    done = True

            if step == 1000:
                outcomes[gameNumber] = 0
                if verbose:
                    print("x", end="", flush=True)
                done = True
        steps[gameNumber] = step
    wins = {g[0]: g[1] for g in np.array(np.unique(outcomes, return_counts=True)).T}
    outcomes = (wins.get(-1, 0), wins.get(0, 0), wins.get(1, 0))
    if verbose:
        print("")
        print("Training took {} seconds".format(time.time() - trainQStartTime))

    return outcomes, np.mean(steps), QWhite, QBlack, epsilon


# learningRates: epsilonDecay, rho, Net learning rate, net iterations, batch size
def trainNetworks(maxGames, hiddens, rates, verbose=False, startE=1.0, whiteRandom=False, blackRandom=False):
    print("Training start...")
    """    
    # Load log files
    try:
        with open("outcomes.res", 'rb') as file:
            Out = pickle.load(file)
    except FileNotFoundError:
        Out = []
    try:
        with open("steps.res", 'rb') as file:
            Step = pickle.load(file)
    except FileNotFoundError:
        Step = []
    """

    try:
        whiteNet = torch.load("WhiteNetwork.nn")
        blackNet = torch.load("BlackNetwork.nn")
        print("Network loaded from file")
    except FileNotFoundError:
        whiteNet = nn.nnet(148, hiddens, 1, rates[2])
        blackNet = nn.nnet(148, hiddens, 1, rates[2])
        print("No network file found, creating new network")

    # Begin training
    trainingStartTime = time.time()
    epsilon = startE
    for i in range(0, maxGames, rates[4]):

        outcomes, avgSteps, Qw, Qb, epsilon = trainQ(rates[4], epsilon, rates[0], rates[1], whiteNet, blackNet, verbose,
                                                     whiteRandom=whiteRandom, blackRandom=blackRandom)
        # Out.append(outcomes)
        # Step.append(avgSteps)
        if verbose:
            print("{} Games complete".format(str(i + rates[4])))
            print("Outcomes: White {}; Black {}; Draw {}".format(outcomes[2], outcomes[0], outcomes[1]))
            print("Games took an average of {} moves".format(avgSteps))
        if not blackRandom:
            trainNetwork(blackNet, Qb, rates[3], file="BlackNetwork.nn")
        if not whiteRandom:
            trainNetwork(whiteNet, Qw, rates[3], file="WhiteNetwork.nn")
        Qw, Qb = None, None

    if verbose:
        print("Trained {} games in {}".format(maxGames, time.time() - trainingStartTime))
        print("Final epsilon is: {}".format(epsilon))

    '''
    # Save log files
    with open("outcomes.res", 'wb') as file:
        pickle.dump(Out, file)
    with open("steps.res", 'wb') as file:
        pickle.dump(Step, file)
    '''


def trainNetwork(net, q, iterations, file=None):
    X = np.zeros([len(q), 148])
    T = np.zeros([len(q), 1])
    i = 0
    for g in q.keys():
        X[i] = list(g[0]) + list(g[1][0]) + list(g[1][1])
        T[i] = q[g]
        i += 1
    net.train(iterations, X, T)
    if file is not None:
        torch.save(net, file)


games = 10000
networkStructure = [500, 200, 200, 100, 100, 50, 50]
trainingRates = (.9999, .2, .05, 100, 500)
epsilonStart = 1.0


def bestMove(game, turn):
    try:
        if turn:  # True means black
            net = torch.load("BlackNetwork.nn")
        else:
            net = torch.load("WhiteNetwork.nn")

        game = gm.ChadGame(game, turn)
        move = epsilonGreedy(0, net, {}, game)[0]
        print(chr(move[0][0] + ord('a')) + chr(move[0][1] + ord('A')) + chr(move[1][0] + ord('a')) + chr(
            move[1][1] + ord('A')))
    except FileNotFoundError:
        print("No network file found")
        exit(1)


def train():
    trainNetworks(games, networkStructure, trainingRates, True, startE=epsilonStart, whiteRandom=True)


if __name__ == '__main__':
    # import cProfile
    # cProfile.run("train()")
    print("run")
    train()
