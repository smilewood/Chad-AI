import time
import pickle
import Chad as gm
import ChadNN as nn
import numpy as np
import torch
import copy as cp


def epsilonGreedy(epsilon, net, q, game, verbose = False):
    validMoves = game.moves()
    gameState = game.networkFormat()
    if np.random.uniform() < epsilon:
        # Random Move
        move = int(np.random.uniform(0, len(validMoves)))
        moveChoice = validMoves[move]
        Q = net.use(list(gameState) + list(moveChoice[0]) + list(moveChoice[1]))[0]
    else:
        # Greedy Move
        ls = list(gameState)
        Qs = np.array([q.get((gameState,m), net.use(ls + list(m[0]) + list(m[1]))) for m in validMoves])
        # print(Qs)
        moveChoice = validMoves[np.argmax(Qs)]
        Q = max(Qs)
    return moveChoice, Q


def trainQ(maxGames, e, decay, rho, whiteNet, blackNet, verbose=False):
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

        # A bunch of declarations to get pycharm to stop highlighting things as possibly referenced before assignment
        gameOldWhite = None
        moveOldWhite = None
        gameOldBlack = None
        moveOldBlack = None

        while not done:
            step += 1

            gameTurnWhite = activeGame.networkFormat()
            # Make White move
            WhiteMove, Wq = epsilonGreedy(epsilon, whiteNet, QWhite, activeGame)
            activeGame.move(WhiteMove)

            if (gameTurnWhite, WhiteMove) not in QWhite:
                QWhite[(gameTurnWhite, WhiteMove)] = Wq
            if step > 1:
                QBlack[(gameOldBlack, moveOldBlack)] += rho * (
                    (step * -.0005) - (QBlack[(gameTurnBlack, BlackMove)] - QBlack[(gameOldBlack, moveOldBlack)]))

            if activeGame.gameOver():
                # White has won. This cannot happen on the first move, so BlackMove will be initialized.
                QWhite[(gameTurnWhite, WhiteMove)] = 1
                QBlack[(gameTurnBlack, BlackMove)] = rho * (-1 - QBlack[(gameTurnBlack, BlackMove)])
                outcomes[gameNumber] = 1
                done = True
            else:  # Game is not over
                gameTurnBlack = activeGame.networkFormat()
                BlackMove, Bq = epsilonGreedy(epsilon, blackNet, QBlack, activeGame)
                activeGame.move(BlackMove)
                if (gameTurnBlack, BlackMove) not in QBlack:
                    QBlack[(gameTurnBlack, BlackMove)] = Bq

                if activeGame.gameOver():  # Black has won
                    QBlack[(gameTurnBlack, BlackMove)] = 1
                    QWhite[(gameTurnWhite, WhiteMove)] = rho * (-1 - QWhite[(gameTurnWhite, WhiteMove)])
                    outcomes[gameNumber] = -1
                    done = True
            if step > 1:
                QWhite[(gameOldWhite, moveOldWhite)] += rho * (
                    (step * -.0005) - (QWhite[(gameTurnWhite, WhiteMove)] - QWhite[(gameOldWhite, moveOldWhite)]))

            gameOldWhite, moveOldWhite = gameTurnWhite, WhiteMove
            gameOldBlack, moveOldBlack = gameTurnBlack, BlackMove

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
def trainNetworks(maxGames, hiddens, rates, verbose=False, networks=None, startE=1.0):
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

    # Create networks if none provided
    if networks is None:
        whiteNet = nn.nnet(148, hiddens, 1, rates[2])
        blackNet = nn.nnet(148, hiddens, 1, rates[2])
    else:
        whiteNet = networks[0]
        blackNet = networks[1]

    # Begin training
    trainingStartTime = time.time()
    epsilon = startE
    for i in range(0, maxGames, rates[4]):
        outcomes, avgSteps, Qw, Qb, epsilon = trainQ(rates[4], epsilon, rates[0], rates[1], whiteNet, blackNet, verbose)
        Out.append(outcomes)
        Step.append(avgSteps)
        if verbose:
            print("{} Games complete".format(str(i + rates[4])))
            print("Outcomes: White {}; Black {}; Draw {}".format(outcomes[2], outcomes[0], outcomes[1]))
            print("Games took an average of {} moves".format(avgSteps))
        # Split result into input and output and train black network
        Xb = np.zeros([len(Qb), 148])
        Tb = np.zeros([len(Qb), 1])
        i = 0
        for g in Qb.keys():
            Xb[i] = list(g[0]) + list(g[1][0]) + list(g[1][1])
            Tb[i] = Qb[g]
            i += 1
        if verbose:
            print("Training Black Network")
        blackNet.train(rates[3], Xb, Tb)

        # Split result into input and output and train white network
        Xw = np.zeros([len(Qw), 148])
        Tw = np.zeros([len(Qw), 1])
        i = 0
        for g in Qw.keys():
            Xw[i] = list(g[0]) + list(g[1][0]) + list(g[1][1])
            Tw[i] = Qw[g]
            i += 1
        if verbose:
            print("Training White Network")
        whiteNet.train(rates[3], Xw, Tw)

        # Save network training in case program is ended before training is complete
        torch.save(whiteNet, "WhiteNetwork.nn")
        torch.save(blackNet, "BlackNetwork.nn")
        print("Networks saved.")

    if verbose:
        print("Trained {} games in {}".format(maxGames, time.time() - trainingStartTime))
        print("Final epsilon is: {}".format(epsilon))

    # Save log files
    with open("outcomes.res", 'wb') as file:
        pickle.dump(Out, file)
    with open("steps.res", 'wb') as file:
        pickle.dump(Step, file)
    return whiteNet, blackNet


games = 2000
networkStructure = [200, 100, 100, 50, 50]
trainingRates = (.99, .3, .04, 200, 500)
epsilonStart = 1.0


def bestMove(game, turn):
    try:
        if turn:  # True means black
            net = torch.load("BlackNetwork.nn")
        else:
            net = torch.load("WhiteNetwork.nn")

        game = gm.ChadGame(game, turn)
        return epsilonGreedy(0, net, {}, game)[0]

    except FileNotFoundError:
        print("No network file found")
        exit(1)


def train():
    try:
        wNet = torch.load("WhiteNetwork.nn")
        bNet = torch.load("BlackNetwork.nn")
        print("Networks loaded from saved file")
        whiteNet, blackNet = trainNetworks(games, networkStructure, trainingRates, True, networks=(wNet, bNet),
                                           startE=epsilonStart)
    except FileNotFoundError:
        print("No network file found")
        whiteNet, blackNet = trainNetworks(games, networkStructure, trainingRates, True)
    torch.save(whiteNet, "WhiteNetwork.nn")
    torch.save(blackNet, "BlackNetwork.nn")
    print("Networks saved.")


if __name__ == '__main__':
    # import cProfile
    # cProfile.run("train()")
    train()
