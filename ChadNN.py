# import gc to work on garbage collection. use gc.collect() to run it.
# then sleep for .1 sec to give it time.

import torch
import torch.nn as nn
import time
import numpy as np

Xtc = None
Ttc = None


class Torchnn(nn.Module):
    def __init__(self, n_inputs, hidden_units, n_outputs):
        super(Torchnn, self).__init__()
        self.l1 = nn.Linear(n_inputs, hidden_units[0])
        self.hidden = nn.ModuleList()
        for i in range(1, len(hidden_units)):
            self.hidden.append(nn.Linear(hidden_units[i - 1], hidden_units[i]))
        self.tanh = nn.Tanh()
        self.output = nn.Linear(hidden_units[-1], n_outputs)

    def forward(self, X):
        out = self.l1(X)
        for i in range(len(self.hidden)):
            out = self.tanh(self.hidden[i](out))
        out = self.output(out)
        return out

class TorchCnn(nn.Module):
    def __init__(self, hidden_units, n_outputs):
        super(TorchCnn, self).__init__()

        self.CL1 = nn.Sequential(
            #(12x12x1)
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            #(12x12x16)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        #(6x6x16)
        self.CL2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            #(6X6X32)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
            #(3x3x32)
        self.CL3 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=0),
            #(3x3x48)
            nn.ReLU())
        self.l1 = nn.Linear(292, hidden_units[0])
        self.hidden = nn.ModuleList()
        for i in range(1, len(hidden_units)):
            self.hidden.append(nn.Linear(hidden_units[i - 1], hidden_units[i]))
        self.tanh = nn.Tanh()
        self.output = nn.Linear(hidden_units[-1], n_outputs)
# pass the board through a convolution, then add the move at the linear layers

    def forward(self, board, move):
        X = board.unsqueeze_(0)


        out = self.CL1(X)


        out = self.CL2(out)


        #out = self.CL3(out)
        #print(out.shape)

        out = torch.flatten(out, start_dim=0, end_dim=-1)
        out = torch.cat((out, move), 0)
        out = self.l1(out)
        for i in range(len(self.hidden)):
            out = self.tanh(self.hidden[i](out))
        out = self.output(out)
        return out


class nnet:
    def __init__(self, ni, nhs, no, learning_rate):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Running on', self.device)

        self.network = Torchnn(ni, nhs, no).double()
        self.conNet = TorchCnn(nhs, no).double()

        if self.device == "cuda":
            self.network.cuda()
            self.conNet.cuda()

        self.optimizer = torch.optim.Adam(Torchnn.parameters(self.network), lr=learning_rate)
        self.cOptimizer = torch.optim.Adam(Torchnn.parameters(self.conNet), lr=learning_rate)
        self.loss_func = nn.MSELoss()

    def train(self, nIterations, X, T):
        startTime = time.time()

        Xt = torch.from_numpy(X)
        Tt = torch.from_numpy(T)

        if self.device == "cuda":
            Xt = Xt.cuda()
            Tt = Tt.cuda()

        for iteration in range(nIterations):
            # Forward pass
            outputs = self.network.forward(Xt)
            loss = self.loss_func(outputs, Tt)

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print('Training took {} seconds'.format(time.time() - startTime))

    def cTrain(self, boards, moves, targets, iterations):
        startTime = time.time()
        Xboard = torch.from_numpy(boards).double()
        Xmove = torch.from_numpy(moves).double()
        T = torch.from_numpy(targets).double()
        if self.device == "cuda":
            Xboard = Xboard.cuda()
            Xmove = Xmove.cuda()
            T = T.cuda()
        for i in range(iterations):
            outputs = self.conNet.forward(Xboard, Xmove)
            loss = self.loss_func(outputs, T)

            self.cOptimizer.zero_grad()
            loss.backward()
            self.cOptimizer.step()
        print('Training took {} seconds'.format(time.time() - startTime))


    def cUse(self, board, move):
        with torch.no_grad():
            Xboard = torch.from_numpy(board).double()
            Xmove = torch.from_numpy(move).double()
            if self.device == "cuda":
                Xboard = Xboard.cuda()
                Xmove = Xmove.cuda()

            res = self.conNet.forward(Xboard, Xmove)

            if self.device == "cuda":
                res = res.cpu()
            return list(res.numpy())

    def use(self, X):
        with torch.no_grad():
            Xt = torch.from_numpy(np.array(X)).double()
            if self.device == "cuda":
                Xt = Xt.cuda()
            res = self.network.forward(Xt)

            if self.device == "cuda":
                res = res.cpu()
            return list(res.numpy())

    def __repr__(self):
        return self.network.__repr__()


if __name__ == '__main__':


    net = nnet(1, [100], 1, .01)
    n = 10000
    X = np.linspace(0., 20.0, n).reshape((n, 1)) - 10
    T = 0.2 + 0.05 * (X + 10) + 0.4 * np.sin(X + 10) + 0.2 * np.random.normal(size=(n, 1))

    Y = net.use(X)
    print(Y)
    print(X)
    #net.train(1000, X, T)

    #Y = net.use(X)
    #print('RMSE is', np.sqrt(((T - Y) ** 2).mean()))
