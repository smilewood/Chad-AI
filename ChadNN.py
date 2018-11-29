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


class nnet:
    def __init__(self, ni, nhs, no, learning_rate):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Running on', self.device)

        self.network = Torchnn(ni, nhs, no).double()

        if self.device == "cuda":
            self.network.cuda()

        self.optimizer = torch.optim.Adam(Torchnn.parameters(self.network), lr=learning_rate)
        self.loss_func = nn.MSELoss()
        self.errors = []

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
            self.errors.append(torch.sqrt(loss))

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print('Training took {} seconds'.format(time.time() - startTime))

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
