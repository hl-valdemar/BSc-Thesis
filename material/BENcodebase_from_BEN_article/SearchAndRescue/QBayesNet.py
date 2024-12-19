from torch import nn
import torch.nn.init as init


class QBayesNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.rnn = nn.GRUCell(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(64, output_size)

    def reset_net(self):
        self.fc1.reset_parameters()
        self.rnn.reset_parameters()
        self.fc2.reset_parameters()
        self.out.reset_parameters()

    def init_params(self):
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        init.ones_(self.out.weight)
        init.xavier_uniform_(self.rnn.weight_hh)
        init.xavier_uniform_(self.rnn.weight_ih)
        init.ones_(self.fc1.bias)
        init.ones_(self.fc2.bias)
        init.ones_(self.out.bias)
        init.zeros_(self.rnn.bias_hh)
        init.zeros_(self.rnn.bias_ih)

    def init_hidden(self):
        # make hidden states on same device as model, and same batch size as input
        return self.fc1.weight.new(1, 64).zero_()

    def forward(self, x, hidden_state):
        x1 = self.fc1(x)
        x2 = self.relu1(x1)
        h_in = hidden_state.reshape(-1, 64)
        h_out = self.rnn(x2, h_in)
        x3 = self.fc2(h_out)
        x3 = self.relu2(x3)
        x3 = self.out(x3)
        return x3, h_out
