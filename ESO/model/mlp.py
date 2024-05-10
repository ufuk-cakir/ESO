import torch.nn as nn


class MLP(nn.Module):
    # TODO do we need stuff like dropout rate, batch norm, etc.?
    def __init__(self, input_dim, output_dim, n_hidden_layers, hidden_dim, activation_function, dropout_rate=None,
                 output_activation_function=lambda x: x):
        super(MLP, self).__init__()
        # Set up parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden_layers = n_hidden_layers
        self.hidden_dim = hidden_dim
        self.activation_function = activation_function
        self.output_activation_function = output_activation_function
        self.dropout = nn.Dropout(
            dropout_rate) if dropout_rate is not None else None
        # Set up layers
        self.input_to_hidden = nn.Linear(self.input_dim, self.hidden_dim)
        self.hidden_layers = nn.ModuleList()
        for i in range(self.n_hidden_layers):
            self.hidden_layers.append(
                nn.Linear(self.hidden_dim, self.hidden_dim))
            if self.dropout is not None:
                self.hidden_layers.append(self.dropout)
        self.hidden_to_output = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        x = self.activation_function(self.input_to_hidden(x))
        for layer in self.hidden_layers:
            x = self.activation_function(layer(x))
        x = self.output_activation_function(self.hidden_to_output(x))
        return x
