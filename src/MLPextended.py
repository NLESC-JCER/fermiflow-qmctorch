import torch
torch.set_default_dtype(torch.float64)

class MLP(torch.nn.Module):
    """
        A MLP with multiple hidden layers, whose output is set to be a scalar.
    The gradient with respect to the input is handcoded for further convenience.
    The network is fully connected. At each layer, sigmoid activation is used.
    """
    def __init__(self, layer_widths):
        """
            list_of_widths: List of layer widths/dimensions, first value
                            is taken to be input dimension.
                
                [D_in, D_hidden_1, ..., D_hidden_N]
        """
        super(MLP, self).__init__()
        self.layers = []
        self.layer_dims = layer_widths
        self.N_layers = len(layer_widths)-1
        
        for i in range(self.N_layers-1):
            self.layers.append(torch.nn.Linear(self.layer_dims[i], self.layer_dims[i+1]))
        self.layers.append(torch.nn.Linear(self.layer_dims[-1], 1, bias=False))
        
        self.activation = torch.nn.Sigmoid()

    def init_zeros(self):
        for i in range(self.N_layers-1):
            torch.nn.init.zeros_(self.layers[i].weight)
            torch.nn.init.zeros_(self.layers[i].bias)
        torch.nn.init.zeros_(self.layers[-1].weight)

    def init_gaussian(self, seed):
        torch.manual_seed(seed)
        std = 1e-3
        for i in range(self.N_layers-1):
            torch.nn.init.normal_(self.layers[i].weight, std=std)
            torch.nn.init.normal_(self.layers[i].bias, std=std)
        torch.nn.init.normal_(self.layers[-1].weight, std=std)

    def forward(self, x):
        for i in range(self.N_layers-1):
            x = self.activation(self.layers[i](x))
        output = self.layers[-1](x)
        return output

    def d_sigmoid(self, output):
        return output * (1. - output)

    def grad(self, x):
        """
            Note that this implementation of grad works for the general case
        where x has ANY batch dimension, i.e., x has shape (..., D_in).
        """
        grad_x = self.layers[0].weight
        for i in range(1, self.N_layers):
            x = self.activation(self.layers[i-1](x))
            grad_layer = self.layers[i].weight * self.d_sigmoid(x)
            grad_x = grad_layer.matmul(grad_x)
        return grad_x
