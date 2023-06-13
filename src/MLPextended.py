import torch
torch.set_default_dtype(torch.float64)

class MLP(torch.nn.Module):
    """
        A MLP with multiple hidden layers, whose output is set to be a scalar.
    The gradient with respect to the input is handcoded for further convenience.
    The network is fully connected. At each layer, sigmoid activation is used.
    """
    def __init__(self, layer_widths, device='cpu'):
        """
            list_of_widths: List of layer widths/dimensions, first value
                            is taken to be input dimension.
                
                [D_in, D_hidden_1, ..., D_hidden_N]
        """
        super(MLP, self).__init__()
        self.device = device
        self.layers = torch.nn.ModuleList()
        self.layer_dims = layer_widths
        self.N_layers = len(layer_widths)-1
        
        for i in range(self.N_layers):
            self.layers.append(torch.nn.Linear(self.layer_dims[i], self.layer_dims[i+1]))
        self.layers.append(torch.nn.Linear(self.layer_dims[-1], 1, bias=False))
        
        self.activation = torch.nn.Sigmoid()

    def init_zeros(self):
        for i in range(self.N_layers):
            torch.nn.init.zeros_(self.layers[i].weight)
            torch.nn.init.zeros_(self.layers[i].bias)
        torch.nn.init.zeros_(self.layers[-1].weight)

    def init_gaussian(self, seed):
        torch.manual_seed(seed)
        std = 1e-3
        for i in range(self.N_layers):
            torch.nn.init.normal_(self.layers[i].weight, std=std)
            torch.nn.init.normal_(self.layers[i].bias, std=std)
        torch.nn.init.normal_(self.layers[-1].weight, std=std)

    def forward(self, x):
        for i in range(self.N_layers):
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
        # Start with gradient of output wrt values last layer
        # Then with weights and activation of layer before that,
        #   obtain gradient of output wrt layer before
        # Repeat until end.
        # Do one forward run and save values.
        # x_N = layer_n(x_N-1)
        # z_N = activation(x_N)
        # Pseudo code:
        """
        # Calculate all outputs of layers:
            [z_1, z_2, ..., z_N] 
                where   z_i = sigmoid(x_i)
                and     x_i = z_(i-1) @ weight_i + bias_i
        # Go backward, calculate jacobian of layer n, matmul with jacobian of layers n+1...N for
        # jacobian of layers n...N
            grad_total = identity of appropiate size
            for each n from N-1 to 1:
                grad_n = weight_N+1 * d_sigmoid(z_n)
                grad_total = grad_total@grad_n
        # Finish by matmul with weights of layer 1
            grad_total = grad_total@weight_1

        """
        z = {}
        for i in range(self.N_layers):
            x = self.activation(self.layers[i](x))
            z[i] = x[:,:,None]
        grad_total = torch.eye(1)
        for i in range(self.N_layers, 0, -1):
            grad_n = self.layers[i].weight * self.d_sigmoid(z[i-1])
            grad_total = grad_total.matmul(grad_n.to(self.device))
        grad_x = grad_total.matmul(self.layers[0].weight)
        """
        grad_x = self.layers[0].weight
        print(grad_x.shape)
        for i in range(1, self.N_layers+1):
            x = self.activation(self.layers[i-1](x))
            print(x.shape)
            print(self.d_sigmoid(x).shape)
            print(self.layers[i].weight.shape)
            grad_layer = self.layers[i].weight * self.d_sigmoid(x)
            grad_x = grad_layer.matmul(grad_x)
        """
        
        return torch.squeeze(grad_x, dim=2)
