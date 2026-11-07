import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class Generator(nn.Module):
    """
    Polynomial generator for 2D point data.

    Input:
        z with shape [batch_size, z_dim]

    Output:
        generated points with shape [batch_size, 2]
        Each row is one point: [x_coordinate, y_coordinate]
    """

    def __init__(
        self,
        z_dim=1,
        hidden_dim=15,
        out_dim=2,
        num_orders=12,
        b_dim=1, #constant per ncp formula
        activation_fn=False,
        bound_output=False

       
    ):
        super().__init__()

        
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.out_dim=out_dim
        self.num_orders =  num_orders
        self.b_dim=b_dim
        self.activation_fn = activation_fn
        self.bound_output = bound_output

        
        self.activation = (
            nn.ReLU(inplace=False)
            if activation_fn
            else nn.Identity()
        )

        

        self.z_transform = nn.Linear(z_dim,z_dim,bias=True)
        self.A = nn.ModuleList([nn.Linear(z_dim, hidden_dim,bias=False)for _ in range(num_orders)]) 
        self.S = nn.ModuleList([nn.Linear(hidden_dim,hidden_dim, bias=False)for _ in range(num_orders-1)])

        #B1^TB1 bias term for X1
        # B_n^T
        #self.B = nn.ModuleList([nn.Linear(in_features=b_dim,out_features=hidden_dim,bias=False)for _ in range(num_orders)])
        self.B = nn.ModuleList([nn.Identity()for _ in range(num_orders)])

        # One b_n vector for each polynomial order.
        self.b = nn.ParameterList([
            nn.Parameter(torch.ones(hidden_dim))
            for _ in range(num_orders)
        ])


        self.out = nn.Linear(hidden_dim,out_dim, bias=True)

        # Tanh restricts generated coordinates to approximately [-1, 1].
        #
        # This can be useful for a unit-circle dataset.
        # However, Tanh means the complete generator is no longer
        # strictly one global polynomial.
        self.output_activation = (
            nn.Tanh()
            if bound_output
            else nn.Identity()
        )

    def forward(self, zeta):

        z = self.z_transform(zeta)
        
        #A_1^T z
        az = self.A[0](z) 

          # B_1^T b_1
        #
        # Shape: [hidden_dim]
        bb = self.B[0](self.b[0])

        # [batch, hidden_dim] * [hidden_dim]
        #
        # x_1 = (A_1^T z) ∘ (B_1^T b_1)
        x = az * bb

        for n in range(1, self.num_orders):
            # A_n^T z
            az = self.A[n](z)

            # S_n^T x_(n-1)
            sx = self.S[n-1](x)

            # B_n^T b_n
            bb = self.B[n](self.b[n])

            # x_n = (A_n^T z) ∘
            #       (S_n^T x_(n-1) + B_n^T b_n)
            x = az * (sx + bb)
           
            #x=self.activation(x)

        # f(z) = Q x_k + psi
        return self.output_activation(self.out(x))


class Discriminator(nn.Module):
    """
    Discriminator for 2D point data.

    Input:
        points with shape [batch_size, 2]

    Output:
        logits with shape [batch_size, 1]

    One score is produced for every [x, y] point.
    """

    def __init__(
        self,
        input_dim=2,
        hidden_dim=128,
        use_spectral_norm=False
    ):
        super().__init__()

        self.input_dim = input_dim

        
        self.hidden_dim = hidden_dim

        #again only used to print the config file
        self.use_spectral_norm = use_spectral_norm

        
        def make_linear(in_features, out_features):
            layer = nn.Linear(
                in_features=in_features,
                out_features=out_features,
                bias=True
            )
            if use_spectral_norm:
                layer = spectral_norm(layer)
            return layer

        self.main = nn.Sequential(
           
            make_linear(input_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.2,inplace=False),
            make_linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.2,inplace=False),
            make_linear(hidden_dim, 1),
            #nn.Sigmoid(),


            # make_linear(input_dim, hidden_dim),
            # nn.Sigmoid(),
            # make_linear(hidden_dim, hidden_dim),
            # nn.Sigmoid(),
            # make_linear(hidden_dim, 1),
            # nn.Sigmoid(),

            # make_linear(input_dim, hidden_dim),
            # nn.LeakyReLU(),
            # make_linear(hidden_dim,hidden_dim),
            # nn.LeakyReLU(),
            # make_linear(hidden_dim,hidden_dim),
            # nn.LeakyReLU(),
            # make_linear(hidden_dim,hidden_dim),
            # nn.LeakyReLU(),
            # make_linear(hidden_dim,hidden_dim//2),
            # nn.LeakyReLU(),
            # make_linear(hidden_dim//2,1),
            #nn.Sigmoid()
            
        )

    def forward(self, points):
        
    
        logits = self.main(points)

        return logits