import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class Generator(nn.Module):
    """
    Standard nonlinear MLP generator for 2D point data.
    Used as a control model for the NCP generator.

    Input:
        z with shape [batch_size, z_dim]

    Output:
        generated points with shape [batch_size, 2]
        Each row is one point: [x_coordinate, y_coordinate]
    """

    def __init__(
        self,
        z_dim=2,
        hidden_dim=4,
        out_dim=2,
        num_layers=2,

        activation_fn=True,
        bound_output=False

       
    ):
        super().__init__()

        # z_dim:
        # Number of random values in one noise vector.
        # This is not the number of circle coordinates.
        # Example: one z vector has shape [16].
        self.z_dim = z_dim

        # hidden_dim:
        # Number of internal features used by the generator.
        # 128 is a design choice, not a property of the circle.
        # self.hidden_dim is here save generator config
        self.hidden_dim = hidden_dim

        #note
        self.out_dim=out_dim

    
        # self.hidden_dim is here to save generator config
        self.num_layers =  num_layers

      

        #simply here just to save generator config
        self.activation_fn = activation_fn

        #note   
        self.bound_output = bound_output

        # ReLU can improve practical training.
        #
        # activation_fn=False:
        # The generator remains a strict polynomial function,
        # apart from an optional output Tanh.
        #
        # activation_fn=True:
        # The generator becomes piecewise polynomial because of ReLU.
        self.activation = (
            nn.ReLU(inplace=False)
            if activation_fn
            else nn.Identity()
        )

        


        # A_1^t+b first linear transform
        self.A0= nn.Linear(z_dim, hidden_dim,bias=True)

        # A_N^t+b linear transform
        self.A = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim,bias=True)for _ in range(num_layers-1)]) 

        
       

 


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
       
        x = self.A0(zeta)
        x = self.activation(x)

        for layer in self.A:
         x = layer(x)
         x = self.activation(x)

            

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
            


            # make_linear(input_dim, hidden_dim),
            # nn.Sigmoid(),
            # make_linear(hidden_dim, hidden_dim),
            # nn.Sigmoid(),
            # make_linear(hidden_dim, 1),
            # nn.Sigmoid(),

            # make_linear(input_dim, hidden_dim),
            # nn.ReLU(),
            # make_linear(hidden_dim,1),
            # nn.Sigmoid()
            
        )

    def forward(self, points):
        
    
        logits = self.main(points)

        return logits