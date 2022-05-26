import torch
import torch.nn.functional as F
import torch.nn as nn

class EncoderLayer(nn.Sequential):
    """
    Encoder layer
    in_features: input dimension (Integer)
    out_features: output dimension (Integer)
    drop_p: drop out rate (default: 0.5)
    """
    def __init__(self, in_features:int, out_features:int, drop_p:float= 0.5):
        super().__init__(
                nn.Linear(in_features=in_features, out_features=out_features),
                nn.Dropout(p=drop_p, inplace=False),
                nn.ReLU()
        )

class DecoderLayer(nn.Sequential):
    """
    Decoder layer
    in_features: input dimension (Integer)
    out_features: output dimension (Integer)
    drop_p: drop out rate (default: 0.5)
    """
    def __init__(self, in_features:int, out_features:int, drop_p:float= 0.5):
        super().__init__(
                nn.Linear(in_features=in_features, out_features=out_features),
                nn.Dropout(p=drop_p, inplace=False),
                nn.ReLU()
        )

class EncoderDoubleLayer(nn.Sequential):
    """
    Encoder layer
    in_features: input dimension (Integer)
    out_features: output dimension (Integer)
    drop_p: drop out rate (default: 0.5)
    """
    def __init__(self, in_features:int, middle_features:int, out_features:int, drop_p:float= 0.5):
        super().__init__(
                nn.Linear(in_features=in_features, out_features=middle_features),
                nn.Dropout(p=drop_p, inplace=False),
                nn.BatchNorm1d(middle_features),
                nn.ReLU(),
                nn.Linear(middle_features, out_features),
                nn.Dropout(p= drop_p, inplace= False),
                nn.BatchNorm1d(out_features)
        )

class DecoderDoubleLayer(nn.Sequential):
    """
    Encoder layer
    in_features: input dimension (Integer)
    out_features: output dimension (Integer)
    drop_p: drop out rate (default: 0.5)
    """
    def __init__(self, in_features:int, middle_features:int, out_features:int, drop_p:float= 0.5):
        super().__init__(
                nn.Linear(in_features=in_features, out_features=middle_features),
                nn.Dropout(p=drop_p, inplace=False),
                nn.BatchNorm1d(middle_features),
                nn.ReLU(),
                nn.Linear(middle_features, out_features),
                nn.Dropout(p= drop_p, inplace= False),
                nn.BatchNorm1d(out_features),
                nn.ReLU()
        )

# deprecated...
class RegressionTaskHead(nn.Module):
    """
    Regression Task Head
    returns y_hat = X@W = X@U@V (thus W=UV)
    where (1) a column vector of u spans weight vector space and
    (2) a column vector of V controls the coefficients of the column vectors of U to represent a weight vector
    """
    def __init__(self, latent_dim, latent2task_dim, n_targets):
        super().__init__()
        # params
        self.u = nn.Parameter(torch.empty((latent_dim, latent2task_dim)))
        self.v = nn.Parameter(torch.empty((latent2task_dim, n_targets)))

        self.latent_dim = latent_dim
        self.latent2task_dim = latent2task_dim
        self.n_targets = n_targets

    def forward(self, x):
        return x@self.u@self.v