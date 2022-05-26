import torch
import torch.nn.functional as F
import torch.nn as nn
from layers import *

class MultiClassifictaionTaskModel(nn.Module):
    """
    AutoEncoder
    in_features: input dimension (int)
    latent_dim: latent vector dimension (int)
    n_targets: the number of targets (tasks) (int)
    n_labels: number of labels of each task (list)
    drop_p: drop out rate (default: 0.5)
    noise_scale: scale noise by "noise_scale"
    """
    def __init__(self, 
                in_features:int, latent_dim:int, n_targets:int= 3,
                n_labels:list= [3, 3, 3], drop_p:float = 0.2,
                noise_scale:float = 1e-2
                ):
        super().__init__()
        
        # Auto encoder structure
        self.encoder_layer = EncoderLayer(in_features, latent_dim, drop_p)
        self.decoder_layer = DecoderLayer(latent_dim, in_features, drop_p)
        
        # Prediction heads
        for i, n_label in enumerate(n_labels):
            setattr(self, f"prediction_head{i}", nn.Linear(latent_dim, n_label)) 

        # Properties
        self.in_features = in_features
        self.latent_dim = latent_dim
        self.n_targets = n_targets
        self.n_labels = n_labels
        self.drop_p = drop_p
        self.noise_scale = noise_scale

    def forward(self, x):
        x_tilde = self.add_noise(x) if self.training else x
        z = self.encoder_layer(x_tilde)
        x_hat = self.decoder_layer(z)
        y_hats = []
        for i in range(self.n_targets):
            head = getattr(self, f"prediction_head{i}")
            y_hats.append(head(z))
        return y_hats, x_hat
    
    def predict(self, x):
        y_hats, _ = self.forward(x)
        y_hats = torch.stack([torch.argmax(F.softmax(y_hat, dim= 1), dim= 1) for y_hat in y_hats], dim= 1)
        return y_hats

    def add_noise(self, x):
        device = x.device
        noise = torch.randn(x.size()) * self.noise_scale
        noise = noise.to(device)
        return x + noise

class MultiClassifictaionTaskModel2(nn.Module):
    """
    AutoEncoder
    in_features: input dimension (int)
    latent_dim: latent vector dimension (int)
    n_targets: the number of targets (tasks) (int)
    n_labels: number of labels of each task (list)
    drop_p: drop out rate (default: 0.5)
    noise_scale: scale noise by "noise_scale"
    """
    def __init__(self, 
                in_features:int, latent_dim:int, n_targets:int= 3,
                n_labels:list= [3, 3, 3], drop_p:float = 0.2,
                noise_scale:float = 1e-2
                ):
        super().__init__()
        
        # Auto encoder structure
        self.encoder_layer = EncoderDoubleLayer(in_features, latent_dim, latent_dim, drop_p)
        self.decoder_layer = DecoderDoubleLayer(latent_dim, latent_dim, in_features, drop_p)
        
        # Prediction heads
        for i, n_label in enumerate(n_labels):
            head = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.BatchNorm1d(latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, n_label)
            )
            setattr(self, f"prediction_head{i}", head) 

        # Properties
        self.in_features = in_features
        self.latent_dim = latent_dim
        self.n_targets = n_targets
        self.n_labels = n_labels
        self.drop_p = drop_p
        self.noise_scale = noise_scale

    def forward(self, x):
        x_tilde = self.add_noise(x) if self.training else x
        z = self.encoder_layer(x_tilde)
        x_hat = self.decoder_layer(z)
        y_hats = []
        for i in range(self.n_targets):
            head = getattr(self, f"prediction_head{i}")
            y_hats.append(head(z))
        return y_hats, x_hat
    
    def predict(self, x):
        y_hats, _ = self.forward(x)
        y_hats = torch.stack([torch.argmax(F.softmax(y_hat, dim= 1), dim= 1) for y_hat in y_hats], dim= 1)
        return y_hats

    def add_noise(self, x):
        device = x.device
        noise = torch.randn(x.size()) * self.noise_scale
        noise = noise.to(device)
        return x + noise