import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import pickle
from transformers import AutoModel


class MultiDAE(nn.Module):
    """
    Container module for Multi-DAE.
    Multi-DAE : Denoising Autoencoders with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, p_dims, q_dims=None, dropout=0.5):
        super(MultiDAE, self).__init__()
        self.p_dims = p_dims
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]

        self.dims = self.q_dims + self.p_dims[1:]
        self.layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                     d_in, d_out in zip(self.dims[:-1], self.dims[1:])])
        self.drop = nn.Dropout(dropout)

        self.init_weights()

    def forward(self, x):
        h = F.normalize(x)
        h = self.drop(h)

        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != len(self.weights) - 1:
                h = torch.tanh(h)
        return h

    def init_weights(self):
        for layer in self.layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)


class MultiVAE(nn.Module):
    """
    Container module for Multi-VAE.
    Multi-VAE : Variational Autoencoders with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, p_dims, q_dims=None, dropout=0.5):
        super(MultiVAE, self).__init__()
        self.p_dims = p_dims
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]

        # Last dimension of q- network is for mean and variance
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]
        self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])])

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def encode(self, x):
        h = F.normalize(x)
        h = self.drop(h)

        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = torch.tanh(h)
            else:
                mu = h[:, :self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1]:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = torch.tanh(h)
        return h

    def init_weights(self):
        for layer in self.q_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.p_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)


def loss_function(recon_x, x, mu, logvar, anneal=1.0):
    # bce = F.binary_cross_entropy(recon_x, x)
    bce = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    kld = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    return bce + anneal * kld


class MovieEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
                        pretrained_model_name_or_path=config['pretrained_model_name_or_path'],
                        cache_dir=config['pretrained_dir_cache'])

        self.classifier = nn.Sequential(
            nn.Dropout(config['dropout']),
            nn.Linear(self.encoder.config.hidden_size, config['classifier_hidden_size']),
            # Batch Normalization here if needed
            nn.RReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['classifier_hidden_size'], config['n_classes']),
            nn.Sigmoid()
        )

        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                # Xavier Initialization for weights
                size = layer.weight.size()
                fan_out = size[0]
                fan_in = size[1]
                std = np.sqrt(2.0 / (fan_in + fan_out))
                layer.weight.data.normal_(mean=0.0, std=std)
                if layer.bias is not None:
                    layer.bias.data.normal_(0.0, 0.001)

    def forward(self, text, attention_mask):
        outputs = self.encoder(text, attention_mask=attention_mask)
        hidden_states = outputs[0]
        # here I use only representation of classification token (<cls>/<s>)
        return self.classifier(hidden_states[:, 0, :])


class UserEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=config.num_movie + config.num_special_token,
                                      embedding_dim=config.hidden_size,
                                      padding_idx=config.num_movie)

        # Initialize embedding weights with pretrained movie embedding
        with open(config.embedding_weights_path, 'rb') as fp:
            weights = torch.tensor(pickle.load(fp))
        self.embedding.weight.data[:config.num_movie, :] = weights

        # Freeze embedding weights
        for name, param in self.embedding.named_parameters():
            param.requires_grad = False

        encoder_layer = nn.TransformerEncoderLayer(d_model=config.hidden_size,
                                                   nhead=config.num_head,
                                                   dim_feedforward=config.feedforward_size)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layer)

        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.classifier_hidden_size_1),
            # Batch Normalization here if needed
            nn.BatchNorm1d(num_features=config.classifier_hidden_size_1),
            nn.Tanh(),
            nn.Dropout(config.dropout),
            nn.Linear(config.classifier_hidden_size_1, config.classifier_hidden_size_2),
            # Batch Normalization here if needed
            nn.BatchNorm1d(num_features=config.classifier_hidden_size_2),
            nn.Tanh(),
            nn.Dropout(config.dropout),
            nn.Linear(config.classifier_hidden_size_2, config.num_movie),
            nn.Sigmoid()
        )

        # Xavier Initialization for classifier's weights
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                size = layer.weight.size()
                fan_out = size[0]
                fan_in = size[1]
                std = np.sqrt(2.0 / (fan_in + fan_out))
                layer.weight.data.normal_(mean=0.0, std=std)
                if layer.bias is not None:
                    layer.bias.data.normal_(0.0, 0.001)

    def forward(self, x):
        x = self.encoder(self.embedding(x))
        return self.classifier(x[:, 0, :])
        # return self.classifier(torch.mean(x, axis=1))

