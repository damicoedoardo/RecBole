#!/usr/bin/env python
__author__ = "Edoardo D'Amico"
__email__ = "edoardo.d'amico@insight-centre.org"

import numpy as np
import scipy.sparse as sps
import torch

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType


class SGCCF(GeneralRecommender):
    """Shallow Graph Convolutional Collaborative Filtering

    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(SGCCF, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)

        # load parameters info
        self.latent_dim = config[
            "embedding_size"
        ]  # int type:the embedding size
        self.n_layers = config["n_layers"]  # int type:convolution_depth
        self.propagation_matrix = config["propagation_matrix"]

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim
        )
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_dim
        )
        self.mf_loss = BPRLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat(self.propagation_matrix).to(self.device)

        # parameters initialization
        self.apply(xavier_uniform_initialization)

    def get_norm_adj_mat(self, kind: str):
        r"""Construct the propagation matrix

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        assert kind in ["SNL", "RW"], "Propagation matrix kind should be in SNL or RW"

        inter_matrix = self.interaction_matrix
        inter_matrix_t = self.interaction_matrix.transpose()
        A = sps.block_diag((inter_matrix, inter_matrix_t))

        # Add self loop
        # convert to lil matrix for efficency of the method set diag
        A = A.tolil()
        # note: setdiag is an inplace operation
        A.setdiag(1)
        # bring back the matrix into csr format
        A = A.tocsr()

        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        if kind == "SNL":
            diag = np.power(diag, -0.5)
            D = sps.diags(diag)
            P = D * A * D
        elif kind == "RW":
            diag = np.power(diag, -1)
            D = sps.diags(diag)
            P = D * A * D

        # covert norm_adj matrix to tensor
        P = sps.coo_matrix(P)
        row = P.row
        col = P.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(P.data)
        torch_sparse_P = torch.sparse.FloatTensor(i, data, torch.Size(P.shape))
        return torch_sparse_P

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)

        user_all_embeddings, item_all_embeddings = torch.split(
            all_embeddings, [self.n_users, self.n_items]
        )
        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        loss = mf_loss

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)
