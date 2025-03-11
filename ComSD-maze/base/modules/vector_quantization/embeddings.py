# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT


import torch
import torch.nn as nn
from .functions import vector_quantization, vector_quantization_st


class VQEmbedding(nn.Module):
    """
    Vector Quantization module for VQ-VAE (van der Oord et al., https://arxiv.org/abs/1711.00937)
    This module is compatible with 1D latents only (i.e. with inputs of shape [batch_size, embedding_dim]).

    Adapted from https://github.com/ritheshkumar95/pytorch-vqvae/blob/master/modules.py#L70

    Variable names follow those in the paper:
        z_e_x: z_e(x), i.e. the *continuous* encoding emitted by the encoder
        z_q_x: z_q(x), i.e. the decoder input -- the vector-quantized version of z_e(x)  [Eq. 2]
    """
    def __init__(self, codebook_size, code_size, beta):
        """
        :param codebook_size: number of codes in the codebook
        :param code_size: dimensionality of each code
        :param beta: weight for the commitment loss
        """
        super().__init__()

        self.codebook_size = int(codebook_size)
        self.code_size = int(code_size)
        self.beta = float(beta)

        self.embedding = nn.Embedding(self.codebook_size, self.code_size)
        self.embedding.weight.data.uniform_(-1./self.codebook_size, 1./self.codebook_size)

        self.mse_loss = nn.MSELoss(reduction='none')

    def quantize(self, z_e_x):
        return vector_quantization(z_e_x, self.embedding.weight)

    def straight_through(self, z_e_x):
        # Quantized vectors (inputs for the decoder)
        z_q_x, indices = vector_quantization_st(z_e_x, self.embedding.weight.detach())
        # Selected codes from the codebook (for the VQ objective)
        selected_codes = torch.index_select(self.embedding.weight, dim=0, index=indices)
        return z_q_x, selected_codes

    def forward(self, z_e_x, selected_codes=None):
        """
        Compute second and third loss terms in Eq. 3 in the paper
        :param z_e_x: encoder output
        :param selected_codes: (optional) second output from straight_through(); avoids recomputing it
        :return: loss = vq_loss + beta * commitment_loss
        """
        # Recompute z_q(x) if needed
        if selected_codes is None:
            _, selected_codes = self.straight_through(z_e_x)
        # Push VQ codes towards the output of the encoder
        vq_loss = self.mse_loss(selected_codes, z_e_x.detach()).sum(dim=1)
        # Encourage the encoder to commit to a code
        commitment_loss = self.mse_loss(z_e_x, selected_codes.detach()).sum(dim=1)
        # The scale of the commitment loss is controlled with beta [Eq. 3]
        loss = vq_loss + self.beta * commitment_loss
        return loss

    def compute_distances(self, inputs):
        with torch.no_grad():
            embedding_size = self.embedding.weight.size(1)
            inputs_flatten = inputs.view(-1, embedding_size)

            codebook_sqr = torch.sum(self.embedding.weight ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr, inputs_flatten, self.embedding.weight.t(),
                                    alpha=-2.0, beta=1.0)

            return distances
