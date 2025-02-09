from abc import ABC
from typing import Union

import torch
from torch import Tensor
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel, RobertaPreTrainedModel
from .layer import PolyAttention, TargetAwareAttention

def pairwise_cosine_similarity(x: Tensor, y: Tensor, zero_diagonal: bool = False) -> Tensor:
    r"""
    Calculates the pairwise cosine similarity matrix

    Args:
        x: tensor of shape ``(batch_size, M, d)``
        y: tensor of shape ``(batch_size, N, d)``
        zero_diagonal: determines if the diagonal of the distance matrix should be set to zero

    Returns:
        A tensor of shape ``(batch_size, M, N)``
    """
    x_norm = torch.linalg.norm(x, dim=2, keepdim=True)
    y_norm = torch.linalg.norm(y, dim=2, keepdim=True)
    distance = torch.matmul(torch.div(x, x_norm), torch.div(y, y_norm).permute(0, 2, 1))
    if zero_diagonal:
        assert x.shape[1] == y.shape[1]
        mask = torch.eye(x.shape[1]).repeat(x.shape[0], 1, 1).bool().to(distance.device)
        distance.masked_fill_(mask, 0)

    return distance

class NewsEncoder(ABC, RobertaPreTrainedModel):
    def __init__(self, config: RobertaConfig, apply_reduce_dim: bool, use_sapo: bool, dropout: float,
                 freeze_transformer: bool, word_embed_dim: Union[int, None] = None,
                 combine_type: Union[str, None] = None, lstm_num_layers: Union[int, None] = None,
                 lstm_dropout: Union[float, None] = None):
        r"""
        Initialization

        Args:
            config: the configuration of a ``RobertaModel``.
            apply_reduce_dim: whether to reduce the dimension of Roberta's embedding or not.
            use_sapo: whether to use sapo embedding or not.
            dropout: dropout value.
            freeze_transformer: whether to freeze Roberta weight or not.
            word_embed_dim: size of each word embedding vector if ``apply_reduce_dim``.
            combine_type: method to combine news information.
            lstm_num_layers: number of recurrent layers in LSTM.
            lstm_dropout: dropout value in LSTM.
        """
        super().__init__(config)
        self.roberta = RobertaModel(config)
        if freeze_transformer:
            for param in self.roberta.parameters():
                param.requires_grad = False

        self.apply_reduce_dim = apply_reduce_dim

        if self.apply_reduce_dim:
            assert word_embed_dim is not None
            self.reduce_dim = nn.Linear(in_features=config.hidden_size, out_features=word_embed_dim)
            self.word_embed_dropout = nn.Dropout(dropout)
            self._embed_dim = word_embed_dim
        else:
            self._embed_dim = config.hidden_size

        self.use_sapo = use_sapo
        if self.use_sapo:
            assert combine_type is not None
            self.combine_type = combine_type
            if self.combine_type == 'linear':
                self.linear_combine = nn.Linear(in_features=self._embed_dim * 2, out_features=self._embed_dim)
            elif self.combine_type == 'lstm':
                self.lstm = nn.LSTM(input_size=self._embed_dim * 2, hidden_size=self._embed_dim // 2,
                                    num_layers=lstm_num_layers, batch_first=True, dropout=lstm_dropout,
                                    bidirectional=True)
                self._embed_dim = (self._embed_dim // 2) * 2

        self.init_weights()

    def forward(self, title_encoding: Tensor, title_attn_mask: Tensor, sapo_encoding: Union[Tensor, None] = None,
                sapo_attn_mask: Union[Tensor, None] = None):
        r"""
        Forward propagation

        Args:
            title_encoding: tensor of shape ``(batch_size, title_length)``.
            title_attn_mask: tensor of shape ``(batch_size, title_length)``.
            sapo_encoding: tensor of shape ``(batch_size, sapo_length)``.
            sapo_attn_mask: tensor of shape ``(batch_size, sapo_length)``.

        Returns:
            Tensor of shape ``(batch_size, embed_dim)``
        """
        news_info = []
        # Title encoder
        title_word_embed = self.roberta(input_ids=title_encoding, attention_mask=title_attn_mask)[0]
        title_repr = title_word_embed[:, 0, :]
        if self.apply_reduce_dim:
            title_repr = self.reduce_dim(title_repr)
            title_repr = self.word_embed_dropout(title_repr)
        news_info.append(title_repr)

        # Sapo encoder
        if self.use_sapo:
            sapo_word_embed = self.roberta(input_ids=sapo_encoding, attention_mask=sapo_attn_mask)[0]
            sapo_repr = sapo_word_embed[:, 0, :]
            if self.apply_reduce_dim:
                sapo_repr = self.reduce_dim(sapo_repr)
                sapo_repr = self.word_embed_dropout(sapo_repr)
            news_info.append(sapo_repr)

            if self.combine_type == 'linear':
                news_info = torch.cat(news_info, dim=1)

                return self.linear_combine(news_info)
            elif self.combine_type == 'lstm':
                news_info = torch.cat(news_info, dim=1)
                news_repr, _ = self.lstm(news_info)

                return news_repr
        else:
            return title_repr

    @property
    def embed_dim(self):
        return self._embed_dim

from typing import Union

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as torch_f

class Miner(nn.Module):
    r"""
    Implementation of Multi-interest matching network for news recommendation. Please see the paper in
    https://aclanthology.org/2022.findings-acl.29.pdf.
    """
    def __init__(self, news_encoder: NewsEncoder, use_category_bias: bool, num_context_codes: int,
                 context_code_dim: int, score_type: str, dropout: float, num_category: Union[int, None] = None,
                 category_embed_dim: Union[int, None] = None, category_pad_token_id: Union[int, None] = None,
                 category_embed: Union[Tensor, None] = None):
        r"""
        Initialization

        Args:
            news_encoder: NewsEncoder object.
            use_category_bias: whether to use Category-aware attention weighting.
            num_context_codes: the number of attention vectors ``K``.
            context_code_dim: the number of features in a context code.
            score_type: the ways to aggregate the ``K`` matching scores as a final user click score ('max', 'mean' or
                'weighted').
            dropout: dropout value.
            num_category: the size of the dictionary of categories.
            category_embed_dim: the size of each category embedding vector.
            category_pad_token_id: ID of the padding token type in the category vocabulary.
            category_embed: pre-trained category embedding.
        """
        super().__init__()
        self.news_encoder = news_encoder
        self.news_embed_dim = self.news_encoder.embed_dim
        self.use_category_bias = use_category_bias
        if self.use_category_bias:
            self.category_dropout = nn.Dropout(dropout)
            if category_embed is not None:
                self.category_embedding = nn.Embedding.from_pretrained(category_embed, freeze=False,
                                                                       padding_idx=category_pad_token_id)
                self.category_embed_dim = category_embed.shape[1]
            else:
                assert num_category is not None
                self.category_embedding = nn.Embedding(num_embeddings=num_category, embedding_dim=category_embed_dim,
                                                       padding_idx=category_pad_token_id)
                self.category_embed_dim = category_embed_dim

        self.poly_attn = PolyAttention(in_embed_dim=self.news_embed_dim, num_context_codes=num_context_codes,
                                       context_code_dim=context_code_dim)
        self.score_type = score_type
        if self.score_type == 'weighted':
            self.target_aware_attn = TargetAwareAttention(self.news_embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, title: Tensor, title_mask: Tensor, his_title: Tensor, his_title_mask: Tensor,
                his_mask: Tensor, sapo: Union[Tensor, None] = None, sapo_mask: Union[Tensor, None] = None,
                his_sapo: Union[Tensor, None] = None, his_sapo_mask: Union[Tensor, None] = None,
                category: Union[Tensor, None] = None, his_category: Union[Tensor, None] = None):
        r"""
        Forward propagation

        Args:
            title: tensor of shape ``(batch_size, num_candidates, title_length)``.
            title_mask: tensor of shape ``(batch_size, num_candidates, title_length)``.
            his_title: tensor of shape ``(batch_size, num_clicked_news, title_length)``.
            his_title_mask: tensor of shape ``(batch_size, num_clicked_news, title_length)``.
            his_mask: tensor of shape ``(batch_size, num_clicked_news)``.
            sapo: tensor of shape ``(batch_size, num_candidates, sapo_length)``.
            sapo_mask: tensor of shape ``(batch_size, num_candidates, sapo_length)``.
            his_sapo: tensor of shape ``(batch_size, num_clicked_news, sapo_length)``.
            his_sapo_mask: tensor of shape ``(batch_size, num_clicked_news, sapo_length)``.
            category: tensor of shape ``(batch_size, num_candidates)``.
            his_category: tensor of shape ``(batch_size, num_clicked_news)``.

        Returns:
            tuple
                - multi_user_interest: tensor of shape ``(batch_size, num_context_codes, embed_dim)``
                - matching_scores: tensor of shape ``(batch_size, num_candidates)``
        """
        batch_size = title.shape[0]
        num_candidates = title.shape[1]
        his_length = his_title.shape[1]

        # Representation of candidate news
        title = title.view(batch_size * num_candidates, -1)
        title_mask = title_mask.view(batch_size * num_candidates, -1)
        sapo = sapo.view(batch_size * num_candidates, -1)
        sapo_mask = sapo_mask.view(batch_size * num_candidates, -1)

        candidate_repr = self.news_encoder(title_encoding=title, title_attn_mask=title_mask, sapo_encoding=sapo,
                                           sapo_attn_mask=sapo_mask)
        candidate_repr = candidate_repr.view(batch_size, num_candidates, -1)

        # Representation of history clicked news
        his_title = his_title.view(batch_size * his_length, -1)
        his_title_mask = his_title_mask.view(batch_size * his_length, -1)
        his_sapo = his_sapo.view(batch_size * his_length, -1)
        his_sapo_mask = his_sapo_mask.view(batch_size * his_length, -1)

        history_repr = self.news_encoder(title_encoding=his_title, title_attn_mask=his_title_mask,
                                         sapo_encoding=his_sapo, sapo_attn_mask=his_sapo_mask)
        history_repr = history_repr.view(batch_size, his_length, -1)

        if self.use_category_bias:
            his_category_embed = self.category_embedding(his_category)
            his_category_embed = self.category_dropout(his_category_embed)
            candidate_category_embed = self.category_embedding(category)
            candidate_category_embed = self.category_dropout(candidate_category_embed)
            category_bias = pairwise_cosine_similarity(his_category_embed, candidate_category_embed)

            multi_user_interest = self.poly_attn(embeddings=history_repr, attn_mask=his_mask, bias=category_bias)
        else:
            multi_user_interest = self.poly_attn(embeddings=history_repr, attn_mask=his_mask, bias=None)

        # Click predictor
        matching_scores = torch.matmul(candidate_repr, multi_user_interest.permute(0, 2, 1))
        if self.score_type == 'max':
            matching_scores = matching_scores.max(dim=2)[0]
        elif self.score_type == 'mean':
            matching_scores = matching_scores.mean(dim=2)
        elif self.score_type == 'weighted':
            matching_scores = self.target_aware_attn(query=multi_user_interest, key=candidate_repr,
                                                     value=matching_scores)
        else:
            raise ValueError('Invalid method of aggregating matching score')

        return multi_user_interest, matching_scores


