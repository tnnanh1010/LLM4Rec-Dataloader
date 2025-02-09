import torch
from torch import nn
import torch.nn.functional as F
from layer import AttentionPooling

class NewsEncoder(nn.Module):
    def __init__(self, args, embedding_matrix, num_categories, num_subcategories):
        super(NewsEncoder, self).__init__()
        self.embedding_matrix = embedding_matrix
        self.drop_rate = args.drop_rate
        self.num_words_title = args.num_words_title
        self.use_category = args.use_category
        self.use_subcategory = args.use_subcategory

        # Title Encoder
        self.cnn = nn.Conv1d(
            in_channels=args.word_embedding_dim,
            out_channels=args.news_dim,
            kernel_size=3,
            padding=1
        )
        self.attn = AttentionPooling(args.news_dim, args.news_query_vector_dim)

        # Topic Encoder
        if self.use_category:
            self.category_embedding = nn.Embedding(num_categories + 1, args.news_dim, padding_idx=0)
        if self.use_subcategory:
            self.subcategory_embedding = nn.Embedding(num_subcategories + 1, args.news_dim, padding_idx=0)
        
        self.final_dense = nn.Linear(
            args.news_dim + (args.category_emb_dim if self.use_category else 0) + (args.category_emb_dim if self.use_subcategory else 0),
            args.news_dim
        )

    def forward(self, x, mask=None):
        '''
            x: batch_size, word_num
            category_encoding: batch_size
            subcategory_encoding: batch_size
            mask: batch_size, word_num
        '''
        # Title Encoder
        title = torch.narrow(x, -1, 0, self.num_words_title).long()
        word_vecs = F.dropout(self.embedding_matrix(title),
                              p=self.drop_rate,
                              training=self.training)
        context_word_vecs = self.cnn(word_vecs.transpose(1, 2)).transpose(1, 2)
        title_vecs = self.attn(context_word_vecs, mask)

        '''
        if self.use_abstract:
            abstract = torch.narrow(x, -1, 0, self.num_words_abstract).long()
            abstract_word_vecs = F.dropout(self.embedding_matrix(abstract),
                                           p=self.drop_rate,
                                           training=self.training)
            abstract_context_word_vecs = self.abstract_cnn(abstract_word_vecs.transpose(1, 2)).transpose(1, 2)
            abstract_vecs = self.abstract_attn(abstract_context_word_vecs, mask)
            news_info.append(abstract_vecs)
        '''
        # Topic Encoder
        news_info = [title_vecs]
        if self.use_category:
            category = torch.narrow(x, -1, self.num_words_title, 1).squeeze(dim=-1).long()
            category_embed = self.category_embedding(category)
            news_info.append(category_embed)
        if self.use_subcategory:
            subcategory = torch.narrow(x, -1, self.num_words_title + 1, 1).squeeze(dim=-1).long()
            subcategory_embed = self.subcategory_embedding(subcategory)
            news_info.append(subcategory_embed)

        news_repr = torch.cat(news_info, dim=1)
        
        news_vec = self.final_dense(news_repr)

        return news_vec

class UserEncoder(nn.Module):
    def __init__(self, args, num_users):
        super(UserEncoder, self).__init__()
        self.user_embedding = nn.Embedding(num_users, args.word_embedding_dim)
        self.long_term_dropout = nn.Dropout(args.long_term_dropout)
        self.num_gru_layers = args.num_gru_layers

        # GRU for short-term representation
        self.short_term_gru = nn.GRU(args.news_dim, args.word_embedding_dim, num_layers=args.num_gru_layers, 
                          batch_first=True, dropout=args.gru_dropout if args.num_gru_layer > 1 else 0)


        self.combine_type = args.combine_type
        if self.combine_type == 'con':
            self.fc = nn.Linear(2 * args.word_embedding_dim, args.word_embedding_dim)

    def forward(self, user_id, news_history):
        """
        Parameters:
        - user_id: Tensor of user IDs (batch_size)
        - news_history: Tensor of news embeddings (batch_size, history_length, news_dim)
        
        Returns:
        - user_representation: Final user representation (batch_size, embedding_dim)
        """
        # Long-term representation with dropout
        long_term_user_rep = self.user_embedding(user_id)
        long_term_user_rep = self.long_term_dropout(long_term_user_rep)

        if self.combine_type == "ini":
            # Initialize GRU hidden state with long-term representation
            hidden_state = long_term_user_rep.unsqueeze(0).repeat(self.gru.num_layers, 1, 1)  # (num_layers, batch_size, embedding_dim)
            _, short_term_user_rep = self.gru(news_history, hidden_state)
        elif self.combine_type == "con":
            # Use default GRU initialization for hidden state
            _, short_term_user_rep = self.gru(news_history)
            short_term_user_rep = short_term_user_rep[-1]  # Use last layer's hidden state (batch_size, embedding_dim)

            # Concatenate short-term and long-term representations
            combined_rep = torch.cat((short_term_user_rep, long_term_user_rep), dim=1)  # (batch_size, 2 * embedding_dim)
            user_representation = self.fc(combined_rep)  # (batch_size, embedding_dim)
            return user_representation
        return short_term_user_rep.squeeze(0)

class LSTURModel(nn.Module):
    def __init__(self, args, embedding_matrix, num_users, num_categories, num_subcategories):
        super(LSTURModel, self).__init__()
        pretrained_word_embedding = torch.from_numpy(embedding_matrix).float()
        word_embedding = nn.Embedding.from_pretrained(pretrained_word_embedding,
                                                      freeze=args.freeze_embedding,
                                                      padding_idx=0)

        self.news_encoder = NewsEncoder(args, word_embedding, num_categories, num_subcategories)
        self.user_encoder = UserEncoder(args, num_users)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, user_id, history, history_mask, candidate, label):
        '''
            user_id: batch_size
            history: batch_size, history_length, num_word_title
            history_mask: batch_size, history_length
            candidate: batch_size, 1+K, num_word_title
            label: batch_size, 1+K
            category_encoding: batch_size
            subcategory_encoding: batch_size
        '''
        num_words = history.shape[-1]
        candidate_news = candidate.reshape(-1, num_words)
        candidate_news_vecs = self.news_encoder(candidate_news, history_mask).reshape(-1, 1 + self.args.npratio, self.args.news_dim)

        history_news = history.reshape(-1, num_words)
        history_news_vecs = self.news_encoder(history_news).reshape(-1, self.args.user_log_length, self.args.news_dim)

        user_vec = self.user_encoder(user_id, history_news_vecs)
        score = torch.bmm(candidate_news_vecs, user_vec.unsqueeze(dim=-1)).squeeze(dim=-1)
        loss = self.loss_fn(score, label)
        return loss, score