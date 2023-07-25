import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from random import randint
from copy import deepcopy
from torch.autograd import Variable
from collections import OrderedDict



class MAB(nn.Module):
    """Multi-head Attention Block.

    Based on Set Transformer implementation
    (Lee et al. 2019, https://github.com/juho-lee/set_transformer).
    """
    def __init__(
            self, dim_Q, dim_KV, dim_emb, dim_out, c):
        super(MAB, self).__init__()
        num_heads = c.model_num_heads
        ln = True
        rff_depth = c.model_rff_depth
        self.att_score_norm = c.model_att_score_norm
        self.pre_layer_norm = True
        dim_out = dim_emb
        self.num_heads = num_heads
        self.dim_KV = dim_KV
        self.dim_split = dim_emb // num_heads
        self.fc_q = nn.Linear(dim_Q, dim_emb)
        self.fc_k = nn.Linear(dim_KV, dim_emb)
        self.fc_v = nn.Linear(dim_KV, dim_emb)
        self.fc_mix_heads = nn.Linear(dim_emb, dim_out)
        self.fc_res = nn.Linear(dim_Q, dim_out)

        self.ln0 = nn.LayerNorm(dim_Q, eps=c.model_layer_norm_eps)
        self.ln1 = nn.LayerNorm(dim_out, eps=c.model_layer_norm_eps)


        self.hidden_dropout = (
            nn.Dropout(p=c.model_hidden_dropout_prob)
            if c.model_hidden_dropout_prob else None)

        self.att_scores_dropout = (
            nn.Dropout(p=c.model_att_score_dropout_prob)
            if c.model_att_score_dropout_prob else None)

        self.init_rff(dim_out, rff_depth)

    def init_rff(self, dim_out, rff_depth):
        self.rff = [nn.Linear(dim_out, 4 * dim_out), nn.GELU()]

        if self.hidden_dropout is not None:
            self.rff.append(self.hidden_dropout)

        for i in range(rff_depth - 1):
            self.rff += [nn.Linear(4 * dim_out, 4 * dim_out), nn.GELU()]

            if self.hidden_dropout is not None:
                self.rff.append(self.hidden_dropout)

        self.rff += [nn.Linear(4 * dim_out, dim_out)]

        if self.hidden_dropout is not None:
            self.rff.append(self.hidden_dropout)

        self.rff = nn.Sequential(*self.rff)

    def forward(self, X, Y):
        if self.pre_layer_norm and self.ln0 is not None:
            X_multihead = self.ln0(X)
        else:
            X_multihead = X

        Q = self.fc_q(X_multihead)

        if self.fc_res is None:
            X_res = Q
        else:
            X_res = self.fc_res(X)

        K = self.fc_k(Y)
        V = self.fc_v(Y)

        Q_ = torch.cat(Q.split(self.dim_split, 2), 0)
        K_ = torch.cat(K.split(self.dim_split, 2), 0)
        V_ = torch.cat(V.split(self.dim_split, 2), 0)

        A = torch.einsum('ijl,ikl->ijk', Q_, K_)

        if self.att_score_norm == 'softmax':
            A = torch.softmax(A / math.sqrt(self.dim_KV), 2)
        elif self.att_score_norm == 'constant':
            A = A / self.dim_split
        else:
            raise NotImplementedError

        if self.att_scores_dropout is not None:
            A = self.att_scores_dropout(A)
        att_weight=A
        multihead = A.bmm(V_)
        multihead = torch.cat(multihead.split(Q.size(0), 0), 2)



        if self.fc_mix_heads is not None:
            H = self.fc_mix_heads(multihead)
        else:
            H = multihead


        if self.hidden_dropout is not None:
            H = self.hidden_dropout(H)#10,20,192

        Q_out = X_res
        H = H + Q_out

        if not self.pre_layer_norm and self.ln0 is not None:
            H = self.ln0(H)

        if self.pre_layer_norm and self.ln1 is not None:
            H_rff = self.ln1(H)
        else:
            H_rff = H


        expanded_linear_H = self.rff(H_rff)

        expanded_linear_H = H + expanded_linear_H

        if not self.pre_layer_norm and self.ln1 is not None:
            expanded_linear_H = self.ln1(expanded_linear_H)

        return expanded_linear_H,att_weight


class MHSA(nn.Module):
    """
    Multi-head Self-Attention Block.

    Based on implementation from Set Transformer (Lee et al. 2019,
    https://github.com/juho-lee/set_transformer).
    Alterations detailed in MAB method.
    """
    has_inducing_points = False

    def __init__(self, dim_in, dim_emb, dim_out, c):
        super(MHSA, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_emb, dim_out, c)
    def forward(self, X):
        H=self.mab(X, X)
        return H

class Decoder(nn.Module):
    def __init__(self, args, dim_hidden, dim_out):
        super(Decoder, self).__init__()
        self.args=args
        self.dim_hidden=dim_hidden
        self.dim_out=dim_out
        self.pred_y=nn.Linear(self.dim_hidden,1)
    def forward(self, x):
        y = self.pred_y(x)
        if self.args.data_set=='movielens':
            y_pred = F.sigmoid(y)*5
        return y_pred


class Encoder(nn.Module):
    def __init__(self, args,if_dim,uf_dim):
        super(Encoder, self).__init__()
        self.args=args
        self.if_dim=if_dim
        self.uf_dim=uf_dim
        self.dropout_rate = args.dropout_rate
        if self.args.data_set=="movielens":
            self.user_emb=Movie_user(args)
            self.item_emb=Movie_item(args)
            self.embedding_rating = torch.nn.Linear(
            in_features=5,
            out_features=self.args.second_embedding_dim//4,
            bias=False)


    def embedding4movielens(self, x):
        gender_dim=self.user_emb.num_gender
        age_dim=self.user_emb.num_age
        occupation_dim=self.user_emb.num_occupation
        zipcode_dim=self.user_emb.num_zipcode
        uf_dim=self.user_emb.feature_dim
        rate_dim=self.item_emb.num_rate
        genre_dim=self.item_emb.num_genre
        director_dim=self.item_emb.num_director
        actor_dim=self.item_emb.num_actor

        gender_idx = Variable(x[:, 0:gender_dim], requires_grad=False)
        age_idx = Variable(x[:,gender_dim:gender_dim+age_dim], requires_grad=False)
        occupation_idx=Variable(x[:,gender_dim+age_dim:gender_dim+age_dim+occupation_dim], requires_grad=False)
        zipcode_idx=Variable(x[:,gender_dim+age_dim+occupation_dim:gender_dim+age_dim+occupation_dim+zipcode_dim], requires_grad=False)
        rate_idx = Variable(x[:, uf_dim:uf_dim+rate_dim], requires_grad=False)
        genre_idx = Variable(x[:,uf_dim+rate_dim:uf_dim+rate_dim+genre_dim], requires_grad=False)
        director_idx=Variable(x[:,uf_dim+rate_dim+genre_dim:uf_dim+rate_dim+genre_dim+director_dim], requires_grad=False)
        actor_idx=Variable(x[:,uf_dim+rate_dim+genre_dim+director_dim:uf_dim+rate_dim+genre_dim+director_dim+actor_dim], requires_grad=False)

        rating_idx=Variable(x[:,uf_dim+rate_dim+genre_dim+director_dim+actor_dim:uf_dim+rate_dim+genre_dim+director_dim+actor_dim+1], requires_grad=False)
        rating_onehot_idx=torch.zeros(rating_idx.size(0),5)
        for i in range(rating_idx.size(0)):
            if rating_idx[i]!=-1:
                rating_onehot_idx[i][int(rating_idx[i])-1]=1
        rating_onehot_idx=rating_onehot_idx.to(self.args.device)

        user_emb = self.user_emb(gender_idx, age_idx, occupation_idx, zipcode_idx)
        item_emb = self.item_emb(rate_idx, genre_idx, director_idx, actor_idx)
        rating_emb=self.embedding_rating(rating_onehot_idx)

        embed = torch.cat((user_emb, item_emb), 1)
        x_out = torch.cat((embed, rating_emb), 1)

        return x_out


    def forward(self,x_app):
        if self.args.data_set=='movielens':
            x = self.embedding4movielens(x_app)
        return x


class Movie_item(torch.nn.Module):
    def __init__(self, args):
        super(Movie_item, self).__init__()
        self.num_rate = args.num_rate
        self.num_genre = args.num_genre
        self.num_director = args.num_director
        self.num_actor = args.num_actor
        self.feature_dim = self.num_actor+self.num_director+self.num_genre+self.num_rate
        self.embedding_dim = args.second_embedding_dim//4

        self.embedding_rate = torch.nn.Linear(
            in_features=self.num_rate,
            out_features=self.embedding_dim,
            bias=False
        )

        self.embedding_genre = torch.nn.Linear(
            in_features=self.num_genre,
            out_features=self.embedding_dim,
            bias=False
        )

        self.embedding_director = torch.nn.Linear(
            in_features=self.num_director,
            out_features=self.embedding_dim,
            bias=False
        )

        self.embedding_actor = torch.nn.Linear(
            in_features=self.num_actor,
            out_features=self.embedding_dim,
            bias=False
        )

    def forward(self, rate_idx, genre_idx, director_idx, actors_idx, vars=None):
        rate_emb = self.embedding_rate(rate_idx.float())/ torch.sum(rate_idx.float(), 1).view(-1, 1)
        genre_emb = self.embedding_genre(genre_idx.float()) / torch.sum(genre_idx.float(), 1).view(-1, 1)
        director_emb = self.embedding_director(director_idx.float()) / torch.sum(director_idx.float(), 1).view(-1, 1)
        actors_emb = self.embedding_actor(actors_idx.float()) / torch.sum(actors_idx.float(), 1).view(-1, 1)
        return torch.cat((rate_emb, genre_emb, director_emb, actors_emb), 1)





class Movie_user(torch.nn.Module):
    def __init__(self,args):
        super(Movie_user, self).__init__()
        self.num_gender = args.num_gender
        self.num_age = args.num_age
        self.num_occupation = args.num_occupation
        self.num_zipcode = args.num_zipcode
        self.feature_dim= self.num_gender+self.num_age+self.num_occupation+self.num_zipcode
        self.embedding_dim = args.second_embedding_dim//4

        self.embedding_gender = torch.nn.Linear(
            in_features=self.num_gender,
            out_features=self.embedding_dim,
            bias=False
        )

        self.embedding_age = torch.nn.Linear(
            in_features=self.num_age,
            out_features=self.embedding_dim,
            bias=False
        )

        self.embedding_occupation = torch.nn.Linear(
            in_features=self.num_occupation,
            out_features=self.embedding_dim,
            bias=False
        )

        self.embedding_area = torch.nn.Linear(
            in_features=self.num_zipcode,
            out_features=self.embedding_dim,
            bias=False
        )

    def forward(self, gender_idx, age_idx, occupation_idx, area_idx):
        gender_emb = self.embedding_gender(gender_idx.float())/ torch.sum(gender_idx.float(), 1).view(-1, 1)
        age_emb = self.embedding_age(age_idx.float())/ torch.sum(age_idx.float(), 1).view(-1, 1)
        occupation_emb = self.embedding_occupation(occupation_idx.float())/ torch.sum(occupation_idx.float(), 1).view(-1, 1)
        area_emb = self.embedding_area(area_idx.float())/ torch.sum(area_idx.float(), 1).view(-1, 1)
        return torch.cat((gender_emb, age_emb, occupation_emb, area_emb), 1)



class MLP(nn.Module):
    def __init__(self, in_ch, hid_ch, head1_out, head2_out, act_type = "relu"):
        super(MLP, self).__init__()
        self.fc1 = FC(in_ch, hid_ch)
        self.fc2 = FC(hid_ch, head1_out)
        self.act_layer = nn.ReLU()
        self.fc3 = FC(in_ch, hid_ch)
        self.fc4 = FC(hid_ch, head2_out)

    def forward(self, x):
        #head 1
        x_1 = self.fc1(x)
        x_1 = self.act_layer(x_1)
        #head 2
        x_2 = self.fc3(x)
        x_2 = self.act_layer(x_2)
        return self.fc2(x_1), self.fc4(x_2)

class FC(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(FC, self).__init__()
        self.fc = torch.nn.Linear(in_ch, out_ch)

    def forward(self, x):
        return self.fc(x)