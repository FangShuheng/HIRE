from itertools import cycle
import torch
import torch.nn as nn
from model.modules import MHSA
from model.modules import Encoder, Decoder





class HIREModel(nn.Module):
    def __init__(self, args,uf_dim,if_dim):
        super().__init__()
        #Configs
        self.args=args
        self.device = args.device
        self.if_dim=if_dim
        self.uf_dim=uf_dim
        self.stacking_depth = args.model_stacking_depth

        #embedding dimension
        if self.args.data_set=='movielens':
            self.F= args.second_embedding_dim//4
            self.H= len(self.if_dim)+len(self.uf_dim)+1
            self.E = self.F*self.H #E=#H*F
        self.D = self.args.col_num #D

        self.num_heads = args.model_num_heads

        # get model
        self.enc = self.hire()

        self.embedding_dropout = (
            nn.Dropout(p=args.model_hidden_dropout_prob)
            if args.model_hidden_dropout_prob else None)

        layer_norm_dims = [self.E]
        self.embedding_layer_norm = nn.LayerNorm(layer_norm_dims, eps=args.model_layer_norm_eps)

        self.encoder=Encoder(args,self.if_dim,self.uf_dim)
        self.decoder = Decoder(args, self.E, 1)

        clip_value = args.exp_gradient_clipping
        print(f'Clipping gradients to value {clip_value}.')
        for p in self.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value)) 


    def hire(self):
        print('Building Model.')

        row_att_args = {'c': self.args}
        col_att_args = {'c': self.args}
        feat_att_args = {'c': self.args}


        if self.args.ablation=='all':
            att_args = cycle([row_att_args, col_att_args, feat_att_args])
        elif self.args.ablation=='onlyfea':
            att_args = cycle([feat_att_args])
        elif self.args.ablation=='onlyrow':
            att_args = cycle([row_att_args])
        elif self.args.ablation=='onlycol':
            att_args = cycle([col_att_args])
        elif self.args.ablation=='onlycolrow':
            att_args = cycle([col_att_args,row_att_args])
        elif self.args.ablation=='onlycolfea':
            att_args = cycle([col_att_args,feat_att_args])
        elif self.args.ablation=='onlyrowfea':
            att_args = cycle([row_att_args,feat_att_args])
        AttentionBlocks = cycle([MHSA])

        enc = []

        enc = self.build(
            enc, AttentionBlocks, att_args)

        enc = nn.Sequential(*enc)
        return enc

    def build(self, enc, AttentionBlocks, att_args):
        stack = []
        layer_index = 0
        if self.args.ablation=='all':
            while layer_index < self.stacking_depth:
                if layer_index % 3 == 0:
                    # shape (N, M, E)
                    stack.append(next(AttentionBlocks)(
                        self.E, self.E, self.E,
                        **next(att_args)))
                    #Transpose
                    stack.append(ReshapeToTranspose())  #(M, N, E)

                elif layer_index % 3 == 1:
                    stack.append(next(AttentionBlocks)(
                        self.E, self.E, self.E,
                        **next(att_args)))
                    #Transpose
                    stack.append(ReshapeToTranspose())


                else:
                    #(N*M, H, F)
                    stack.append(ReshapeToSplit(self.H))
                    stack.append(next(AttentionBlocks)(
                    self.F, self.F,
                    self.F,
                    **next(att_args)))
                    stack.append(ReshapeToCombine(self.D)) #(N, M, E)
                enc += stack
                stack = []
                layer_index += 1

        elif self.args.ablation=='onlyfea':
            while layer_index < self.stacking_depth:
                stack.append(ReshapeToSplit(self.H))
                stack.append(next(AttentionBlocks)(
                self.F, self.F,
                self.F,
                **next(att_args)))
                stack.append(ReshapeToCombine(self.D))
                enc += stack
                stack = []
                layer_index += 1

        elif self.args.ablation=='onlyrowfea':
            stack.append(ReshapeToTranspose())
            while layer_index < self.stacking_depth:
                if layer_index % 2 == 0:
                    stack.append(next(AttentionBlocks)(
                        self.E, self.E, self.E,
                        **next(att_args)))
                if layer_index % 2 ==1:
                    stack.append(ReshapeToSplit(self.H))
                    stack.append(next(AttentionBlocks)(
                    self.F, self.F,
                    self.F,
                    **next(att_args)))
                    stack.append(ReshapeToCombine(self.D))
                enc += stack
                stack = []
                layer_index += 1

        elif self.args.ablation=='onlycolrow':
            while layer_index < self.stacking_depth:
                if layer_index % 2 == 0:
                    stack.append(next(AttentionBlocks)(
                        self.E, self.E, self.E,
                        **next(att_args)))
                    stack.append(ReshapeToTranspose())

                elif layer_index % 2 == 1:
                    stack.append(next(AttentionBlocks)(
                        self.E, self.E, self.E,
                        **next(att_args)))
                    stack.append(ReshapeToTranspose())
                enc += stack
                stack = []
                layer_index += 1

        elif self.args.ablation=='onlycolfea':
            while layer_index < self.stacking_depth:
                if layer_index % 2 == 0:
                    stack.append(next(AttentionBlocks)(
                        self.E, self.E, self.E,
                        **next(att_args)))

                if layer_index % 2 ==1:
                    stack.append(ReshapeToSplit(self.H))
                    stack.append(next(AttentionBlocks)(
                    self.F, self.F,
                    self.F,
                    **next(att_args)))
                    stack.append(ReshapeToCombine(self.D))
                enc += stack
                stack = []
                layer_index += 1

        elif self.args.ablation=='onlycol':
             while layer_index < self.stacking_depth:
                stack.append(next(AttentionBlocks)(
                        self.E, self.E, self.E,
                        **next(att_args)))
                enc += stack
                stack = []
                layer_index += 1

        elif self.args.ablation=='onlyrow':
            stack.append(ReshapeToTranspose())
            while layer_index < self.stacking_depth:
                stack.append(next(AttentionBlocks)(
                        self.E, self.E, self.E,
                        **next(att_args)))
                enc += stack
                stack = []
                layer_index += 1

        enc.append(ReshapeToCombine(self.D))

        return enc

    def forward(self, X_ragged, pretrain=False):
        X_ragged=self.encoder(X_ragged)
        X = X_ragged.reshape(-1, self.D, self.E)

        # Embedding tensor currently has shape (N x D x E)
        if self.embedding_layer_norm is not None:
            X = self.embedding_layer_norm(X)

        if self.embedding_dropout is not None:
            X = self.embedding_dropout(X)

        X = self.enc(X)

        if self.embedding_dropout is not None:
            X = self.embedding_dropout(X)

        X = X.reshape(-1, self.D, self.E)

        X_out=self.decoder(X)
        if self.args.data_set=='movielens':
            X_out=torch.clamp(X_out,0,5)
        return X_out




class ReshapeToTranspose(nn.Module):
    """Reshapes a tensor of shape (N, D, E) to (D, N, E)."""
    def __init__(self):
        super(ReshapeToTranspose, self).__init__()

    def forward(self, X):
        return X.permute(1,0,2)

class ReshapeToSplitCombine(nn.Module):
    def __init__(self, H):
        super(ReshapeToSplit, self).__init__()
        self.H = H
    """Reshapes a tensor of shape (N, D, E) to (N*D, H, F). E=H*F"""
    def split(self, X):
        self.N=X.size(0)
        X=X.reshape(-1, X.size(2)) #to (N*D,E)
        return X.reshape(X.size(0), self.H, -1)
    """Reshapes a tensor of shape (N*D, H, F) to (N, D, E). E=H*F"""
    def combine(self, X):
        X=X.reshape(X.size(0), -1)
        return X.reshape(self.N, -1, X.size(1))

class ReshapeToSplit(nn.Module):
    """Reshapes a tensor of shape (N, D, E) to (N*D, H, F). E=H*F"""
    def __init__(self, H):
        super(ReshapeToSplit, self).__init__()
        self.H = H
    def forward(self, X):
        X=X.reshape(-1, X.size(2)) #to (N*D,E)
        return X.reshape(X.size(0), self.H, -1)

class ReshapeToCombine(nn.Module):
    """Reshapes a tensor of shape (N*D, H, F) to (N, D, E). E=H*F, or shape(N,D,E) to (N,D,E)"""
    def __init__(self,D):
        super(ReshapeToCombine, self).__init__()
        self.D=D
    def forward(self, X):
        if X.size(1)!=self.D:
            N=int(X.size(0)/self.D)
        else:
            N=X.size(0)
        X=X.reshape(X.size(0), -1)
        return X.reshape(N, self.D, -1)


class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print('Debug', x.shape)
        return x
