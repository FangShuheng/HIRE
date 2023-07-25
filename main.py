# -*- coding: utf-8 -*-
from cProfile import run
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
import torch
import time
import random
import os
from get_args import get_args
from train_eval import Recommendation
from model.HIRE import HIREModel
from data_loader import init_data
import wandb
torch.manual_seed(200)
torch.cuda.manual_seed_all(200)
np.random.seed(200)
random.seed(200)



def main(args):
    #print(torch.__version__)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    run_HIRE(args)


def run_HIRE(args):
    datainitial = init_data(args)
    u_dim,i_dim=datainitial.get_dim()
    print("Create model HIRE...")
    HIREmodel=HIREModel(args,u_dim,i_dim)
    recom= Recommendation(args, HIREmodel)

    train_data=datainitial.get_data(args, 'train')
    val_data=datainitial.get_data(args, 'valid')
    test_data=datainitial.get_data(args,'test')
    print('------train phase-----')
    recom.train(train_data, val_data, test_data)

    print('------test phase-----')
    recom.test(test_data,args.epochs)




if __name__ == "__main__":
    args = get_args()
    print(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    main(args)