from json import load
import torch
import torch.nn as nn
from torch import optim
import random
import torch.nn.functional as F
import numpy as np
import argparse
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from sklearn.metrics import precision_score, f1_score, accuracy_score, recall_score
import math
import os
from scorer import *
from utils.optim_utils import Lookahead, Lamb
from utils.optim import LRScheduler
from utils.optim import TradeoffAnnealer
import time
from data_loader import init_data
torch.manual_seed(200)
torch.cuda.manual_seed_all(200)

def MSEloss( p_y_pred, y_target, ymask):
    regression_loss = F.mse_loss(p_y_pred.view(-1, 1), y_target.view(-1, 1), reduction='none')
    if ymask is not None:
        regression_loss = torch.mul(regression_loss, ymask.view(-1,1))
    loss=torch.sum(regression_loss)/len(regression_loss)
    return loss

def smoothL1loss(p_y_pred,y_target,ymask):
    criterion=torch.nn.SmoothL1Loss(reduction='none')
    loss=criterion(p_y_pred.view(-1, 1), y_target.view(-1, 1))
    if ymask is not None:
        loss = torch.mul(loss, ymask.view(-1,1))
    loss=torch.sum(loss)/len(loss)
    return loss

def evaluate_prediction(pred, targets):
    acc = accuracy_score(targets, pred)
    precision = precision_score(targets, pred)
    recall = recall_score(targets, pred)
    f1 = f1_score(targets, pred)
    return acc, precision, recall, f1

class Recommendation(nn.Module):
    def __init__(self, args, hiremodel):
        super(Recommendation, self).__init__()
        self.args = args
        self.hiremodel = hiremodel
        self.batch_size=self.args.batch_size

        self.scaler = GradScaler()

        if 'default' in self.args.exp_optimizer:
            self.optimizer=optim.Adam(self.hiremodel.parameters(), lr=self.args.exp_lr)
        elif 'lamb' in self.args.exp_optimizer:
            lamb = Lamb
            self.optimizer = lamb(self.hiremodel.parameters(), lr=self.args.exp_lr, betas=(0.9, 0.999),weight_decay=self.args.exp_weight_decay, eps=1e-6)
        else:
            raise NotImplementedError
        if self.args.exp_optimizer.startswith('lookahead_'):
            self.optimizer = Lookahead(self.optimizer, k=self.args.exp_lookahead_update_cadence)

        self.scheduler=LRScheduler(c=self.args, name=self.args.exp_scheduler,optimizer=self.optimizer)
        if self.args.exp_tradeoff != -1:
            self.tradeoff_annealer = TradeoffAnnealer(
                c=self.args, num_steps=0)
        else:
            self.tradeoff_annealer = None

        if self.args.cuda:
            self.hiremodel.to(self.args.device)



    def train(self,train_data,val_data,test_data):
        for epoch in range(self.args.epochs):
            self.valid(val_data,epoch=epoch)
            torch.manual_seed(self.args.seed)
            torch.cuda.manual_seed_all(self.args.seed)
            np.random.seed(self.args.seed)
            random.seed(self.args.seed)
            torch.cuda.empty_cache()
            self.hiremodel.train()
            total= list()
            for batch_idx, batch_data in enumerate(train_data):
                x_batch=batch_data[0]
                y_batch=batch_data[1]
                y_mask_batch=batch_data[2]
                for i in range(x_batch.shape[0]):
                    y=y_batch[i,:,:].to(self.args.device)# #user,#item
                    y=y.squeeze(0)
                    num_user=y.size(0)
                    y_mask=y_mask_batch[i,:,:].to(self.args.device)# #user,#item
                    y_mask=y_mask.squeeze(0)
                    x=x_batch[i,:,:].to(self.args.device)
                    x=x.squeeze(0)
                    z=self.hiremodel(x) # #user,#item,1/#user,#item,2
                    if self.args.data_set=='movielens':
                        z=z.squeeze() # #user,#item
                        ymask_temp=torch.ge(y_mask,0.5)
                        temp_list=torch.masked_select(z,ymask_temp)
                        targets=torch.masked_select(y,ymask_temp)
                        loss=F.mse_loss(temp_list.view(-1, 1), targets.view(-1, 1))
                    self.scaler.scale(loss).backward()
                    total.append(loss.item())
                    output_list, recommendation_list=temp_list.sort(descending=True)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.tradeoff_annealer is not None:
                    self.tradeoff_annealer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            total_loss = sum(total)/len(total)


    def test(self,test_data,epoch=None):
        self.hiremodel.eval()
        pre5 = []
        ap5 = []
        ndcg5 = []
        pre7 = []
        ap7 = []
        ndcg7 = []
        pre10 = []
        ap10 = []
        ndcg10 = []
        total=[]
        run_time=0
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        for x_all, y_all, y_mask_all in test_data:
            torch.cuda.empty_cache()
            y_all=y_all.squeeze(0).to(self.args.device)# #user,#item
            y_mask_all=y_mask_all.squeeze(0).to(self.args.device)# #user,#item
            x_all=x_all.squeeze(0).to(self.args.device)
            t_start=time.time()
            z_all=self.hiremodel(x_all) # #user,#item,1
            t_end=time.time()
            run_time=run_time+t_end-t_start
            if self.args.data_set=='movielens':
                z_all=torch.squeeze(z_all,2) # #user,#item
                for i in range(z_all.shape[0]):
                    z=z_all[i,:].view(-1)
                    y=y_all[i,:].view(-1)
                    ymask_temp=y_mask_all[i,:].view(-1).ge(0.5)
                    temp_list=torch.masked_select(z,ymask_temp)
                    targets=torch.masked_select(y,ymask_temp)
                    loss=F.mse_loss(temp_list.view(-1, 1), targets.view(-1, 1))
                    total.append(loss.item())
                    output_list, recommendation_list=temp_list.sort(descending=True)
                    random.shuffle(recommendation_list)
                    add_metric(self.args,recommendation_list, targets.cpu().detach().numpy(), pre5, ap5, ndcg5, 5)
                    add_metric(self.args,recommendation_list, targets.cpu().detach().numpy(), pre7, ap7, ndcg7, 7)
                    add_metric(self.args,recommendation_list, targets.cpu().detach().numpy(), pre10, ap10, ndcg10, 10)
        mpre5, mndcg5, map5 = cal_metric(pre5, ap5, ndcg5)
        mpre7, mndcg7, map7 = cal_metric(pre7, ap7, ndcg7)
        mpre10, mndcg10, map10 = cal_metric(pre10, ap10, ndcg10)
        total_loss=sum(total)/len(total)
        print("Test Result {}-th Epoch\t test_loss:{:.6f}\t  TOP-5 {:.4f}\t{:.4f}\t{:.4f}\t TOP-7: {:.4f}\t{:.4f}\t{:.4f}\t TOP-10: {:.4f}\t{:.4f}\t{:.4f}".format(epoch, total_loss, mpre5, mndcg5, map5, mpre7, mndcg7, map7, mpre10, mndcg10, map10))
        print("Test time {}-th Epoch\t {:.4f}".format(epoch,run_time))


    def valid(self,val_data,epoch=None):
        self.hiremodel.eval()
        pre5 = []
        ap5 = []
        ndcg5 = []
        pre7 = []
        ap7 = []
        ndcg7 = []
        pre10 = []
        ap10 = []
        ndcg10 = []
        total=[]
        torch.manual_seed(self.args.seed+1)
        torch.cuda.manual_seed_all(self.args.seed+1)
        np.random.seed(self.args.seed+1)
        random.seed(self.args.seed+1)
        for x_all, y_all, y_mask_all in val_data:
            torch.cuda.empty_cache()
            y_all=y_all.squeeze(0).to(self.args.device)# #user,#item
            y_mask_all=y_mask_all.squeeze(0).to(self.args.device)# #user,#item
            x_all=x_all.squeeze(0).to(self.args.device)
            z_all=self.hiremodel(x_all) # #user,#item,1
            if self.args.data_set=='movielens':
                z_all=torch.squeeze(z_all,2) # #user,#item
                for i in range(z_all.shape[0]):
                    z=z_all[i,:].view(-1)
                    y=y_all[i,:].view(-1)
                    ymask_temp=y_mask_all[i,:].view(-1).ge(0.5)
                    temp_list=torch.masked_select(z,ymask_temp)
                    targets=torch.masked_select(y,ymask_temp)
                    loss=F.mse_loss(temp_list.view(-1, 1), targets.view(-1, 1))
                    total.append(loss.item())
                    output_list, recommendation_list=temp_list.sort(descending=True)
                    add_metric(self.args,recommendation_list, targets.cpu().detach().numpy(), pre5, ap5, ndcg5, 5)
                    add_metric(self.args,recommendation_list, targets.cpu().detach().numpy(), pre7, ap7, ndcg7, 7)
                    add_metric(self.args,recommendation_list, targets.cpu().detach().numpy(), pre10, ap10, ndcg10, 10)

        mpre5, mndcg5, map5 = cal_metric(pre5, ap5, ndcg5)
        mpre7, mndcg7, map7 = cal_metric(pre7, ap7, ndcg7)
        mpre10, mndcg10, map10 = cal_metric(pre10, ap10, ndcg10)
        total_loss=sum(total)/len(total)


