# -*- coding: utf-8 -*-
from ast import operator
import enum
import json
import random
import torch
import numpy as np
import pickle
import codecs
import re
import os
import datetime
import tqdm
import pandas as pd
from torch.utils.data import DataLoader,Dataset


import operator


def count_values(dict):
    count_val = 0
    for key, value in dict.items():
        count_val += len(value)
    return count_val


class init_data:
    def __init__(self,args):
        self.args=args
        self.data_set=args.data_set
        if self.data_set=='movielens':
            self.get_data_movielens(self.args)

    def get_data_movielens(self,args):
        dataset_path = "/home/XXX/XXX/data/movielens/ml-1m"

        #load list
        rate_list = load_list("{}/m_rate.txt".format(dataset_path))
        genre_list = load_list("{}/m_genre.txt".format(dataset_path))
        actor_list = load_list("{}/m_actor.txt".format(dataset_path))
        director_list = load_list("{}/m_director.txt".format(dataset_path))
        gender_list = load_list("{}/m_gender.txt".format(dataset_path))
        age_list = load_list("{}/m_age.txt".format(dataset_path))
        occupation_list = load_list("{}/m_occupation.txt".format(dataset_path))
        zipcode_list = load_list("{}/m_zipcode.txt".format(dataset_path))

        #load data
        user_data_path = "{}/users.dat".format(dataset_path)
        score_data_path = "{}/ratings.dat".format(dataset_path)
        item_data_path = "{}/movies_extrainfos.dat".format(dataset_path)
        user_data = pd.read_csv(
            user_data_path, names=['user_id', 'gender', 'age', 'occupation_code', 'zip'],
            sep="::", engine='python'
        )
        item_data = pd.read_csv(
            item_data_path, names=['movie_id', 'title', 'year', 'rate', 'released', 'genre', 'director', 'writer', 'actors', 'plot', 'poster'],
            sep="::", engine='python', encoding="utf-8"
        )
        score_data = pd.read_csv(
            score_data_path, names=['user_id', 'movie_id', 'rating', 'timestamp'],
            sep="::", engine='python'
        )
        score_data['time'] = score_data["timestamp"].map(lambda x: datetime.datetime.fromtimestamp(x))
        score_data = score_data.drop(["timestamp"], axis=1)

        # hashmap for item information
        if not os.path.exists("{}/m_movie_dict.pkl".format(dataset_path)):
            self.movie_dict = {}
            for idx, row in item_data.iterrows():
                m_info = item_converting(row, rate_list, genre_list, director_list, actor_list)
                self.movie_dict[row['movie_id']] = m_info
            pickle.dump(self.movie_dict, open("{}/m_movie_dict.pkl".format(dataset_path), "wb"))
        else:
            self.movie_dict = pickle.load(open("{}/m_movie_dict.pkl".format(dataset_path), "rb")) #3881 movie; 10242 each dimension

        # hashmap for user profile
        if not os.path.exists("{}/m_user_dict.pkl".format(dataset_path)):
            self.user_dict = {}
            for idx, row in user_data.iterrows():
                u_info = user_converting(row, gender_list, age_list, occupation_list, zipcode_list)
                self.user_dict[row['user_id']] = u_info
            pickle.dump(self.user_dict, open("{}/m_user_dict.pkl".format(dataset_path), "wb"))
        else:
            self.user_dict = pickle.load(open("{}/m_user_dict.pkl".format(dataset_path), "rb"))

    def get_data(self, args, mode):
        if args.data_set=='movielens':
            if mode=='train':
                TrainDataset=movielensDataset(args, args.task_num, args.train_states,self.user_dict, self.movie_dict,'train')
                train_data=DataLoader(TrainDataset, batch_size = self.args.batch_size, shuffle = True, num_workers=16)
                return train_data
            elif mode=='valid':
                ValidDataset=movielensDataset(args, args.val_task_num, args.train_states,self.user_dict, self.movie_dict,'valid')
                valid_data=DataLoader(ValidDataset, batch_size = 1, shuffle = False, num_workers=16)
                return valid_data
            elif mode=='test':
                TestDataset=movielensDataset(args, args.test_task_num, args.states, self.user_dict, self.movie_dict,'test')
                test_data=DataLoader(TestDataset, batch_size = 1, shuffle = False, num_workers=16)
                return test_data


    def get_dim(self):
        if self.args.data_set=='movielens':
            u_dims=[self.args.num_gender,self.args.num_age,self.args.num_occupation,self.args.num_zipcode]
            i_dims=[self.args.num_rate,self.args.num_genre,self.args.num_director,self.args.num_actor]
            return u_dims,i_dims


class movielensDataset(Dataset):
    def __init__(self, args, task_num, state, user_dict, movie_dict, mode): #states ["warm_state", "user_cold_state", "item_cold_state", "user_and_item_cold_state"]
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        self.args=args
        self.batch_size=args.batch_size
        self.user_dict=user_dict
        self.movie_dict=movie_dict
        self.state=state
        self.mode=mode

        dataset_path = "/home/XXX/XXX/data/movielens/ml-1m"
        with open("{}/{}.json".format(dataset_path, self.state), encoding="utf-8") as f:
            self.dataset_x = json.loads(f.read()) #key:user; value:item (str)
        with open("{}/{}_y.json".format(dataset_path, self.state), encoding="utf-8") as f:
            self.dataset_y = json.loads(f.read())

        self.task_num=task_num

        if self.mode=='train':
            self.user_id=[]
            for _, user_id in enumerate(self.dataset_x.keys()):
                if _%8!=0:
                    self.user_id.append(user_id)
        elif self.mode=='valid':
            self.user_id=[]
            for _, user_id in enumerate(self.dataset_x.keys()):
                if _%8==0:
                    self.user_id.append(user_id)
        elif self.mode=='test':
            self.user_id=[]
            for _, user_id in enumerate(self.dataset_x.keys()):
                self.user_id.append(user_id)
        self.item_id=[]
        for _,item_id in enumerate(self.movie_dict.keys()): #item_id int
            self.item_id.append(str(item_id))
        self.index=list(range(self.args.col_num))
        random.shuffle(self.index)
        np.random.shuffle(self.user_id)


    def __len__(self):
        if self.mode=='train' or self.mode=='valid':
            return self.task_num
        elif self.mode=='test':
            return len(self.user_id)

    def __getitem__(self, index):
        user_list=[]
        item_list=[]
        if self.mode=='train' or self.mode=='valid':
            target_user_idx=random.choice(self.user_id)
            user_list.append(target_user_idx)
        elif self.mode=='test':
            target_user_idx=self.user_id[index]
            user_list.append(target_user_idx)
        if len(self.dataset_x[target_user_idx])<int(self.args.col_num):
            item_list=np.random.choice(self.dataset_x[target_user_idx], size=len(self.dataset_x[target_user_idx]),replace=False)
            item_list=item_list.tolist()
            while len(item_list)<int(self.args.col_num):
                i=random.choice(self.item_id)
                if i not in item_list:
                    item_list.append(i)
        else:
            item_list=np.random.choice(self.dataset_x[target_user_idx], size=int(self.args.col_num),replace=False)
            item_list=item_list.tolist()
        assert len(item_list)==self.args.col_num

        count={}
        for u,user in enumerate(self.user_id):
            count[user]=0
            for i,item in enumerate(item_list):
                if item in self.dataset_x[user]:
                    count[user]=count[user]+1
        c_sort=sorted(count.items(),key=operator.itemgetter(1),reverse=True)
        c_random=random.sample(c_sort[0:self.args.row_num*2], self.args.row_num)
        for i,tup in enumerate(c_random):
            if len(user_list)>=self.args.row_num:
                break
            if tup[0] not in user_list:
                user_list.append(tup[0])



        y_sup=torch.zeros(self.args.row_num,self.args.col_num)
        y_mask=torch.zeros(self.args.row_num,self.args.col_num)
        y_true=torch.zeros(self.args.row_num,self.args.col_num)
        x_rating=torch.zeros(self.args.row_num,self.args.col_num)
        for u,user in enumerate(user_list):
            for i,item in enumerate(item_list):
                if item in self.dataset_x[user]:
                    item_idx=self.dataset_x[user].index(item)
                    y_true[u][i]=self.dataset_y[user][item_idx]
                    #split support and query
                    if i in self.index[:int(self.args.support_ratio*self.args.col_num)]:
                        y_sup[u][i]=1
                        x_rating[u][i]=self.dataset_y[user][item_idx]
                    else:
                        x_rating[u][i]=-1
                    y_mask[u][i]=1
                #no interaction in reality
                else:
                    y_true[u][i]=-1
                    x_rating[u][i]=-1

        #item embedding
        total_item=[]
        for i,item in enumerate(item_list):
            item_temp=self.movie_dict[int(item)]
            try:
                total_item = torch.cat((total_item, item_temp),0)
            except:
                total_item=item_temp

        for u,u_idx in enumerate(user_list):
            x_row=x_rating[u]
            x_row.t()
            user_temp=self.user_dict[int(u_idx)]
            x_app=user_temp.repeat_interleave(self.args.col_num, dim=0) #20,3432
            x_app=torch.cat((x_app, total_item), 1) #total item 4,10247
            try:
                x = torch.cat((x, x_app), 0)
            except:
                x = x_app
            try:
                x_=torch.cat((x_,x_row),0)
            except:
                x_=x_row

        x=torch.cat((x,x_.view(-1,1)),1)
        return (x, y_true, y_mask)


def load_list(fname):
    list_ = []
    with open(fname, encoding="utf-8") as f:
        for line in f.readlines():
            list_.append(line.strip())
    return list_

def item_converting(row, rate_list, genre_list, director_list, actor_list):
    #rate_idx = torch.tensor([[rate_list.index(str(row['rate']))]]).long()
    rate_idx = torch.zeros(1,6).long()
    for rate in str(row['rate']).split(", "):
        idx=rate_list.index(rate)
        rate_idx[0, idx]=1
    genre_idx = torch.zeros(1, 25).long() #genre: one hot
    for genre in str(row['genre']).split(", "):
        idx = genre_list.index(genre)
        genre_idx[0, idx] = 1
    director_idx = torch.zeros(1, 2186).long() #director: one hot
    for director in str(row['director']).split(", "):
        idx = director_list.index(re.sub(r'\([^()]*\)', '', director))
        director_idx[0, idx] = 1
    actor_idx = torch.zeros(1, 8030).long() #actor: one hot
    for actor in str(row['actors']).split(", "):
        idx = actor_list.index(actor)
        actor_idx[0, idx] = 1
    return torch.cat((rate_idx, genre_idx, director_idx, actor_idx), 1)

def user_converting(row, gender_list, age_list, occupation_list, zipcode_list):
    gender_idx=torch.zeros(1,2).long()
    for gender in str(row['gender']).split(","):
        idx=gender_list.index(gender)
        gender_idx[0,idx]=1
    age_idx=torch.zeros(1,7).long()
    for age in str(row['age']).split(","):
        idx=age_list.index(age)
        age_idx[0,idx]=1
    occupation_idx=torch.zeros(1,21).long()
    for occupation in str(row['occupation_code']).split(","):
        idx=occupation_list.index(occupation)
        occupation_idx[0,idx]=1
    zip_idx=torch.zeros(1,3402).long()
    for zip in str(row['zip']).split(","):
        zip=zip[0:5]
        idx=zipcode_list.index(zip)
        zip_idx[0,idx]=1
    return torch.cat((gender_idx, age_idx, occupation_idx, zip_idx), 1)


