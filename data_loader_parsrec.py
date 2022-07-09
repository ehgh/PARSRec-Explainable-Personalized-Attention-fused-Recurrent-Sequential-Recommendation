# Copyright (c) Ehsan G., Mohammad M., Ashwin A.
#
#        _____     _____     _____     _____    _____     _____    _____                                           
#       |  _  |   |  _  |   |  _  |   |  ___   |  _  |   |  ___   |  ___
#       | |_| |   | |_| |   | |_| |   | |___   | |_| |   | |__    | |                                             
#       |   __|   |  _  |   |  _  /   |___  |  |  _  /   | |___   | |___                                                      
#       |  |      | | | |   | | \ \    ___| |  | | \ \   |_____   |_____                                           
# 
# This source code is licensed under the MIT license found in the
#
# Description: an implementation of PARSRec dataloaders
# 
# References:
# Ehsan Gholami, Mohammad Motamedi, and Ashwin Aravindakshan. 2022.
# PARSRec: Explainable Personalized Attention-fused Recurrent Sequential
# Recommendation Using Session Partial Actions. In Proceedings of the 28th
# ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD
# ’22), August 14–18, 2022, Washington, DC, USA. ACM, New York, NY, USA,
# 11 pages. https://doi.org/10.1145/3534678.3539432


import os
import numpy as np
from torch.utils.data import Dataset
import torch
import math
from utils import load_pickle
from numpy.lib.recfunctions import structured_to_unstructured as st_to_unst
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

def parsrec_collate_fn(instance_list):
    '''
    After fetching a list of samples using the indices from batchsampler
    this function is used to collate lists of samples into batches.
    '''
    batch_size = len(instance_list)
    instance_batch = list(zip(*instance_list))
    #+1 for user historical actions
    n_cat = instance_batch[1][0].size(0) + 1
    #categorical features and offsets for embeddingbag
    x_cat_batch = [i for i in torch.hstack(instance_batch[1])]
    x_cat_offset = torch.arange(batch_size).repeat(n_cat, 1)

    ##history will be vector index with possible repeated ids along with offset for each instance
    x_cat_offset[-1] = torch.cumsum(torch.tensor([0]+[len(hist) for hist in instance_batch[2][:-1]]), dim=0)
    x_cat_batch.append(torch.vstack(instance_batch[2]).T[0])
        
    #max session length in the batch
    max_session_sz = max(map(len,instance_batch[3]))
    #pad all sessions within batch by EOB to the max session length
    y_batch = torch.hstack([F.pad(input=b, pad=(0, 0, 0,max_session_sz-b.shape[0]), 
        mode='constant', value=0) for b in instance_batch[3]]).T
    
    return x_cat_offset, x_cat_batch, y_batch

class ParsrecBinDataset(Dataset):
    """
    Binary version of dataset.
    """

    def __init__(self, data_file,
                 batch_size=1, fields_file='', num_dense_fea=0, num_sparse_fea=0,
                 session_length_file='', session_file='', 
                 history_size_file='', history_file='',
                 item_dtype=torch.int16, user_dtype=torch.int32):
        
        #dataset fields and variable datatypes (for various precisions like fp32, int16, ...)
        self._fields = load_pickle(fields_file)
        self.num_entries = self._fields.pop('num_entries')
        self._session_dtype = self._fields.pop('session_items')
        self.item_dtype = item_dtype
        self.user_dtype = user_dtype

        #length of sessions and user historical actions
        self.idx_per_entry_session = np.insert(np.load(session_length_file).cumsum(dtype=np.int32), 0, 0)
        self.idx_per_entry_history = np.insert(np.load(history_size_file).cumsum(dtype=np.int32), 0, 0)
        
        self.batch_size = batch_size
        
        # dataset
        #dense and categorical features
        self.num_dense_fea = num_dense_fea
        self.num_sparse_fea = num_sparse_fea
        self.den_fields = list(self._fields.keys())[:num_dense_fea]
        self.spa_fields = list(self._fields.keys())[num_dense_fea:num_dense_fea+num_sparse_fea]

        #files to store dataset
        self.data_file = np.load(data_file, mmap_mode='r', allow_pickle=True)
        self._sessions = np.load(session_file, mmap_mode='r', allow_pickle=True)
        self._history = np.load(history_file, mmap_mode='r', allow_pickle=True)
        
    def __len__(self):
        return self.num_entries

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if isinstance(idx, slice):
            return [self[i] for i in 
                    range(idx.start or 0, idx.stop or len(self), idx.step or 1)]

        #get session and historical actions
        session = self._sessions[self.idx_per_entry_session[idx]:self.idx_per_entry_session[idx+1]]
        history = self._history[self.idx_per_entry_history[idx]:self.idx_per_entry_history[idx+1]]
        
        #get features
        #if dense features present
        dense = self.data_file[idx][self.den_fields] if self.den_fields else np.empty(0, dtype=[('x', 'f8')])
        sparse = self.data_file[idx][self.spa_fields]
        
        #convert to tensor
        session_tensor = torch.from_numpy(session).type(self.item_dtype).view(-1, 1)
        history_tensor = torch.from_numpy(history).type(self.item_dtype).view(-1, 1)
        den_tensor = torch.from_numpy(st_to_unst(dense)).view(1,-1)
        sparse_tensor = torch.from_numpy(st_to_unst(sparse)).view(1,-1).type(self.user_dtype).t()
        batch_size, feature_count = sparse_tensor.size()
        sparse_tensor_offset = torch.zeros(feature_count).reshape(-1, 1).repeat(1, batch_size)
        
        #if dense features present, normalize them and return den_tensor as well
        return sparse_tensor_offset, sparse_tensor, history_tensor, session_tensor

class BatchSampler(torch.utils.data.Sampler):

    '''
    Fetching a list of samples using the indices from batchsampler
    '''
    def __init__(self, length, batch_size, drop_last):
        self.length = length
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        indices = np.arange((self.length//self.batch_size)*self.batch_size).reshape(self.length//self.batch_size, self.batch_size)
        
        #shuffle indices to select instances in random order
        np.random.shuffle(indices)
        
        batch = []
        for _, idx in enumerate(indices):
            batch = idx
            yield batch

        if self.length - indices.size > 0 and not self.drop_last:
            yield np.arange(indices.size, self.length)

    def __len__(self):
        if self.drop_last:
            return self.length // self.batch_size
        else:
            return math.ceil(self.length / self.batch_size)

def make_dataset_and_dataloader(args):

    '''
    generates binary dataset files and returns dataloader
    When training phase AND in distributed mode, it returns the dataset instead of dataloader because
    batchsampler has to initiate after parallel process is initiated in the PARSRec.py file
    '''
    #dataset binary files
    binary_data_file = os.path.join(args.data_directory, '{}_data_{}.npy'.format(args.dataset, args.split))
    fields_file = os.path.join(args.data_directory, '{}_data_fields_{}'.format(args.dataset, args.split))
    session_length_file = os.path.join(args.data_directory, '{}_session_length_{}.npy'.format(args.dataset, args.split))
    session_file = os.path.join(args.data_directory, '{}_session_{}.npy'.format(args.dataset, args.split))
    history_size_file = os.path.join(args.data_directory, '{}_history_size_{}.npy'.format(args.dataset, args.split))
    history_file = os.path.join(args.data_directory, '{}_history_{}.npy'.format(args.dataset, args.split))

    #number of sparse and dense features
    num_dense_fea = 0
    num_sparse_fea = 1 #user ids

    #dataset in binary
    dataset_binary = ParsrecBinDataset(binary_data_file, 
                                      batch_size=1, 
                                      num_dense_fea = num_dense_fea,
                                      num_sparse_fea = num_sparse_fea,
                                      fields_file=fields_file,
                                      session_length_file=session_length_file,
                                      session_file=session_file,
                                      history_size_file=history_size_file,
                                      history_file=history_file,
                                      item_dtype=torch.int16,
                                      user_dtype=torch.int32)

    #dataloader
    batch_sampler = BatchSampler(len(dataset_binary), batch_size=args.train_batch_size, drop_last=False)
    binary_loader = torch.utils.data.DataLoader(
            dataset_binary,
            num_workers=0,
            collate_fn=parsrec_collate_fn,
            pin_memory=False,
            batch_sampler = batch_sampler
        )

    return (dataset_binary.num_dense_fea, dataset_binary) if (args.distributed and args.split == 'train') else (dataset_binary.num_dense_fea, binary_loader)
