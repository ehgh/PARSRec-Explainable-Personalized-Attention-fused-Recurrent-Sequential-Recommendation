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
# Description: an implementation of miscellaneous methods
# 
# References:
# Ehsan Gholami, Mohammad Motamedi, and Ashwin Aravindakshan. 2022.
# PARSRec: Explainable Personalized Attention-fused Recurrent Sequential
# Recommendation Using Session Partial Actions. In Proceedings of the 28th
# ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD
# ’22), August 14–18, 2022, Washington, DC, USA. ACM, New York, NY, USA,
# 11 pages. https://doi.org/10.1145/3534678.3539432

import os
import gc
import time
import json
import torch
import errno
import pickle
import swifter
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from itertools import chain
from matplotlib import pyplot as plt
from multiprocessing import Pool as mpPool


tqdm.pandas()

def time_wrap(use_gpu=False):
    if use_gpu:
        torch.cuda.synchronize()
    return time.time()

def dash_separated_ints(value):
    vals = value.split("-")
    for val in vals:
        try:
            int(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of ints" % value)
    return value

def resetfile(directory, filenames, mode='w'):
    '''
    reset a list of filenames in directory if exist (remove contents)
    '''
    files = []
    for filename in filenames:
        path = os.path.join(directory, filename)
        try:
            os.remove(path)
        except OSError as e:
            if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
                raise Exception("Cannot remove file while it may exist!")# re-raise exception if a different error occurred
        files.append(open(path, mode))
    return files

def sanity_check(feature_sizes, emb_dims, debug_mode=False):

    # sanity check: number of features and embedding dimensions must match
    if len(emb_dims)!=len(feature_sizes):
        sys.exit(
            "ERROR: # of categorical features "
            + str(len(feature_sizes))
            + " does not match # of embedding dimensions "
            + str(len(emb_dims)))

    if sum(emb_dims)==0:
        print('No embedding dimensions provided for categorical features, all features emb dim will set to {}'.format(128))
        emb_dims = [128] * len(feature_sizes)    

    # test prints (model arch)
    if debug_mode:
        print("model arch:")
        print(
            "features and their categories are:\n"
            + "# of users:user embedding dimension:\n"
            + "{}:{}".format(feature_sizes[0], emb_dims[0])
            + "# of items:item embedding dimension:\n"
            + "{}:{}".format(feature_sizes[1], emb_dims[1])
        )

def plot_helper(subplot=(1,1,1),
                    yscale='linear',
                    ylabel='',
                    ylim=[],
                    plot={},
                    loc="upper left"
                    ):

    plt.subplot(*subplot)
    plt.xlabel('Epochs', fontsize=15)
    plt.yscale(yscale)
    plt.ylabel(ylabel, fontsize=15)
    plt.ylim(ylim)
    for lbl, data in plot.items():
        plt.plot(data, label=lbl)
    plt.legend(loc=loc)

def plot_loss_perf(args,
                    train_loss_epochs,
                    sess_prec_validation_epoch,
                    sess_prec_test_epoch,
                    HR_validation_epoch,
                    HR_test_epoch,
                    NDCG_validation_epoch,
                    NDCG_test_epoch,
                    ):
    '''
    plot train loss and validation/test performance metrics
    '''
    plt.rcParams["figure.figsize"] = [12,12]
    sns.set_theme(font_scale=1.2)
    plt.clf()
    plot_helper(subplot=(4,1,1),
                yscale='log',
                ylabel='Loss',
                ylim=[min(train_loss_epochs)*0.9, min(train_loss_epochs)*2],
                plot={'train': train_loss_epochs},
                loc="upper right")
    plot_helper(subplot=(4,1,2),
                ylabel='Average\n sess_prec@{}'.format(args.num_recoms),
                ylim=[-10, 80],
                plot={'validation': sess_prec_validation_epoch,
                    'test': sess_prec_test_epoch})
    plot_helper(subplot=(4,1,3),
                ylabel='Average\n HR@{}'.format(args.num_recoms),
                ylim=[-10, 80],
                plot={'validation': HR_validation_epoch,
                    'test': HR_test_epoch})
    plot_helper(subplot=(4,1,4),
                ylabel='Average\n NDCG@{}'.format(args.num_recoms),
                ylim=[-10, 80],
                plot={'validation': NDCG_validation_epoch,
                    'test': NDCG_test_epoch})
    plt.gcf().set_size_inches(12, 12)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_directory, 'perf_loss.png'), bbox_inches='tight')
              
def load_pickle(filename):
    with open(filename + '.pkl', 'rb') as f:
        return pickle.load(f)

def save_pickle(obj, filename):
    with open(filename + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_json(filename):
    with open(filename, 'r') as fp:
        return json.load(fp)

def df2binary(df, 
              data_directory, 
              feature_filename, 
              bytes_per_field_filename, 
              session_length_filename, 
              session_filename, 
              session_dtype, 
              history_size_filename, 
              history_filename, 
              flag_debug=False):
    '''
    Save pandas dataframe to binary files
    '''

    #save sessions and session_lengths
    np.save(os.path.join(data_directory, session_length_filename), df['session_length'].to_numpy())
    np.save(os.path.join(data_directory, session_filename), np.hstack(df['session_items'])) 
    
    #save history and history_lengths
    np.save(os.path.join(data_directory, history_size_filename), df['history'].map(len).to_numpy())
    np.save(os.path.join(data_directory, history_filename), np.hstack(df['history'])) 

    df.drop(['session_length', 'session_items', 'history'], axis=1, inplace=True)
    # convert dataframe to record array, then cast to structured array (user ids)
    data = df.to_records(index=False).view(type=np.ndarray, dtype=list(df.dtypes.items()))
    np.save(os.path.join(data_directory, feature_filename), data)
    
    #store each fieldname:(dtype, num_of_bytes_from_head)
    fields = dict(data.dtype.fields)
    fields['num_entries'] = data.shape[0]
    fields['session_items'] = np.dtype(session_dtype)
    save_pickle(fields, os.path.join(data_directory, bytes_per_field_filename)) 

def history_helper(gr, history_col, item_col_dtype):
        '''
        helper to process build_history in parallel
        '''
        gr[history_col] = gr.apply(lambda x:np.array(list(chain.from_iterable(
                                gr.loc[x.first_idx:x.last_idx-1, 'session_items'])), 
                                dtype=item_col_dtype), axis=1)
        gc.collect()
        return gr

def build_history(df, 
                user_col='user_id', 
                item_col='item_id', 
                item_col_dtype=np.int16, 
                history_col='history', 
                history_len=1, 
                cpu_count=1):
    '''
    extract user historical actions for each session
    '''
    print('extracting user historical interactions' + '.' * 50)
    df = df.sort_values([user_col, 'time']).reset_index(drop=True)
    df['prev_time'] = df['time'] - history_len
    
    gc.collect()

    df_list = []

    #find the time window of last 'history_len' sessions for every session
    for gr_name, gr in tqdm(df.groupby(user_col)):
      last_idx = (gr.reset_index().groupby('time')['index'].first()).rename('last_idx')
      first_idx = gr.time.searchsorted(gr.prev_time)
      dfi = pd.DataFrame({'first_idx':gr.index[first_idx], 'last_idx':last_idx}, index=gr.time)
      df_list.append(dfi)
    
    idx_df = pd.concat(df_list).reset_index(drop=True)
    gc.collect()
    df = pd.merge(df, idx_df, left_index=True, right_index=True)
    gc.collect()

    ##breakdown df to process faster  
    if len(df) < 1000:
      #small dataset
      df[history_col] = df.progress_apply(lambda x:np.array(list(chain.from_iterable(
                                  df.loc[x.first_idx:x.last_idx-1, 'session_items'])), dtype=item_col_dtype), axis=1)
    else:
      ##big datasets, break down by each group and process and concat
      pool = mpPool(cpu_count)
      results = [pool.apply_async(history_helper, 
                                  args=(gr, history_col, item_col_dtype),
                                  ) for (gr_name, gr) in df.groupby(user_col)] # maps function to iterator
      df_list = [p.get() for p in tqdm(results)]   # collects and returns the results
      pool.close()
      pool.join()
      df = pd.concat(df_list)
      gc.collect()
    
    df.drop(['prev_time', 'first_idx', 'last_idx'], axis=1, inplace=True)

    return df

def load_dataset(filename, 
                items_dtype=np.int16, 
                history_len=1, 
                dataset=None, 
                cpu_count=1,
                data_directory=''):
    '''
    load dataset file and convert it into dataframe, then extract user historical actions for 
    each session and split into train-validation-test datasets and finally save to binary files
    These binary files later will be used by parsrec_data_loader.py to generate a fast dataloader
    
    dataset file format:

    Header line:
    user_id,time,session_length,session_items
    
    Data lines:
    #e.g. case (user_id:1 ,time:5 ,session_length:6,session_items:3,7,4,8,12,34) is:
    1,5,6,3,7,4,8,12,34
    
    NOTE: item_ids must start from 2, (0 and 1 are reserved for SOB and EOB, respectively)

    args:
        filename (str): dataset file address
        items_dtype (dtype): dtype to store item ids (max(item_id) must fit into dtype)
        history_len (int): number of previous sessions to include in the user history
        dataset (str): name of dataset
        cpu_count (int): number of cpu cores to use in parallel process the build_history function
        data_directory (str): directory to save binary output files
    '''
    data, instance_counter = {}, 0
    with open(filename, 'r') as f:
        #skip first line (headers)
        next(f)
        for line in f:
            line = list(map(int, line.strip().split(',')))
            assert len(line) == line[2] + 3, 'number of session items is not equal to session length'
            #add instance to the data dictionary
            data[instance_counter] = [line[0], line[1], line[2], np.array(line[3:], dtype=items_dtype)]
            instance_counter += 1
    
    #convert data dictionary to dataframe
    df = pd.DataFrame.from_dict(data, orient='index',
                       columns=['user_id', 'time', 'session_length', 'session_items'])
    #extract user interaction history for each user-session
    df = build_history(df, 
                    user_col='user_id', 
                    item_col='session', 
                    item_col_dtype=np.int16, 
                    history_col='history', 
                    history_len=history_len,
                    cpu_count=cpu_count)

    ##reorder columns and change column dtype
    ##session_length and session always have to be first and second columns respectively
    ##continuous vars come before categorical vars
    print('Reorder columns'+'.'*50)
    columns_dtype = load_json(os.path.join(data_directory, dataset + '_columns_dtype.json'))
    df = df[columns_dtype.keys()].astype(columns_dtype)
    gc.collect()

    #split into train-validation-test sets
    print('split into train/test and save'+'.'*50)
    df = df.set_index('time')
    df.sort_values(['time','user_id'], inplace=True)
    grps = df.groupby('user_id', as_index=False)
    #last session in test, second to last in validation and rest in train
    train = grps.apply(lambda x: x.iloc[:-2])
    validation = grps.nth(-2)
    test = grps.nth(-1)
  
    #print stats
    stat = df.groupby('user_id').agg({'session_length':'sum'})['session_length']
    print('avg num of actions per household', stat.mean())
    print('median num of actions per household', stat.median())
    print('max num of actions per household', stat.max())
    print('min num of actions per household', stat.min())

    #drop interaction date (not needed anymore)
    train = train.reset_index(drop=True).drop('time', axis=1, errors='ignore')
    validation = validation.reset_index(drop=True).drop('time', axis=1, errors='ignore')
    test = test.reset_index(drop=True).drop('time', axis=1, errors='ignore')
      
    ##sort by session size for minimal zero padding of sessions within a minibatch in dataloader
    train = train.sort_values('session_length')
    validation = validation.sort_values('session_length')
    test = test.sort_values('session_length')

    #save binary files
    df2binary(train, data_directory, 
            dataset + '_data_train.npy', 
            dataset + '_data_fields_train', 
            dataset + '_session_length_train.npy', 
            dataset + '_session_train.npy', 
            'int16', 
            dataset + '_history_size_train.npy', 
            dataset + '_history_train.npy', 
            flag_debug=False)
    df2binary(validation, data_directory, 
            dataset + '_data_validation.npy', 
            dataset + '_data_fields_validation', 
            dataset + '_session_length_validation.npy', 
            dataset + '_session_validation.npy', 
            'int16', 
            dataset + '_history_size_validation.npy', 
            dataset + '_history_validation.npy', 
            flag_debug=False)
    df2binary(test, data_directory, 
            dataset + '_data_test.npy', 
            dataset + '_data_fields_test', 
            dataset + '_session_length_test.npy', 
            dataset + '_session_test.npy', 
            'int16', 
            dataset + '_history_size_test.npy', 
            dataset + '_history_test.npy', 
            flag_debug=False)
    print('train/validation/test sets shape: {}/{}/{}'.format(train.shape, validation.shape, test.shape))
