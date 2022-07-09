# Copyright (c) Ehsan G., Mohammad M. Ashwin A.
#        _____     _____     _____     _____    _____     _____    _____                                           
#       |  _  |   |  _  |   |  _  |   |  ___   |  _  |   |  ___   |  ___
#       | |_| |   | |_| |   | |_| |   | |___   | |_| |   | |__    | |                                             
#       |   __|   |  _  |   |  _  /   |___  |  |  _  /   | |___   | |___                                                      
#       |  |      | | | |   | | \ \    ___| |  | | \ \   |_____   |_____                                           
# 
# This source code is licensed under the MIT license
#
# Description: an implementation of PARSRec: Explainable Personalized 
# Attention-fused Recurrent Sequential Recommendation Using Session Partial Actions
# 
# References:
# Ehsan Gholami, Mohammad Motamedi, and Ashwin Aravindakshan. 2022.
# PARSRec: Explainable Personalized Attention-fused Recurrent Sequential
# Recommendation Using Session Partial Actions. In Proceedings of the 28th
# ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD
# ’22), August 14–18, 2022, Washington, DC, USA. ACM, New York, NY, USA,
# 11 pages. https://doi.org/10.1145/3534678.3539432


# miscellaneous
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from collections import Counter

# data generation
from utils import *
import data_loader_parsrec
from data_loader_parsrec import parsrec_collate_fn, BatchSampler

# parsrec model
from model import PARSRec

# pytorch
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torchviz import make_dot
from torch.profiler import record_function
from torch.utils.data.sampler import Sampler
from torch.nn.parallel import DistributedDataParallelCPU as DDP

import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
def performance_metrics(target, prediction, prediction_rank):

    '''
    function to calculate performance metrics
    sess_prec (session precision): average percentage of sessions recommended accurately
    HR (hit ratio): average percentage of items recommended accurately
    NDCG (Normalized Discounted Cumulative Gain): average percentage of 
        items recommended accurately penalized by prediction rank
    args:
        target (numpy array): target values for session items in order, shape = (minibatchsize, session_length)
        prediction (numpy array): recommended items for session items in order, shape = (minibatchsize, session_length)
        prediction_rank (numpy array): rand of recommended items for session items in order, shape = (minibatchsize, session_length)
    '''
    global sess_prec, num_instances, HR, HR_instances, NDCG
    batch_size, session_sz = target.shape
    
    #mask EOB items
    mask = target==1
    maskedarray = np.ma.masked_array(prediction, mask)==np.ma.masked_array(target, mask)
    
    #count correct recommendations
    sum_1, count_1 = maskedarray.sum(1), maskedarray.count(1)
    ratio = sum_1/count_1

    if any([i<0 for i in ratio]):
        raise Exception('Negative performance value detected')

    #session precions
    ratio = sum(ratio)
    sess_prec = (ratio * 100 + sess_prec * num_instances) / (num_instances + batch_size)
    num_instances += batch_size
    
    #hit ratio
    HR_count = count_1.sum()
    HR = (sum_1.sum() * 100 + HR * HR_instances) / (HR_instances + HR_count)
    
    #NDCG
    NDCG = (((maskedarray/np.log2(prediction_rank + 2)).sum()) * 100 + NDCG * HR_instances) / (HR_instances + HR_count)
    HR_instances += HR_count

def load_data():
    #convert dataset text file to binary files with train/val/test split
    #for format and details please refer to load_dataset() function in utils.py file
    if args.convert_dataset2binary:
        load_dataset(os.path.join(args.data_directory, args.dataset_filename), 
              items_dtype=np.int16, history_len=args.history_len, 
              dataset=args.dataset, cpu_count=args.cpu_count,
              data_directory=args.data_directory)
    #generate dataloader
    args.split = 'train'
    num_dense_fea, train_ld = data_loader_parsrec.make_dataset_and_dataloader(args)
    train_len = len(train_ld)
    args.split = 'validation'
    num_dense_fea, validation_ld = data_loader_parsrec.make_dataset_and_dataloader(args)
    validation_len = len(validation_ld)
    args.split = 'test'
    num_dense_fea, test_ld = data_loader_parsrec.make_dataset_and_dataloader(args)
    test_len = len(test_ld)
    
    return ((train_len, num_dense_fea, train_ld), (validation_len, num_dense_fea, validation_ld), (test_len, num_dense_fea, test_ld))

def neg_items(T):
        
    '''
    adds pool of negative items for each session-user to minibatch
    this pool of negative items will be used for negative sampling in inference (and train if desired)
    '''
    def torch_delete(tensor, indices):
        #remove indices from a tensor and returns the filtered tensor
        mask = torch.ones(tensor.numel(), dtype=torch.bool)
        mask[indices.type(torch.LongTensor)] = False
        return tensor[mask]

    device = torch.device("cuda", 0) if (args.use_gpu and torch.cuda.is_available()) else torch.device('cpu')

    eob = torch.tensor([0], device=device)
    #total pool of items
    itemset = torch.arange(args.feature_sizes[-1], device=device)

    #set of negative items for session-user
    negs = [torch_delete(itemset, torch.hstack((eob, t))) for t in T]

    return negs
    
def inference(
            data_ld,
            device,
            parsrec,
            split_set='validation',
            epoch=-1,
            skip_upto_batch=0,
            use_gpu=False,
            process_rank=0,
            **kwargs
            ):
    
    global nbatches_inference
    
    #initialize performance metrics
    global sess_prec, num_instances, HR, HR_instances, NDCG, sess_prec_list
    sess_prec, num_instances, HR, HR_instances, NDCG = 0, 0, 0, 0, 0
    sess_prec_list = Counter()
    total_iter, total_time = 0, 0

    parsrec.eval()
    #embedding matrices
    item_emb_ly = parsrec.emb_l[-1]
    household_emb_ly = parsrec.emb_l

    #inference loop over mini batches
    for j, inputBatch in enumerate(data_ld):

        if j < skip_upto_batch:
            continue
        
        #load a batch of data
        lS_o, lS_i, T = inputBatch
        lS_i = [tensor.int().to(device) for tensor in lS_i]
        lS_o = lS_o.to(device)
        T = T.float().to(device)
        #positive + negative sampling
        negs = neg_items(T)
        
        t1 = time_wrap(use_gpu)

        #categorical feature embedding
        h_id = []
        for layer, (cat_index, offset) in enumerate(zip(lS_i, lS_o)):
            h_id.append(household_emb_ly[layer](cat_index, offset))
        
        #sample items for each session for inference pool
        if args.sampling:
            negs_samples = torch.stack([neg[torch.randperm(len(neg))[:args.sample_size_inference]] for neg in negs]).to(device)
        
        ##### initializing tensors and variables #####

        #user embedding vectors and initial hidden state
        user_vector = h_id[0]
        hidden = h_id[1]

        batch_sz, session_sz = T.shape
        #target item
        Target = torch.zeros(batch_sz, dtype=torch.int32, device=device)
        seq_offeset = torch.arange(batch_sz, device=device)
            
        prediction_out = np.zeros((batch_sz, session_sz), dtype=np.int32)
        prediction_rank = np.zeros((batch_sz, session_sz), dtype=np.int16)
        target_out = np.zeros((batch_sz, session_sz), dtype=np.int32)

        ##### end of initializing tensors and variables #####

        #loop over session items
        for step in range(T.shape[1]):
            
            #items from step 0 upto current step (avoids leaks of future items)
            seqs = torch.from_numpy(target_out[:, :max(step, 1)].reshape(-1)).to(device=device)
            seqs_offset = torch.arange(seqs.shape[0], device=device)
            input_item = item_emb_ly(seqs, seqs_offset)
            #reshape back to (batch_size, step, embedding_dim)
            input_item = input_item.reshape(batch_sz, -1, input_item.shape[-1])
            
            #forward step parsrec
            S, hidden, att_weights = parsrec(user_vector, input_item, hidden)

            ##### prediction layer #####
            
            #matrix-multiply by item embedding
            if args.sampling:
                #multiply output of parsrec by only samples and remaining items in session
                E_v = item_emb_ly.emb.weight
                indices = torch.hstack((T, negs_samples))
                indices_emb = E_v[indices.ravel().type(torch.LongTensor)].reshape(indices.shape+(-1,))
                S = S.unsqueeze(1).matmul(torch.transpose(indices_emb,-1,-2)).squeeze()
            else:
                #no sampling, multiply by all items embeddings
                S = S.matmul(item_emb_ly(torch.arange(args.feature_sizes[-1], device=device), torch.arange(args.feature_sizes[-1], device=device)).T)
                    
            #recommend top k products
            SS_idx = torch.topk(S, args.num_recoms, dim=1, sorted=True)[1]
            
            if args.sampling:
                ##if sampling, indices are not sequential anymore. Has to select back right indices from topk argsort indices
                SS_idx = indices[torch.repeat_interleave(torch.arange(SS_idx.shape[0]),SS_idx.shape[1]), SS_idx.ravel()].reshape(SS_idx.shape).float()
            SS_idx = SS_idx.float()
            
            ##### end of prediction layer #####

            ##### teacher enforcing #####

            #check if any of top k recommended items are in the rest of the session
            # and filter out recommended SOB(=0) or EOB(=1) (convert to -1)
            SS_idx[SS_idx<2] = -1
            recoms_in_T = (SS_idx[...,None]==T[:,None,:])
            
            #predicted items not in session
            idx = (~recoms_in_T.any(-1).any(-1))
            #predicted items in session
            recoms_in_T = recoms_in_T.any(-1).int()
            
            #extract correctly recommended top items and their prediction rank
            rank = recoms_in_T.argmax(1)
            Target = SS_idx[np.arange(recoms_in_T.shape[0]), rank].clone()
            
            #save predictions for performance evaluation
            prediction_out[:, step] = Target.detach().cpu().clone()
            prediction_rank[:, step] = rank.detach().cpu().clone()
            
            #randomly select items from rest of the session for next step input 
            #if current prediction is incorrect (teacher enforcing+curriculum learning)
            ind = [torch.ravel(torch.nonzero(row>1)) for row in T[idx]]
            ind = [row[0] if row.numel() else 0 for row in ind]
            Target[idx] = T[idx, ind]
            Target[Target==0] = 1
            Target = Target.long()
                
            #save inferred sessions during training to files
            target_out[:, step] = Target.detach().cpu().clone()

            #remove current target product from rest of the session for next steps
            h, w = T.shape            
            mask = torch.zeros_like(T, dtype=bool).to(device)
            mask[idx, ind] = True
            mask[~idx] = Target[~idx,None]==T[~idx]
            mask[(T==0).all(1),0] = True
            T = T[~mask].reshape(h, w-1)
                
            ##### end of teacher enforcing #####

        performance_metrics(target_out, prediction_out, prediction_rank)
        
        total_iter += 1
        t2 = time_wrap(use_gpu)
        total_time += t2 - t1
    
    # print time, loss and performance
    if not process_rank:
        print(
            "{} it {}/{} of epoch {}, {:.2f} ms/it\n".format(
                split_set, j + 1, nbatches_inference, epoch, 1000.0 * total_time / total_iter)
            + "    sess_prec@{} {:.2f}".format(args.num_recoms, sess_prec)
            + ", HR@{} {:.2f}".format(args.num_recoms, HR)
            + ", NDCG@{} {:.2f}".format(args.num_recoms, NDCG)
            + ", best val_sess_prec {:3.2f} %".format(val_best_perf)
            + " ({})".format(time.strftime("%H:%M")),
            flush=True)

    return

def train(
        data_ld,
        device,
        parsrec,
        optimizer_parsrec,
        optimizer_parsrec_sparse,
        split_set='train',
        epoch=-1,
        skip_upto_batch=0,
        use_gpu=False,
        process_rank=0,
        **kwargs
        ):
    
    global args
    global nbatches
    
    #initialize performance metrics
    total_loss, total_samp, total_iter, total_time = 0, 0, 0, 0

    parsrec.train()
    #embedding matrices
    if args.distributed:
        item_emb_ly = parsrec.module.emb_l[-1]
        household_emb_ly = parsrec.module.emb_l
    else:
        item_emb_ly = parsrec.emb_l[-1]
        household_emb_ly = parsrec.emb_l

    #training loop over mini batches
    for j, inputBatch in enumerate(data_ld):

        if j < skip_upto_batch:
            continue

        #load a batch of data
        lS_o, lS_i, T = inputBatch
        lS_i = [tensor.int().to(device) for tensor in lS_i]
        lS_o = lS_o.to(device)
        T = T.float().to(device)

        t1 = time_wrap(use_gpu)
    
        parsrec.zero_grad()
        
        #loss  
        minibatch_loss_ = 0

        #loop over session items
        with torch.autograd.set_detect_anomaly(True):
          
            # forward pass
            h_id = []
            for layer, (cat_index, offset) in enumerate(zip(lS_i, lS_o)):
                h_id.append(household_emb_ly[layer](cat_index, offset))

            ##### initializing tensors and variables #####
            
            #user embedding vectors and initial hidden state
            user_vector = h_id[0]
            hidden = h_id[1]

            batch_sz, session_sz = T.shape
            #target item
            Target = torch.zeros(batch_sz, dtype=torch.int32, device=device)
            seq_offeset = torch.arange(batch_sz, device=device)                        
            
            #per minibatch loss
            minibatch_loss = 0

            #array to keep track of the teacher enforcing targets
            target_out = np.zeros((batch_sz, session_sz), dtype=np.int32)

            ##### end of initializing tensors and variables #####

            #loop over session items
            for step in range(T.shape[1]):
                
                #items from step 0 upto current step (avoids leaks of future items)
                seqs = torch.from_numpy(target_out[:, :max(step, 1)].reshape(-1)).to(device=device)
                seqs_offset = torch.arange(seqs.shape[0], device=device)
                input_item = item_emb_ly(seqs, seqs_offset)
                #reshape back to (batch_size, step, embedding_dim)
                input_item = input_item.reshape(batch_sz, -1, input_item.shape[-1])
                
                #forward step parsrec
                S, hidden, att_weights = parsrec(user_vector, input_item, hidden)

                ##### prediction layer #####

                #no sampling during training phase, multiply by all items embeddings
                S = S.matmul(item_emb_ly(torch.arange(args.feature_sizes[-1], device=device), torch.arange(args.feature_sizes[-1], device=device)).T)
                
                #recommend top product
                SS_idx = torch.topk(S, 1, dim=1, sorted=True)[1].float()
                
                ##### end of prediction layer #####

                ##### teacher enforcing #####
                
                #check if any of top k recommended items are in the rest of the session
                # and filter out recommended SOB(=0) or EOB(=1) (convert to -1)
                SS_idx[SS_idx<2] = -1
                recoms_in_T = (SS_idx[...,None]==T[:,None,:])

                #predicted items not in session
                idx = (~recoms_in_T.any(-1).any(-1))
                #predicted items in session
                recoms_in_T = recoms_in_T.any(-1).int()
                
                #extract correctly recommended top items and their prediction rank
                rank = recoms_in_T.argmax(1)
                Target = SS_idx[np.arange(recoms_in_T.shape[0]), rank].clone()
                
                #randomly select items from rest of the session for next step input 
                #if current prediction is incorrect (teacher enforcing+curriculum learning)
                ind = [torch.ravel(torch.nonzero(row>1)) for row in T[idx]]
                ind = [row[0] if row.numel() else 0 for row in ind]
                Target[idx] = T[idx, ind]
                Target[Target==0] = 1
                Target = Target.long()
                
                #save inferred sessions during training to files
                target_out[:, step] = Target.detach().cpu().clone()

                #remove current target products from session for next steps
                h, w = T.shape
                mask = torch.zeros_like(T, dtype=bool).to(device)
                mask[idx, ind] = True 
                mask[~idx] = Target[~idx,None]==T[~idx]
                mask[(T==0).all(1),0] = True
                T = T[~mask].reshape(h, w-1)
                
                ##### end of teacher enforcing #####
                
                #loss
                if args.distributed:
                    step_loss = parsrec.module.loss_fn(S, Target)
                else:
                    step_loss = parsrec.loss_fn(S, Target)
                minibatch_loss += step_loss

                if torch.isnan(S).any():
                    print('step {}\n, hidden\n {}\nparsrec output\n{}\nError\n{}\n'.format(step, hidden, S, minibatch_loss))
                    exit('NaN detected during training phase. Requires further debugging')
                
            minibatch_loss_ += minibatch_loss.detach().cpu().numpy()

            #update weights, backward step parsrec
            with record_function("parsrec backward"):
                optimizer_parsrec.zero_grad()
                optimizer_parsrec_sparse.zero_grad()                
                minibatch_loss.backward()
                
                #gradient clipping
                torch.nn.utils.clip_grad_norm_(parsrec.parameters(), args.clip_grad_threshold)

                optimizer_parsrec.step()
                optimizer_parsrec_sparse.step()

            # plot compute graph
            if args.plot_compute_graph:
                assert not args.distributed, 'to debug the model please use non-distributed mode'
                dot = make_dot(minibatch_loss, params=dict(parsrec.named_parameters()))
                dot.render(os.path.join(args.output_directory, 'PARSRec_graph')) # write pdf file
                args.plot_compute_graph = False

        total_loss += minibatch_loss_ * T.shape[0]
        total_samp += T.shape[0] * session_sz
        total_iter += 1

        t2 = time_wrap(use_gpu)
        total_time += t2 - t1

    # print time and loss
    loss = total_loss / total_samp
    if not args.distributed:
        print('.' * 100 + '\n' + \
            "{} it {}/{} of epoch {}, {:.2f} ms/it\n".format(
                split_set, j + 1, nbatches, epoch, 1000.0 * total_time / total_iter
            )
            + "    loss {:.6f}".format(loss)
            + " ({})".format(time.strftime("%H:%M")),
            flush=True)

    return loss

def validate_test(args,
                validation_ld,
                test_ld,
                device,
                model_module,
                parsrec,
                optimizer_parsrec,
                optimizer_parsrec_sparse,
                epoch,
                skip_upto_batch,
                train_loss_epochs,
                sess_prec_validation_epoch,
                sess_prec_test_epoch,
                HR_validation_epoch,
                HR_test_epoch,
                NDCG_validation_epoch,
                NDCG_test_epoch,
                use_gpu,
                val_best_perf,
                perf_loss_file,
                rank=0):

    '''
    run validation and test inference and evaluate performance and log results
    return best validation session_precision
    '''
    global sess_prec, num_instances, HR, HR_instances, NDCG

    ##### validation #####
    inference(validation_ld,
            device,
            model_module,
            epoch=epoch,
            split_set='validation',
            skip_upto_batch=skip_upto_batch,
            use_gpu=use_gpu,
            process_rank=rank)

    #save best model
    isbest = sess_prec - val_best_perf >= -1e-4
    val_best_perf = max(sess_prec, val_best_perf)
    if isbest and args.save_best_model:
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': parsrec.state_dict(),
                    'optimizer_state_dict': optimizer_parsrec.state_dict(),
                    'optimizer_sparse_state_dict': optimizer_parsrec_sparse.state_dict(),
                    'best_perf': val_best_perf,
                    'train_loss': train_loss_epochs[epoch]},
                    os.path.join(args.output_directory, args.save_model_path))
    #loss and performance
    sess_prec_validation_epoch.append(sess_prec)
    HR_validation_epoch.append(HR)
    NDCG_validation_epoch.append(NDCG)
    perf_loss_file.write('valid sess_prec: {}, HR: {}, NDCG: {}, best val_sess_prec: {} \n'.format(sess_prec, HR, NDCG, val_best_perf))

    ##### test #####
    nbatches_inference = len(test_ld)
    inference(test_ld,
            device,
            model_module,
            epoch=epoch,
            split_set='test',
            skip_upto_batch=skip_upto_batch,
            use_gpu=use_gpu,
            process_rank=rank
            )
    #performance
    sess_prec_test_epoch.append(sess_prec)
    HR_test_epoch.append(HR)
    NDCG_test_epoch.append(NDCG)
    perf_loss_file.write('test sess_prec: {}, HR: {}, NDCG: {} \n'.format(sess_prec, HR, NDCG))

    return val_best_perf

def distributed_worker(rank,
                        train_ld,
                        validation_ld,
                        test_ld,
                        device,
                        parsrec,
                        optimizer_parsrec,
                        optimizer_parsrec_sparse,
                        skip_upto_epoch,
                        skip_upto_batch,
                        use_gpu,
                        args_local,
                        nbatches_local,
                        nbatches_inference_local,
                        train_loss_epochs):

    '''
    train distributed data parallel on multi-device (CPUs or GPUs)
    '''
    print('Rank ', rank)

    ##### initializing variables #####
    
    global args, nbatches_inference, nbatches
    args, nbatches_inference, nbatches = args_local, nbatches_inference_local, nbatches_local
    
    global val_best_perf
    val_best_perf = 0
    
    global sess_prec, num_instances, sess_prec_list, HR, HR_instances, NDCG
    sess_prec_validation_epoch, sess_prec_test_epoch = [0] * skip_upto_epoch, [0] * skip_upto_epoch
    HR_validation_epoch, HR_test_epoch = [0] * skip_upto_epoch, [0] * skip_upto_epoch
    NDCG_validation_epoch, NDCG_test_epoch = [0] * skip_upto_epoch, [0] * skip_upto_epoch
    #file to log performance and loss
    perf_loss_file = resetfile(args.output_directory, ['perf_loss.txt'], mode='a')[0]
    
    ##### initializing variables #####

    #start the parallel process
    dist.init_process_group(                                   
        backend='gloo',                                         
        init_method='env://',                                   
        world_size=args.cpu_count,                              
        rank=rank                                               
    )

    #distributed data parallel the model
    model = DDP(parsrec.to(device), find_unused_parameters=False)

    #distribute dataloader over nodes for training
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ld)
    train_loader = torch.utils.data.DataLoader(dataset=train_ld, batch_size=args.train_batch_size, shuffle=False, 
                                                num_workers=0, sampler=train_sampler, collate_fn=parsrec_collate_fn) 
    
    for k in range(skip_upto_epoch, args.nepochs):
        
        #file to log performance and loss
        perf_loss_file = open(os.path.join(args.output_directory, 'perf_loss.txt'),'a')

        t0 = time_wrap(use_gpu)

        train_sampler.set_epoch(k)
        
        # train for one epoch
        loss_ = train(train_loader, 
                    device,
                    model,
                    optimizer_parsrec,
                    optimizer_parsrec_sparse,
                    split_set='train',
                    epoch=k,
                    skip_upto_batch=skip_upto_batch,
                    use_gpu=use_gpu)

        #loss and performance
        train_loss_epochs[k] += loss_/args.cpu_count
        
        #print and log loss and performance
        if not rank:
            train_loss = train_loss_epochs[k]
            print_str = '.' * 100 + '\n' + \
                        'training of epoch {} in {:.2f} s.\n    train loss: {:.6f}\n'.format(
                                k, time_wrap(use_gpu)-t0, train_loss_epochs[k])
            print(print_str)
            perf_loss_file.write(print_str)

        #validation and test
        if not rank:
            val_best_perf = validate_test(args,
                                        validation_ld,
                                        test_ld,
                                        device,
                                        model.module,
                                        parsrec,
                                        optimizer_parsrec,
                                        optimizer_parsrec_sparse,
                                        k,
                                        skip_upto_batch,
                                        train_loss_epochs,
                                        sess_prec_validation_epoch,
                                        sess_prec_test_epoch,
                                        HR_validation_epoch,
                                        HR_test_epoch,
                                        NDCG_validation_epoch,
                                        NDCG_test_epoch,
                                        use_gpu,
                                        val_best_perf,
                                        perf_loss_file,
                                        rank=rank)
            
            #plot loss and performance
            if k % args.plot_freq == 0:
                plot_loss_perf(args,
                    train_loss_epochs[:k+1],
                    sess_prec_validation_epoch,
                    sess_prec_test_epoch,
                    HR_validation_epoch,
                    HR_test_epoch,
                    NDCG_validation_epoch,
                    NDCG_test_epoch)
                
    dist.destroy_process_group()
    perf_loss_file.close()
    
def parse_args():
    ### parse arguments ###
    parser = argparse.ArgumentParser(description="Train PARSRec")

    # model related parameters
    parser.add_argument("--num-recoms", type=int, default=1)
    parser.add_argument("--dropout-rate", type=float, default=0.0)
    parser.add_argument("--feature-sizes", type=dash_separated_ints, default="128")
    parser.add_argument("--num-attn-blocks", type=int, default=1)
    parser.add_argument("--num-att-heads", type=int, default=1)
    parser.add_argument("--emb-dims", type=dash_separated_ints, default="0")
    # activations and loss
    parser.add_argument("--loss-function", type=str, choices=["mse", "bce", "bcewl", "wbce", "cce", "wcce"], default="cce")
    #distributed data parallel
    parser.add_argument("--cpu-count", type=int, default=1)
    parser.add_argument("--distributed", action="store_true", default=False)
    parser.add_argument("--masterport", type=str, default="1235")
    # data
    parser.add_argument("--history-len", type=int, default=sys.maxsize)
    parser.add_argument("--num-batches", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="synthetic")    
    parser.add_argument("--dataset-filename", type=str, default="synthetic.txt")    
    parser.add_argument("--convert-dataset2binary", action="store_true", default=False)
    # training
    parser.add_argument("--numpy-rand-seed", type=int, default=123)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--nepochs", type=int, default=100)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--optimizer-sparse", type=str, default="sparseadam")
    parser.add_argument("--clip-grad-threshold", type=float, default=20.0)
    parser.add_argument("--need-weights", action="store_true", default=False)
    parser.add_argument('--split', choices=['train', 'test', 'val'], default='train')
    # inference
    parser.add_argument("--sampling", action="store_true", default=False)
    parser.add_argument("--inference-only", action="store_true", default=False)
    parser.add_argument("--sample-size-inference", type=int, default=100)
    # gpu
    parser.add_argument("--use-gpu", action="store_true", default=False)
    #print options
    parser.add_argument("--print-precision", type=int, default=2)
    # debugging and profiling
    parser.add_argument("--save-freq", type=int, default=1)
    parser.add_argument("--save-best-model", action="store_true", default=False)
    parser.add_argument("--plot-freq", type=int, default=1)
    parser.add_argument("--debug-mode", action="store_true", default=False)
    parser.add_argument("--plot-compute-graph", action="store_true", default=False)
    # store/load model
    parser.add_argument("--load-model", action="store_true", default=False)
    parser.add_argument("--save-model-path", type=str, default="best_model.tar")
    parser.add_argument("--load-model-path", type=str, default="best_model.tar")
    #directories
    parser.add_argument('--data-directory', default='data')
    parser.add_argument('--output-directory', default='out/nielsen_out')
    
    return parser.parse_args()

def run():

    #torch.backends.cudnn.benchmark = True
    global args
    global nbatches, nbatches_inference
    args = parse_args()

    #create directory for output files
    if not os.path.isdir(args.output_directory):
        os.makedirs(args.output_directory)
    
    #save a log of variable arguments
    with open(os.path.join(args.output_directory, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ' , ' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))    
    
    #log performance and loss
    perf_loss_file = resetfile(args.output_directory, ['perf_loss.txt'], mode='a')[0]

    ##### some basic setup #####
    np.random.seed(args.numpy_rand_seed)
    torch.manual_seed(args.numpy_rand_seed)
    np.set_printoptions(precision=args.print_precision)
    torch.set_printoptions(precision=args.print_precision)

    use_gpu = args.use_gpu and torch.cuda.is_available()

    if use_gpu:
        ngpus = torch.cuda.device_count()
        device = torch.device("cuda", 0)
        print("Using {} GPU(s)...".format(ngpus))
    else:
        device = torch.device("cpu")
        print("Using CPU...")

    #list of vocabulary size for all categorical features
    args.feature_sizes = np.fromstring(args.feature_sizes, dtype=int, sep="-")
    feature_sizes = np.asarray(args.feature_sizes)
    #embedding dimensions of categorical features
    emb_dims = np.fromstring(args.emb_dims, dtype=int, sep="-")
    
    ##### prepare data #####
    print('Loading data'+'.'*50)
    ((train_len, num_dense_fea, train_ld), 
     (validation_len, num_dense_fea, validation_ld), 
     (test_len, num_dense_fea, test_ld)) = load_data()
    
    nbatches = args.num_batches if args.num_batches > 0 else (train_len//args.train_batch_size//args.cpu_count if args.distributed else train_len)
    nbatches_inference = validation_len

    ##### main loop #####

    # initiating variables
    skip_upto_epoch = 0
    skip_upto_batch = 0
    global val_best_perf
    val_best_perf = 0
                    
    print('number of train      batches ',train_len//args.train_batch_size if args.distributed else train_len)
    print('number of validation batches ',validation_len)
    print('number of test       batches ',test_len)
    
    ###initiating model

    sanity_check(feature_sizes, emb_dims, args.debug_mode)

    global parsrec
    parsrec = PARSRec(
            feature_sizes=feature_sizes,
            emb_dims=emb_dims,
            loss_function=args.loss_function,
            dropout_rate=args.dropout_rate,
            num_attn_blocks=args.num_attn_blocks,
            num_att_heads=args.num_att_heads,
            need_weights=args.need_weights
            )
    parsrec = parsrec.to(device)

    print('model capacity:', sum(p.numel() for p in parsrec.parameters() if p.requires_grad))
    print('.' * 50 + '\n')

    # test prints
    if args.debug_mode:
        print("initial parameters (weights and bias):")
        for param in parsrec.parameters():
            print(param.detach().cpu().numpy())
        print(parsrec)

        
    # specify the optimizer algorithm
    if not args.inference_only:
        if use_gpu and args.optimizer in ["rwsadagrad", "adagrad"]:
            sys.exit("GPU version of Adagrad is not supported by PyTorch.")
        opts = {"sgd": torch.optim.SGD,
                "adagrad": torch.optim.Adagrad,
                "sparseadam": torch.optim.SparseAdam,
                "adam": torch.optim.Adam}
        
        #use sparseadam optimizer for embeddings and Adam for other layers
        #if model structure is modified, opt assignment needs to be updated too
        parameters = list(parsrec.parameters())
        optimizer_parsrec = opts[args.optimizer](parameters[:-len(feature_sizes)])
        optimizer_parsrec_sparse = opts[args.optimizer_sparse](parameters[-len(feature_sizes):])

    #load previously trained model
    if args.load_model:
        checkpoint = torch.load(os.path.join(args.output_directory, args.load_model_path))
        skip_upto_epoch = checkpoint['epoch'] + 1
        parsrec.load_state_dict(checkpoint['model_state_dict'])
        val_best_perf = checkpoint['best_perf']
        if not args.inference_only:
            optimizer_parsrec.load_state_dict(checkpoint['optimizer_state_dict'])
            optimizer_parsrec_sparse.load_state_dict(checkpoint['optimizer_sparse_state_dict'])

    ###end of model initiation

    #initialize performance metrics 
    global sess_prec, num_instances, sess_prec_list, HR, HR_instances, NDCG
    train_loss_epochs = [0] * skip_upto_epoch
    sess_prec_validation_epoch, sess_prec_test_epoch = [0] * skip_upto_epoch, [0] * skip_upto_epoch 
    HR_validation_epoch, HR_test_epoch = [0] * skip_upto_epoch, [0] * skip_upto_epoch
    NDCG_validation_epoch, NDCG_test_epoch = [0] * skip_upto_epoch, [0] * skip_upto_epoch

    if not args.inference_only:   #train mode
        
        if not args.distributed:      #non-parallel mode
            
            for k in range(skip_upto_epoch, args.nepochs):

                perf_loss_file.write('epoch {}\n'.format(k))
                perf_loss_file.close()
                perf_loss_file = open(os.path.join(args.output_directory, 'perf_loss.txt'),'a')

                t0 = time_wrap(use_gpu)

                train_loss = train(train_ld,
                                    device,
                                    parsrec,
                                    optimizer_parsrec,
                                    optimizer_parsrec_sparse,
                                    split_set='train',
                                    epoch=k,
                                    skip_upto_batch=skip_upto_batch,
                                    use_gpu=use_gpu)

                print('training epoch {} in {:.2f} s\n'.format(k, time_wrap(use_gpu)-t0))
                train_loss_epochs.append(train_loss)
                perf_loss_file.write('    train loss: {:.6f} \n'.format(train_loss))

                #run validation every epoch and save best model on validation
                val_best_perf = validate_test(args,
                                            validation_ld,
                                            test_ld,
                                            device,
                                            parsrec,
                                            parsrec,
                                            optimizer_parsrec,
                                            optimizer_parsrec_sparse,
                                            k,
                                            skip_upto_batch,
                                            train_loss_epochs,
                                            sess_prec_validation_epoch,
                                            sess_prec_test_epoch,
                                            HR_validation_epoch,
                                            HR_test_epoch,
                                            NDCG_validation_epoch,
                                            NDCG_test_epoch,
                                            use_gpu,
                                            val_best_perf,
                                            perf_loss_file)
                
                #plot loss and performance
                if k % args.plot_freq == 0:
                    plot_loss_perf(args,
                                    train_loss_epochs,
                                    sess_prec_validation_epoch,
                                    sess_prec_test_epoch,
                                    HR_validation_epoch,
                                    HR_test_epoch,
                                    NDCG_validation_epoch,
                                    NDCG_test_epoch)
            perf_loss_file.close()

        else:   #parallel mode
            mp.set_sharing_strategy("file_system")
            os.environ['MASTER_ADDR'] = 'localhost'              #
            os.environ['MASTER_PORT'] = args.masterport
            os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'  

            train_loss_epochs = [torch.tensor(0.0) for _ in range(args.nepochs)]
            for tens in train_loss_epochs: tens.share_memory_()
            
            mp.spawn(distributed_worker, 
                    nprocs=args.cpu_count, 
                    args=(train_ld,
                            validation_ld,
                            test_ld,
                            device,
                            parsrec,
                            optimizer_parsrec,
                            optimizer_parsrec_sparse,
                            skip_upto_epoch,
                            skip_upto_batch,
                            use_gpu,
                            args,
                            nbatches,
                            nbatches_inference,
                            train_loss_epochs,
                            ),)

    else:   #inference only
        assert args.load_model, 'no model is loaded. please turn on load_model flag'
        assert args.distributed != args.inference_only, 'when inference_only do not use distributed' 
        print("Testing for inference only")
        nbatches_inference = test_len
        inference(test_ld,
                device,
                parsrec,
                epoch=skip_upto_epoch,
                split_set='test',
                skip_upto_batch=skip_upto_batch,
                use_gpu=use_gpu)

        sess_prec_test_epoch.append(sess_prec)
        HR_test_epoch.append(HR)
        NDCG_test_epoch.append(NDCG)
        perf_loss_file.write('test sess_prec: {}, HR: {}, NDCG: {} \n'.format(sess_prec, HR, NDCG))
        perf_loss_file.close()

if __name__ == "__main__":
    run()
