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
# Description: an implementation of PARSRec model
# 
# References:
# Ehsan Gholami, Mohammad Motamedi, and Ashwin Aravindakshan. 2022.
# PARSRec: Explainable Personalized Attention-fused Recurrent Sequential
# Recommendation Using Session Partial Actions. In Proceedings of the 28th
# ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD
# ’22), August 14–18, 2022, Washington, DC, USA. ACM, New York, NY, USA,
# 11 pages. https://doi.org/10.1145/3534678.3539432

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as Functional

class Custom_Attention(torch.nn.Module):
    '''
    Attention block
    '''
    def __init__(self, 
                num_attn_blocks, 
                num_heads, 
                dropout_rate, 
                embedding_dim,
                key_dim,
                last_layer_output_size,
                need_weights):

        super().__init__()

        self.need_weights = need_weights
        
        #modules
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layers = nn.ModuleList()  #to match attention output to RNN input

        for i in range(num_attn_blocks):

            #layer norm
            self.attention_layernorms.append(nn.LayerNorm(embedding_dim, eps=1e-8))

            #attention layer
            new_attn_layer =  nn.MultiheadAttention(embedding_dim,
                                                    num_heads,
                                                    dropout=dropout_rate,
                                                    kdim=key_dim,
                                                    vdim=key_dim,
                                                    batch_first=True)
            self.attention_layers.append(new_attn_layer)

            #feed forward network to match attention output shape to rnn input size
            llos = last_layer_output_size if i==num_attn_blocks-1 else embedding_dim
            self.forward_layers.append(nn.Linear(embedding_dim, llos))


    def forward(self, key, query):
        
        self.att_weights = []
        for i in range(len(self.attention_layers)):
            if query.dim() < 3:
                query = query.unsqueeze(1)

            #attention layer
            att_outputs, att_weight = self.attention_layers[i](query, key, key, 
                                            need_weights=self.need_weights)
            
            #return attention weights
            if self.need_weights:
                self.att_weights.append(att_weight)
            
            #Add and layernorm
            query = self.attention_layernorms[i](query + att_outputs)

            #match output dim to rnn input dim
            query = self.forward_layers[i](query)

        return query

class Custom_EmbeddingBag(nn.Module):
    '''
    Customized embedding layer
    '''
    def __init__(self, n, m, dropout_rate=0):
        super(Custom_EmbeddingBag, self).__init__()
        
        #embeddingbag
        self.emb = nn.EmbeddingBag(n, m, mode="mean", sparse=True)
        
        # initialize embeddings values randomly
        W = np.random.uniform(
            low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
        ).astype(np.float32)
        self.emb.weight.data = torch.tensor(W, requires_grad=True)
        
        #embedding dropout
        self.emb_dropout = torch.nn.Dropout(p=dropout_rate)
        
    def forward(self, x, offset):

        y = self.emb(x, offset)
        y = self.emb_dropout(y)

        return y

class PARSRec(nn.Module):
    '''
    PARSRec in pytorch
    '''
    def create_mlp(self, ln_in, ln_out):
        '''
        Create MultiLayer Perceptron (MLP) layers
        '''
        LL = nn.Linear(ln_in, ln_out)

        #initiate weights randomly
        mean = 0.0
        std_dev = np.sqrt(2 / (ln_in + ln_out))
        W = np.random.normal(mean, std_dev, size=(ln_out, ln_in)).astype(np.float32)
        LL.weight.data = torch.tensor(W, requires_grad=True)

        std_dev = np.sqrt(1 / ln_out)
        bt = np.random.normal(mean, std_dev, size=ln_out).astype(np.float32)
        LL.bias.data = torch.tensor(bt, requires_grad=True)

        return LL

    def __init__(self, 
                feature_sizes=None,
                emb_dims=None, 
                loss_function='bce',
                dropout_rate=0, 
                num_attn_blocks=1,
                num_att_heads=1,
                need_weights=False,
                ):

        super(PARSRec, self).__init__()
                
        #setting params
        user_vector_size=emb_dims[0]
        input_size=emb_dims[-1]
        hidden_size=sum(emb_dims[1:])
        parsrec_output_size=emb_dims[-1]
        last_layer_output_size=emb_dims[-1]
        self.hidden_size = hidden_size
        print('PARSRec dimensions\n', '  user embedding size {}\n   hidden state size   {}\n   item embedding size {}'.format(user_vector_size , hidden_size, input_size))

        #rnn layer Linear layers
        self.i2h = self.create_mlp(user_vector_size + input_size + hidden_size, hidden_size)
        self.i2o = self.create_mlp(user_vector_size + input_size + hidden_size, parsrec_output_size)
        
        #attention layer
        self.att = Custom_Attention(num_attn_blocks=num_attn_blocks, 
                                    num_heads=num_att_heads, 
                                    dropout_rate=0.0, 
                                    embedding_dim=user_vector_size + hidden_size, 
                                    key_dim=input_size, 
                                    last_layer_output_size=input_size,
                                    need_weights=need_weights)

        #embedding layers
        if feature_sizes is not None:
            self.emb_l = nn.ModuleList()
            for (n, m) in zip(feature_sizes, emb_dims):
                self.emb_l.append(Custom_EmbeddingBag(n, m, dropout_rate=dropout_rate))

        # specify the loss function
        self.loss_function=loss_function
        loss_function_dict = {
            "mse": nn.MSELoss(reduction="mean"),
            "bce": nn.BCELoss(reduction="mean"),
            "cce": nn.CrossEntropyLoss(reduction="mean"),
        }
        if not loss_function in loss_function_dict:
            sys.exit("ERROR: --loss-function=" + self.loss_function + " is not supported")
        self.loss_fn = loss_function_dict[loss_function]
        
    def forward(self, user_vector, inputs, hidden):

        #concat user vector to hidden in each step of ARNN
        query = torch.cat((user_vector, hidden), 1)
        
        #attention layer (key=value=session_item_embs, query=concat(user_emb,hidden))
        input_i = self.att(inputs, query).squeeze(1)
        
        #rnn layer
        input_combined = torch.cat((query, input_i), 1)
        hidden = Functional.relu(self.i2h(input_combined))
        output = self.i2o(input_combined)
        
        return output, hidden, self.att.att_weights
