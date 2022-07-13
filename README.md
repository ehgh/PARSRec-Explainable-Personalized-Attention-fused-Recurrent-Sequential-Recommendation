# PARSRec: Explainable Personalized Attention-fused Recurrent Sequential Recommendation Using Session Partial Actions

This repository provides a reference implementation of the model and data generation of the following paper:

> PARSRec: Explainable Personalized Attention-fused Recurrent Sequential Recommendation Using Session Partial Actions
> Ehsan Gholami, Mohammad Motamedi, Ashwin Aravindakshan, 2022.
>  
> %Open access link to the paper coming soon

This code provides tools to:
1. Generate the synthetic dataset described in the paper
2. Generate binary Dataset and Dataloader for *any dataset with specified file format*
3. Train and evaluate the PARSRec model on *any dataset*

For inquiries, don't hesitate to get in touch with Ehsan Gholami (contact: egholami@ucdavis.edu). Please feel free to file an issue if one is detected.

## Citing

If you find this code/data useful for your research, please consider citing the following paper:
> Ehsan Gholami, Mohammad Motamedi, and Ashwin Aravindakshan. 2022. PARSRec: Explainable Personalized Attention-fused Recurrent Sequential Recommendation Using Session Partial Actions. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’22),  August 14–18, 2022, Washington, DC, USA. ACM, New York, NY, USA, 11 pages. https://doi.org/10.1145/3534678.3539432

## Architecture Image


<img src="https://user-images.githubusercontent.com/17379116/172081336-f1d4372d-edee-4fe8-800c-956850e10c8d.jpg" width="400">


## Basic Usage
### 1. Generate Synthetic data
To generate the synthetic dataset, choose the argument values in the *"synthetic_data_generator.sh"* file and run this command in the terminal:

    ./synthetic_data_generator.sh

This will generate the synthetic dataset in the default mode folder *"data/"*. You can set the data directory via the *"-data-directory"* argument.

To get the list of parameters and their definitions, run this line in the terminal:

    python3 synthetic_data_generator.py --help
    
If you need to make the *"synthetic_data_generator.sh"* file executable on your device, in terminal run:
 
    chmod +x synthetic_data_generator.sh

### 2. Train PARSRec

To train PARSRec on a dataset, set the arguments in the *"PARSRec.sh"* file and run this command in the terminal:

    ./PARSRec.sh

The code saves the output files in the default mode folder *"output/"*.

#### How to train on arbitrary data:

#### Dataset file format:

Create two files in folder *"data/"* with following names:

> \<dataset name in PARSRec.sh\>.txt
> 
> \<dataset name in PARSRec.sh\>_columns_dtype.json

Where *\<dataset name in PARSRec.sh\>* is the dataset name you set for argument *"dataset"* in *"PARSRec.sh"* file (default is *"synthetic"*)

##### Format of '\<dataset name in PARSRec.sh\>.txt' file:

- Header line:
    
    user_id,time,session_length,session_items
    
- Data lines (one user-session per line):

#e.g. line for (user_id: 1 ,time: 5 ,session_length: 6,session_items: 3,7,4,8,12,34) is:

    1,5,6,3,7,4,8,12,34
    
**NOTE**: item_ids must start from 2 (0 and 1 are reserved for *SOB* and *EOB*, respectively)

Please refer to 'data/synthetic.txt' for an example file format.

#### Content of '\<dataset name in PARSRec.sh\>_columns_dtype.json' file:

The data type of your content for binary dataset generator. Always use *"object"* for *"session_items"* and *"history"*. The rest can be user-defined dtypes.

    {
        "session_length":"int16",
        "session_items":"object", 
        "history":"object", 
        "time":"int", 
        "user_id":"int32"
    }

And finally, set the flag ```--convert-dataset2binary``` to *True* in the ```PARSRec.sh``` file and run the ```./PARSRec.sh``` in the terminal. You only need to activate it once. After a single run, the binary files are generated and saved to the folder *"data"*, and you can deactivate the ```--convert-dataset2binary``` flag in the next runs.

#### Output image in the 'output' folder

This is output for sample synthetic dataset of 1024 users, 100 sessions per user, and 2000 items (in *"output"* folder):

<img src="output/perf_loss.png" width="400">

## Abstract

The emerging meta- and multi-verse landscape is yet another step towards the more prevalent use of already ubiquitous online markets. In such markets, recommender systems play critical roles by offering items of interest to the users, thereby narrowing down a vast search space that comprises hundreds of thousands of products. Recommender systems are usually designed to learn common user behaviors and rely on them for inference. This approach, while effective, is oblivious to subtle idiosyncrasies that differentiate humans from each other. Focusing on this observation, we propose an architecture that relies on common patterns as well as individual behaviors to tailor its recommendations for each person. Simulations under a controlled environment show that our proposed model learns interpretable personalized user behaviors. Our empirical results on Nielsen Consumer Panel dataset indicate that the proposed approach achieves up to 27.9% performance improvement compared to the state-of-the-art.

