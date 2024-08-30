import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        # TODO
        super(T5Dataset, self).__init__()
        self.tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
        
        self.data = self.process_data(data_folder, split, self.tokenizer)

        self.len = 0
        sefl.eval = False



    def process_data(self, data_folder, split, tokenizer):
        self.eval = split == 'test'

        nl_file = f'{data_folder}/{split}.nl'
        sql_file = f'{data_folder}/{split}.sql' if split != 'test' else None

        with open(nl_file, 'r') as f_nl:
            nl_lines = [line.strip() for line in f_nl.readlines()]

        if sql_file:
            with open(sql_file, 'r') as f_sql:
                sql_lines = [line.strip() for line in f_sql.readlines()]
            assert len(nl_lines) == len(sql_lines), "Mismatch between number of NL and SQL lines"

        self.len = len(nl_lines)
        inputs = tokenizer(nl_lines, padding='max_length', truncation=True, return_tensors="pt")

        if split == 'test':
            return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']}

        labels = tokenizer(sql_lines, padding='max_length', truncation=True, return_tensors="pt")
        return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'], 'labels': labels['input_ids']}
    
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        assert self.__len__ >= idx
        ordered_keys = ['input_ids', 'attention_mask', 'labels']
        if self.split == 'test':
            ordered_keys = ['input_ids', 'attention_mask']
        return [self.data[key][idx] for key in ordered_keys]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO
    return [], [], [], [], []

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO
    return [], [], []

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    return train_x, train_y, dev_x, dev_y, test_x