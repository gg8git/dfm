import copy
import pickle

import pytorch_lightning as pl
import torch, esm, random, os, json
import numpy as np
from Bio import SeqIO

import pandas as pd
import itertools
from tqdm import tqdm


class EnhancerDataset(torch.utils.data.Dataset):
    def __init__(self, args, split='train'):
        all_data = pickle.load(open(f'data/the_code/General/data/Deep{"MEL2" if args.mel_enhancer else "FlyBrain"}_data.pkl', 'rb'))
        self.seqs = torch.argmax(torch.from_numpy(copy.deepcopy(all_data[f'{split}_data'])), dim=-1)
        self.clss = torch.argmax(torch.from_numpy(copy.deepcopy(all_data[f'y_{split}'])), dim=-1)
        self.num_cls = all_data[f'y_{split}'].shape[-1]
        self.alphabet_size = 4

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.clss[idx]


class TwoClassOverfitDataset(torch.utils.data.IterableDataset):
    def __init__(self, args):
        super().__init__()
        self.seq_len = args.toy_seq_len
        self.alphabet_size = args.toy_simplex_dim
        self.num_cls = 2

        if args.cls_ckpt is not None:
            distribution_dict = torch.load(os.path.join(os.path.dirname(args.cls_ckpt), 'overfit_dataset.pt'))
            self.data_class1 = distribution_dict['data_class1']
            self.data_class2 = distribution_dict['data_class2']
        else:
            self.data_class1 = torch.stack([torch.from_numpy(np.random.choice(np.arange(self.alphabet_size), size=args.toy_seq_len, replace=True)) for _ in range(args.toy_num_seq)])
            self.data_class2 = torch.stack([torch.from_numpy(np.random.choice(np.arange(self.alphabet_size), size=args.toy_seq_len, replace=True)) for _ in range(args.toy_num_seq)])
            distribution_dict = {'data_class1': self.data_class1, 'data_class2': self.data_class2}
        torch.save(distribution_dict, os.path.join(os.environ["MODEL_DIR"], 'overfit_dataset.pt'))

    def __len__(self):
        return 10000000000

    def __iter__(self):
        while True:
            if np.random.rand() < 0.5:
                yield self.data_class1[np.random.choice(np.arange(len(self.data_class1)))], torch.tensor([0])
            else:
                yield self.data_class2[np.random.choice(np.arange(len(self.data_class2)))], torch.tensor([1])

class ToyDataset(torch.utils.data.IterableDataset):
    def __init__(self, args):
        super().__init__()
        self.num_cls = args.toy_num_cls
        self.seq_len = args.toy_seq_len
        self.alphabet_size = args.toy_simplex_dim

        if args.cls_ckpt is not None:
            distribution_dict = torch.load(os.path.join(os.path.dirname(args.cls_ckpt), 'toy_distribution_dict.pt'))
            self.probs = distribution_dict['probs']
            self.class_probs = distribution_dict['class_probs']
        else:
            self.probs = torch.softmax(torch.rand((self.num_cls, self.seq_len, self.alphabet_size)), dim=2)
            self.class_probs = torch.ones(self.num_cls)
            if self.num_cls > 1:
                self.class_probs = self.class_probs * 1 / 2 / (self.num_cls - 1)
                self.class_probs[0] = 1 / 2
            assert self.class_probs.sum() == 1

            distribution_dict = {'probs': self.probs, 'class_probs': self.class_probs}
        torch.save(distribution_dict, os.path.join(os.environ["MODEL_DIR"], 'toy_distribution_dict.pt' ))

    def __len__(self):
        return 10000000000
    def __iter__(self):
        while True:
            cls = np.random.choice(a=self.num_cls,size=1,p=self.class_probs)
            seq = []
            for i in range(self.seq_len):
                seq.append(torch.multinomial(replacement=True,num_samples=1,input=self.probs[cls,i,:]))
            yield torch.tensor(seq), cls

class KmerDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, k, version=1, load_data=True ): 
        super().__init__() 
        self.batch_size = batch_size 
        if version == 1: DatasetClass = KmerDataset
        else: raise RuntimeError('Invalid data version') 
        self.train  = DatasetClass(dataset='train', k=k, load_data=load_data) 
        self.val    = DatasetClass(dataset='val', k=k, vocab=self.train.vocab, vocab2idx=self.train.vocab2idx, load_data=load_data )
        self.test   = DatasetClass(dataset='test', k=k, vocab=self.train.vocab, vocab2idx=self.train.vocab2idx, load_data=load_data )
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, batch_size=self.batch_size, pin_memory=True, shuffle=True, collate_fn=collate_fn, num_workers=10)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val,   batch_size=self.batch_size, pin_memory=True, shuffle=False, collate_fn=collate_fn, num_workers=10)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test,   batch_size=self.batch_size, pin_memory=True, shuffle=False, collate_fn=collate_fn, num_workers=10)


class KmerDataset(torch.utils.data.Dataset): # asssuming train data 
    def __init__(self, dataset='train', data_path=None, k=3, vocab=None, vocab2idx=None, load_data=False):
        if data_path is None: 
            path_to_data = 'data/uniref-cropped.csv' 
        df = pd.read_csv(path_to_data )
        self.dataset = dataset
        train_seqs = df['sequence'].values  # 4_500_000  sequences 
        # SEQUENCE LENGTHS ANALYSIS:  Max = 299, Min = 100, Mean = 183.03 

        self.k = k
        regular_data = [] 
        for seq in train_seqs: 
            regular_data.append([token for token in seq]) # list of tokens
        
        # first get initial vocab set 
        if vocab is None:
            self.regular_vocab = set((token for seq in regular_data for token in seq))  # 21 tokens 
            self.regular_vocab.discard(".") 
            if '-' not in self.regular_vocab: 
                self.regular_vocab.add('-')  # '-' used as pad token when length of sequence is not a multiple of k
            self.vocab = ["".join(kmer) for kmer in itertools.product(self.regular_vocab, repeat=k)] # 21**k tokens 
            self.vocab = ['<start>', '<stop>', *sorted(list(self.vocab))] # 21**k + 2 tokens 
        else: 
            self.vocab = vocab 

        if vocab2idx is None:
            self.vocab2idx = { v:i for i, v in enumerate(self.vocab) }
        else:
            self.vocab2idx = vocab2idx
        
        self.data = []
        if load_data:
            for seq in tqdm(regular_data):
                token_num = 0
                kmer_tokens = []
                while token_num < len(seq):
                    kmer = seq[token_num:token_num+k]
                    while len(kmer) < k:
                        kmer += '-' # padd so we always have length k 
                    kmer_tokens.append("".join(kmer)) 
                    token_num += k 
                self.data.append(kmer_tokens) 
        
        num_data = len(self.data) 
        ten_percent = int(num_data/10) 
        five_percent = int(num_data/20) 
        if self.dataset == 'train': # 90 %
            self.data = self.data[0:-ten_percent] 
        elif self.dataset == 'val': # 5 %
            self.data = self.data[-ten_percent:-five_percent] 
        elif self.dataset == 'test': # 5 %
            self.data = self.data[-five_percent:] 
        else: 
            raise RuntimeError("dataset must be one of train, val, test")


    def tokenize_sequence(self, list_of_sequences):   
        ''' 
        Input: list of sequences in standard form (ie 'AGYTVRSGCMGA...')
        Output: List of tokenized sequences where each tokenied sequence is a list of kmers
        '''
        tokenized_sequences = []
        for seq in list_of_sequences:
            token_num = 0
            kmer_tokens = []
            while token_num < len(seq):
                kmer = seq[token_num:token_num + self.k]
                while len(kmer) < self.k:
                    kmer += '-' # padd so we always have length k  
                if type(kmer) == list: kmer = "".join(kmer)
                kmer_tokens.append(kmer) 
                token_num += self.k 
            tokenized_sequences.append(kmer_tokens) 
        return tokenized_sequences 

    def encode(self, tokenized_sequence):
        return torch.tensor([self.vocab2idx[s] for s in [*tokenized_sequence, '<stop>']])

    def decode(self, tokens):
        '''
        Inpput: Iterable of tokens specifying each kmer in a given protien (ie [3085, 8271, 2701, 2686, ...] )
        Output: decoded protien string (ie GYTVRSGCMGA...)
        '''
        dec = [self.vocab[t] for t in tokens]
        # Chop out start token and everything past (and including) first stop token
        stop = dec.index("<stop>") if "<stop>" in dec else None # want first stop token
        protien = dec[0:stop] # cut off stop tokens
        while "<start>" in protien: # start at last start token (I've seen one case where it started w/ 2 start tokens)
            start = (1+dec.index("<start>")) 
            protien = protien[start:]
        protien = "".join(protien) # combine into single string 
        return protien

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.encode(self.data[idx]) 

    @property
    def vocab_size(self):
        return len(self.vocab)


def collate_fn(data):
    # Length of longest peptide in batch 
    max_size = max([x.shape[-1] for x in data])
    batch_x = torch.vstack(
        # Pad with stop token
        [torch.nn.functional.pad(x, (0, max_size - x.shape[-1]), value=1) for x in data]
    )
    return batch_x - 1, torch.zeros(batch_x.shape[0], dtype=torch.int64)

    