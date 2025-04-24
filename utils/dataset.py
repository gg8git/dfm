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
        return torch.utils.data.DataLoader(self.train, batch_size=self.batch_size, pin_memory=True, shuffle=True, collate_fn=kmer_collate_fn, num_workers=10)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val,   batch_size=self.batch_size, pin_memory=True, shuffle=False, collate_fn=kmer_collate_fn, num_workers=10)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test,   batch_size=self.batch_size, pin_memory=True, shuffle=False, collate_fn=kmer_collate_fn, num_workers=10)


class KmerDataset(torch.utils.data.Dataset): # asssuming train data 
    def __init__(self, dataset='train', data_path=None, k=3, vocab=None, vocab2idx=None, load_data=False):
        if data_path is None: 
            path_to_data = 'data/uniref-cropped.csv' 
        df = pd.read_csv(path_to_data )
        self.dataset = dataset
        train_seqs = df['sequence'].values  # 4_500_000  sequences 
        # SEQUENCE LENGTHS ANALYSIS:  Max = 299, Min = 100, Mean = 183.03 

        self.alphabet_size = 22 # 27/28 depending on padding
        self.num_cls = 1
        self.k = k
        regular_data = [] 
        for seq in train_seqs: 
            regular_data.append([token for token in seq]) # list of tokens
        
        # first get initial vocab set 
        if vocab is None:
            self.regular_vocab = set((token for seq in regular_data for token in seq))  # 21 tokens 
            self.regular_vocab.discard(".") 
            self.regular_vocab.discard("B") 
            self.regular_vocab.discard("O") 
            self.regular_vocab.discard("U") 
            self.regular_vocab.discard("X") 
            self.regular_vocab.discard("Z") 
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
        
        # map irrelevant tokens to pad (?) token
        for c in "BOUXZ":
            self.vocab2idx[c] = self.vocab2idx['<stop>']
        
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


def kmer_collate_fn(data):
    # Length of longest peptide in batch 
    max_size = max([x.shape[-1] for x in data])
    batch_x = torch.vstack(
        # Pad with stop token
        [torch.nn.functional.pad(x, (0, max_size - x.shape[-1]), value=1) for x in data] # value=0/1
    )
    return batch_x - 1, torch.zeros(batch_x.shape[0], dtype=torch.int64) # batch_x - 0/1


class MoleculeDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, k, version=1, load_data=True ): 
        super().__init__() 
        self.batch_size = batch_size 

        self.full_dataset = MoleculeDataset('../cfm/data/guacamol_v1_train.selfies', max_length=128)
        self.full_dataset.save_vocabulary('./workdir/molecule-dfm')
        val_size = min(10000, len(self.full_dataset) // 10)  # Ensure val set is at most 10% of data
        train_size = len(self.full_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(self.full_dataset, [train_size, val_size])
        
        self.train  = train_dataset
        self.val    = val_dataset
        self.test   = val_dataset
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, batch_size=self.batch_size, pin_memory=True, shuffle=True, num_workers=10, drop_last=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val,   batch_size=self.batch_size, pin_memory=True, shuffle=False, num_workers=10, drop_last=True)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test,   batch_size=self.batch_size, pin_memory=True, shuffle=False, num_workers=10, drop_last=True)

import re
SELFIES_PATTERN = r'\[[^\]]*\]'
STOP_TOKEN = "[STOP]"  # Custom stop token for SELFIES

class MoleculeDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, max_length=200, stop_token=STOP_TOKEN):
        """
        Dataset for SELFIES molecular representations.
        
        Args:
            file_path: Path to file containing SELFIES strings (one per line)
            max_length: Maximum number of tokens in a sequence
            stop_token: Token used to mark the end of a sequence
        """
        self.max_length = max_length
        self.stop_token = stop_token
        
        # Load SELFIES strings from file
        with open(file_path, 'r') as f:
            self.selfies_strings = [line.strip() for line in f]
        
        print(f"Loaded {len(self.selfies_strings)} SELFIES strings from {file_path}")
        print(f"Example SELFIES: {self.selfies_strings[0]}")
        
        # Build vocabulary from all SELFIES strings
        self.build_vocabulary()
        
    def build_vocabulary(self):
        """Build vocabulary from all SELFIES strings in the dataset."""
        # Find all unique tokens across all SELFIES strings
        all_tokens = set()
        for selfies in self.selfies_strings:
            tokens = re.findall(SELFIES_PATTERN, selfies)
            all_tokens.update(tokens)
        
        # Add stop token
        all_tokens.add(self.stop_token)
        
        # Sort tokens to ensure consistent ordering
        all_tokens = sorted(list(all_tokens))
        
        # Create mapping dictionaries
        self.token_to_idx = {token: idx for idx, token in enumerate(all_tokens)}
        self.idx_to_token = {idx: token for idx, token in enumerate(all_tokens)}
        self.vocab_size = len(all_tokens)
        self.alphabet_size = self.vocab_size
        
        print(f"Vocabulary built with {self.vocab_size} unique tokens")
        print(f"First 10 tokens: {list(all_tokens)[:10]}")
        
    def save_vocabulary(self, vocab_path):
        """Save vocabulary to file for later use."""
        vocab_data = {
            'token_to_idx': self.token_to_idx,
            'idx_to_token': self.idx_to_token,
            'vocab_size': self.vocab_size
        }
        
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab_data, f)
        
        print(f"Vocabulary saved to {vocab_path}")
    
    def __len__(self):
        return len(self.selfies_strings)
    
    def __getitem__(self, idx):
        selfies = self.selfies_strings[idx]
        
        # Tokenize SELFIES string
        tokens = re.findall(SELFIES_PATTERN, selfies)
        
        # Truncate if needed
        if len(tokens) > self.max_length - 1:  # Leave room for at least one stop token
            tokens = tokens[:self.max_length-1]
        
        # Convert to indices
        indices = [self.token_to_idx[token] for token in tokens]
        
        # Add stop token
        indices.append(self.token_to_idx[self.stop_token])
        
        # Pad with stop tokens to fixed length
        indices += [self.token_to_idx[self.stop_token]] * (self.max_length - len(indices))
        
        return torch.tensor(indices)

import numpy as np
import selfies as sf
import torch
import torch.nn.functional as F

class SELFIESDataModule(pl.LightningDataModule):
    def __init__(self, batch_size): 
        super().__init__() 
        self.batch_size = batch_size 

        self.dataset = SELFIESDataset(fname='../cfm/data/guacamol_v1_train.selfies')
        print('VOCAB SIZE', self.dataset.vocab_size)
        val_size = min(10000, len(self.dataset) // 10)  
        train_size = len(self.dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(self.dataset, [train_size, val_size])
        
        self.train  = train_dataset
        self.val    = val_dataset
        self.test   = val_dataset
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, batch_size=self.batch_size, pin_memory=True, shuffle=True, collate_fn=collate_fn, num_workers=10, drop_last=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val,   batch_size=self.batch_size, pin_memory=True, shuffle=False, collate_fn=collate_fn, num_workers=10, drop_last=True)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test,   batch_size=self.batch_size, pin_memory=True, shuffle=False, collate_fn=collate_fn, num_workers=10, drop_last=True)


# DEFAULT_SELFIES_VOCAB = ['<start>', '<stop>',] + list(sf.ge1t_semantic_robust_alphabet()) + ["[NH1]","[NH1+1]", "[Cl+1]", "[Si]", "[PH1]", "[Se]"]

class SELFIESDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        fname=None,
    ):
        self.data = []
        assert fname is not None
        with open(fname, 'r') as f:
            selfie_strings = [line.strip() for line in f]
        for string in selfie_strings:
            self.data.append(list(sf.split_selfies(string)))
        self.vocab = set(token for selfie in self.data for token in selfie)
        self.vocab.discard(".")
        self.vocab = ["<start>", "<stop>", *sorted(list(self.vocab))]
            
        DEFAULT_SELFIES_VOCAB = [
            "<start>",
            "<stop>",
            "[#Branch1]",
            "[#Branch2]",
            "[#C-1]",
            "[#C]",
            "[#N+1]",
            "[#N]",
            "[#O+1]",
            "[=B]",
            "[=Branch1]",
            "[=Branch2]",
            "[=C-1]",
            "[=C]",
            "[=N+1]",
            "[=N-1]",
            "[=NH1+1]",
            "[=NH2+1]",
            "[=N]",
            "[=O+1]",
            "[=OH1+1]",
            "[=O]",
            "[=PH1]",
            "[=P]",
            "[=Ring1]",
            "[=Ring2]",
            "[=S+1]",
            "[=SH1]",
            "[=S]",
            "[=Se+1]",
            "[=Se]",
            "[=Si]",
            "[B-1]",
            "[BH0]",
            "[BH1-1]",
            "[BH2-1]",
            "[BH3-1]",
            "[B]",
            "[Br+2]",
            "[Br-1]",
            "[Br]",
            "[Branch1]",
            "[Branch2]",
            "[C+1]",
            "[C-1]",
            "[CH1+1]",
            "[CH1-1]",
            "[CH1]",
            "[CH2+1]",
            "[CH2]",
            "[C]",
            "[Cl+1]",
            "[Cl+2]",
            "[Cl+3]",
            "[Cl-1]",
            "[Cl]",
            "[F+1]",
            "[F-1]",
            "[F]",
            "[H]",
            "[I+1]",
            "[I+2]",
            "[I+3]",
            "[I]",
            "[N+1]",
            "[N-1]",
            "[NH0]",
            "[NH1+1]",
            "[NH1-1]",
            "[NH1]",
            "[NH2+1]",
            "[NH3+1]",
            "[N]",
            "[O+1]",
            "[O-1]",
            "[OH0]",
            "[O]",
            "[P+1]",
            "[PH1]",
            "[PH2+1]",
            "[P]",
            "[Ring1]",
            "[Ring2]",
            "[S+1]",
            "[S-1]",
            "[SH1]",
            "[S]",
            "[Se+1]",
            "[Se-1]",
            "[SeH1]",
            "[SeH2]",
            "[Se]",
            "[Si-1]",
            "[SiH1-1]",
            "[SiH1]",
            "[SiH2]",
            "[Si]",
            "[=Cl-1]",
            "[OH1+1]",
            "[=Br-1]",
            "[#Br-1]",
            "[=OH0]",
            "[=SiH1]",
            "[=I+2]",
            "[=CH1]",
            "[=SeH2]",
            "[=BH2-1]",
            "[=SiH2]",
            "[#PH1]",
            "[=Br+2]",
            "[=F+1]",
            "[=NH1]",
            "[=Cl+3]",
            "[=SiH2]",
            "[#SeH2]",
            "[=I+3]",
            "[=Se-1]",
            "[#Se]",
            "[#Se+1]",
            "[#NH0]",
            "[#SiH2]",
            "[=NH0]",
            "[=SeH1]",
            "[#I+2]",
            "[#CH1]",
            "[#Cl+2]",
            "[#Cl+1]",
            "[#F+1]",
            "[=SiH1-1]",
            "[=Si-1]",
            "[=PH2+1]",
            "[#Ring1]",
            "[=Cl+1]",
            "[#SiH1-1]",
            "[=CH2+1]",
            "[#Se-1]",
            "[#PH2+1]",
            "[#Si]",
            "[=Cl+2]",
            "[#I+3]",
            "[#NH1+1]",
            "[#Br+2]",
            "[#SeH1]",
            "[=BH0]",
            "[=CH1+1]",
            "[=I+1]",
            "[#CH1+1]",
            "[=CH2]",
            "[#BH0]",
            "[#CH2+1]",
            "[#I+1]",
            "[=CH2]",
            "[#SiH1]",
            "[#Cl-1]",
            "[=CH1-1]",
            "[=BH1-1]",
            "[=F-1]",
            "[#Si-1]",
            "[#F-1]",
            "[#BH1-1]",
            "[#Cl+3]",
            "[#Ring2]",
        ]

        DEFAULT_SELFIES_VOCAB = (
            DEFAULT_SELFIES_VOCAB
            + list(sf.get_semantic_robust_alphabet() - set(DEFAULT_SELFIES_VOCAB))
            + [
                "[#SH1]",
            ]
        )

        self.vocab = DEFAULT_SELFIES_VOCAB

        self.vocab2idx = {v: i for i, v in enumerate(self.vocab)}
        self.idx2vocab = dict(zip(range(len(self.vocab)), self.vocab))

    def tokenize_selfies(self, selfies_list):
        tokenized_selfies = []
        for string in selfies_list:
            tokenized_selfies.append(list(sf.split_selfies(string)))
        return tokenized_selfies

    def encode(self, smiles, maxl=None):
        if type(smiles[0]) == list:
            maxl = max([len(x) for x in smiles] + ([0] if maxl == None else [maxl]))

            smiles_pad = [x + ["<stop>"] * (maxl - len(x)) for x in smiles]
            return torch.tensor(np.vectorize(self.vocab2idx.get)(np.array(smiles_pad)))
        else:
            return torch.tensor([self.vocab2idx[s] for s in [*smiles, "<stop>"]])

    def decode(self, tokens):
        if tokens.dim() == 2:
            dec = np.vectorize(self.idx2vocab.get)(tokens.cpu())
            dec[torch.tensor(dec == "<stop>").cummax(dim=-1).values] = ""
            return ["".join(x) for x in dec]
        else:
            dec = [self.vocab[t] for t in tokens]  # type: ignore
            # Chop out start token and everything past (and including) first stop token
            stop = dec.index("<stop>") if "<stop>" in dec else None  # want first stop token
            selfie = dec[0:stop]  # cut off stop tokens
            while (
                "<start>" in selfie
            ):  # start at last start token (I've seen one case where it started w/ 2 start tokens)
                start = 1 + dec.index("<start>")
                selfie = selfie[start:]
            selfie = "".join(selfie)
            return selfie

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.encode(self.data[idx])

    @property
    def vocab_size(self):
        return len(self.vocab)

def collate_fn(data):
    # Length of longest molecule in batch
    max_size = max(max([x.shape[-1] for x in data]), 128)
    return torch.vstack(
        # Pad with stop token
        [F.pad(x, (0, max_size - x.shape[-1]), value=1) for x in data]
    )
