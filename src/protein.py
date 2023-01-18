import os
from random import sample
from statistics import mode, stdev
import torch
                        
class ProteinDataset(object):
    def __init__(self, filename: str, config: dict):
        super(ProteinDataset).__init__()
        self.filename = filename
        self.config = config
        self.dataset = self.read_data()
    
    def tokenize(self, sequence):
        base2idx = {'0':0, '1':1, 'A':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'H':8, 'I':9, 'K':10, 'L':11, 
                        'M':12, 'N':13, 'P':14, 'Q':15, 'R':16, 'S':17, 'T':18, 'V':19, 'W':20, 'Y':21, 'X':22}
        ids = [ base2idx[aa] for aa in sequence ]
        return ids

    def read_data(self):
        file = open(os.path.join(self.config["data_dir"], self.filename))
        lines = file.readlines()[1:]
        dataset = []
        for line in lines:
            _, upsampling, _, _, seq = line.split()
            seq = self.tokenize(seq)
            dataset.append((seq, upsampling))
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        seq = self.dataset[idx][0]
        upsampling = int(self.dataset[idx][1])
        return seq, upsampling


def get_d_model(train_dataset, val_dataset, max_d_model):
    lenghts = [len(seq[0]) for seq in train_dataset] + [len(seq[0]) for seq in val_dataset]
    md = mode(lenghts)
    sd = round(stdev(lenghts))
    d = md + sd
    if d > max_d_model:
        return max_d_model
    return d + 1 # considering start/end tokens


class ProteinDataLoader(object):
    '''
    Samples elements in a random order to for batches, considering the upsampling factor.
    '''

    def __init__(self, dataset: ProteinDataset, batch_size: int, d_model: int, device: torch.device):
        self.dataset = dataset
        self.batch_size = batch_size
        self.d_model = d_model
        self.device = device
    
    def pad(self, batch):
        inputs, targets = [], []
        for seq in batch:
            padding = self.d_model - len(seq) - 1
            if padding > 0:
                for i in range(padding):
                    seq.append(22)
            else:
                seq = seq[0:self.d_model - 1]
            input = seq + [1] 
            input = torch.Tensor(input).type(torch.int32).to(self.device)
            inputs.append(input)
            
            target = [0] + seq + [1]
            target = torch.Tensor(target).type(torch.int32).to(self.device)
            targets.append(target)
        
        inputs = torch.stack(inputs, dim=0).to(self.device)
        targets = torch.stack(targets, dim=0).to(self.device)
        return [inputs, targets]

    def __iter__(self):
        batch = []
        random_order = sample(range(len(self.dataset)), len(self.dataset))
        for idx in random_order:
            seq, upsampling = self.dataset[idx]
            if len(batch) + upsampling > self.batch_size:
                for _ in range(self.batch_size - len(batch)):
                    batch.append(seq)
                yield self.pad(batch)
                batch = []
            elif len(batch) + upsampling <= self.batch_size:
                for _ in range(upsampling):
                    batch.append(seq)
                if len(batch) == self.batch_size:
                    yield self.pad(batch)
                    batch = []   


#idx2base = {0:'0', 1:'1', 2:'A', 3:'C', 4:'D', 5:'E', 6:'F', 7:'G', 8:'H', 9:'I', 10:'K', 11:'L', 
                #12:'M', 13:'N', 14:'P', 15:'Q', 16:'R', 17:'S', 18:'T', 19:'V', 20:'W', 21:'Y', 22:'X'}