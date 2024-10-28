import os
import tarfile
from statistics import mode, stdev
from random import sample
import torch


ALPHABET =  "XACDEFGHIKLMNPQRSTVWY"
AMINO_ACID_TO_ID = dict((aa, id) for id, aa in enumerate(ALPHABET))
ID_TO_AMINO_ACID = dict((id, aa) for id, aa in enumerate(ALPHABET))


class ProteinDataset(object):
    def __init__(self, file_path, config):
        super(ProteinDataset).__init__()
        self.file_path = file_path
        self.config = config
        self.dataset = self.read_data()
    
    def encode_seq(self, sequence):
        integer_encoded = [AMINO_ACID_TO_ID[aa] for aa in sequence]
        onehot_encoded = list()
        for value in integer_encoded:
            letter = [0 for _ in range(self.config["num_classes"])]
            letter[value] = 1
            onehot_encoded.append(letter)
        return onehot_encoded

    def read_data(self):
        with tarfile.open(self.file_path, 'r:gz') as tar:
            member = tar.getmembers()[-1]
            f = tar.extractfile(member)
            lines = f.readlines()[1:]
            # Read sequences
            dataset = []
            for line in lines:
                upsampling, _, seq = line.split()
                seq = self.encode_seq(seq.decode("utf-8"))
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
    return d


class ProteinDataLoader(object):
    r"""Samples elements in a random order, considering the upsampling factor.

    Arguments:
        dataset (ProteinDataset): Dataset of tuples (sequence, upsampling) to sample from.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the last batch will be dropped if
            its size is less than ``batch_size``.
    """

    def __init__(self, dataset, batch_size, d_model, device):
        self.dataset = dataset
        self.batch_size = batch_size
        self.d_model = d_model
        self.device = device
    
    def pad(self, batch):
        pad = [0 for _ in range(len(batch[0][0]))]
        pad[-1] = 1
        padded_batch = []
        for seq in batch:
            padding = self.d_model - len(seq)
            if padding > 0: 
                for _ in range(padding):
                    seq.append(pad)
            else:
                seq = seq[0:self.d_model]
            seq = torch.Tensor(seq).type(torch.float32).to(self.device)
            seq = torch.transpose(seq, 0, 1) # aa, len
            seq = seq.unsqueeze(dim=1) # aa, 1, len
            padded_batch.append(seq)
        return padded_batch

    def __iter__(self):
        batch = []
        random_order = sample(range(len(self.dataset)), len(self.dataset))
        for idx in random_order:
            seq, upsampling = self.dataset[idx]
            if len(batch) + upsampling > self.batch_size:
                for _ in range(self.batch_size - len(batch)):
                    batch.append(seq)
                batch = torch.stack(self.pad(batch), dim=0).to(self.device)
                yield batch
                batch = []
            elif len(batch) + upsampling <= self.batch_size:
                for _ in range(upsampling):
                    batch.append(seq)
                if len(batch) == self.batch_size:
                    batch = torch.stack(self.pad(batch), dim=0).to(self.device)
                    yield batch
                    batch = []


def get_protein_sequences(sequences):
    fasta_sequences = []
    for i, seq in enumerate(sequences):
        seq = seq.squeeze() # aa, len
        seq = torch.transpose(seq, 0, 1) # len, aa
        _, indices = torch.max(seq, 1)
        protein_seq = "".join([ID_TO_AMINO_ACID[id.item()] for id in indices])
        record = "{}{}{}{}".format(">", "seq" + str(i), os.linesep, protein_seq)
        fasta_sequences.append(record)
    return fasta_sequences
