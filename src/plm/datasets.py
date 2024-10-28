import torch
import tarfile
import numpy as np
from random import sample, randint
from statistics import mode, stdev
from vocabulary import TOKENS, EC_NUMBERS


class ProteinSequenceTokenizer(object):
    def __init__(self, sequence_cache, tags_cache, pad_length=None, masking=False):
        self.masking = masking
        self.pad_length = self.get_pad_length(sequence_cache, pad_length)
        self.base2index = dict((base, idx) for idx, base in enumerate(TOKENS))
        self.num_tokens = len(set(self.base2index.values()))
        self.tag2index = dict((tag, idx) for idx, tag in enumerate(np.unique(list(tags_cache.values())), self.num_tokens))
        self.num_tokens = self.num_tokens + len(self.tag2index)
        self.noise_distribution = (self.get_noise_distribution(sequence_cache) if masking else None)

    def get_pad_length(self, sequence_cache, pad_length, max_length = 512):
        if pad_length is not None:
            return pad_length
        lenghts = [ len(sequence_cache[index]) for index in sequence_cache ]
        md = mode(lenghts)
        sd = round(stdev(lenghts))
        pad_length = md + sd
        if pad_length > max_length:
            return max_length
        return pad_length

    def get_noise_distribution(self, sequence_cache):
        sequences = list(sequence_cache.values())
        concatenated_seqs = np.concatenate([list(seq) for seq in sequences])
        aa, counts = np.unique(concatenated_seqs, return_counts=True)
        tokens = [self.base2index[a] for a in aa]
        probs = counts / counts.sum()
        return tokens, probs

    def pad(self, sequence, pad_length):
        padding = pad_length - len(sequence)
        if padding > 0:
            return sequence + ([self.base2index[b'PAD']] * padding)
        j = randint(0, len(sequence) - pad_length)
        return sequence[j:j+pad_length]

    def mask(self, sequence, p=0.1):
        sequence = torch.tensor(sequence, dtype=torch.long)
        mask = torch.rand(len(sequence)) < p
        non_padding_mask = (sequence != self.base2index[b'PAD']) & (sequence < len(TOKENS))
        mask = mask & non_padding_mask
        masked_positions = torch.where(mask, sequence, self.base2index[b'MASK'])
        tokens, probs = self.noise_distribution
        noise = torch.tensor(np.random.choice(tokens, len(sequence), replace=True, p=probs))
        masked_sequence = torch.where(mask, noise, sequence)
        return torch.stack((masked_sequence, masked_positions))

    def encode(self, sequence, tags):
        sequence = [self.base2index[b'START']] + [self.base2index[aa] for aa in sequence] + [self.base2index[b'END']]
        sequence = self.pad(sequence, self.pad_length)
        if tags is not None:
            tags = [ self.tag2index[tag.decode()] for tag in tags ]
            sequence = tags + sequence 
        if self.masking:
            sequence = self.mask(sequence)
        else:
            sequence = torch.tensor(sequence, dtype=torch.long)
        return sequence


class ProteinSequenceDataset(object):
    def __init__(self, file_path, file_format, pad_length=None, upsampling=True, masking=False):
        self.sequence_cache, self.tags_cache, self.indices_cache = self.read_file(file_path, file_format, upsampling)
        self.tokenizer = ProteinSequenceTokenizer(self.sequence_cache, self.tags_cache, pad_length, masking)
        self.pad_length = self.tokenizer.pad_length
        self.num_tokens = self.tokenizer.num_tokens

    def read_file(self, file_path, file_format, upsampling):
        tags = [ tag for tag in file_format if tag not in ["sequence", "upsampling"] ]
        with tarfile.open(file_path, 'r:gz') as tar:
            member = tar.getmembers()[-1]
            f = tar.extractfile(member)
            lines = f.readlines()[1:]
            # Read sequences
            sequences = [ line.strip().split(b"\t")[file_format["sequence"]] for line in lines ]
            for t in range(len(tags)):
                tags[t] = [ line.strip().split(b"\t")[file_format[tags[t]]] for line in lines ]
            # Set sequence order
            random_order = sample(range(len(lines)), len(lines))
            if upsampling:
                factors = [ int(line.strip().split(b"\t")[file_format["upsampling"]]) for line in lines ]
                ordered_factors = [ factors[i] for i in random_order ]
                ordered_indices = sum([[index] * ordered_factors[i] for i, index in enumerate(random_order)], [])
                return dict(enumerate(sequences)), dict(enumerate(zip(*tags))), dict(zip(range(sum(ordered_factors)), ordered_indices))
            return dict(enumerate(sequences)), dict(enumerate(zip(*tags))), dict(enumerate(random_order))

    def __len__(self):
        return len(self.indices_cache)

    def __getitem__(self, index):
        real_index = self.indices_cache[index]
        sequence = self.sequence_cache[real_index]
        tags = (self.tags_cache[real_index] if bool(self.tags_cache) else None)
        encoded_sequence = self.tokenizer.encode(sequence, tags)
        return encoded_sequence