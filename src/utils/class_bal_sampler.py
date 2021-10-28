import torch
from torch.utils.data.sampler import Sampler

class RandomBatchedWeightedSampler(Sampler):
    ########################################################
    ###### Samples batches randomly, with replacement ######
    ###### Items in batch are not shuffled            ######
    ###### Weights samples based on loss score        ######
    ###### Weights can be updated after each epoch    ######
    ########################################################

    def __init__(self, data_source, batch_size, seq_len, weights):
        self.data_len = len(data_source)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.indices = self.find_valid_indices(set(data_source.video_metadata.values()))
        
        weights = [weights[i] for i in self.indices]
        self.weights = torch.tensor(weights, dtype=torch.double)


    def __iter__(self):
        # for i in self.indices:
        #     yield list(range((i + 1) - self.batch_size, i + 1))

        for i in torch.multinomial(self.weights, num_samples=len(self.indices), replacement=True):
            idx = self.indices[i]
            yield list(range((idx + 1) - self.batch_size, idx + 1))


    def __len__(self):
        return len(self.indices)


    def find_valid_indices(self, video_boundaries):
        invalid = set()
        for boundary_index in video_boundaries:
            invalid |= {i for i in range(boundary_index, boundary_index + self.batch_size + self.seq_len)}

        print(invalid)

        valid_indices = [i for i in range(self.data_len) if i not in invalid]
        return [valid_indices[i] for i in torch.randperm(len(valid_indices))]


## torch.multinomial
    # each row in return tensor has num_samples
    # tensor row contains indices drawn from probability distribution of input tensor
    # have to set replacement to True
    # if input is a single vector, output is also a single vector
