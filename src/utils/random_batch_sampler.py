import torch
from torch.utils.data.sampler import Sampler

class RandomBatchSampler(Sampler):
    ############################################################
    ###### Samples batches randomly, without replacement  ######
    ###### Items in batch are not shuffled                ######
    ###### data_source: Dataset to sample from            ######
    ############################################################

    def __init__(self, data_source, batch_size):
        self.data_len = len(data_source)
        self.batch_size = batch_size
        self.indices = self.find_valid_indices(set(data_source.video_metadata.values()))


    def __iter__(self):
        for i in self.indices:
            yield list(range((i + 1) - self.batch_size, i + 1))


    def __len__(self):
        return len(self.indices)


    ## remove indices within batch_size of first frame of video
    ## each batch represents a short clip of the video -- clip cannot end with first frame
    def find_valid_indices(self, video_boundaries):
        invalid = set()
        for boundary_index in video_boundaries:
            invalid |= {i for i in range(boundary_index, boundary_index + (self.batch_size - 1))}

        valid_indices = [i for i in range(self.data_len) if i not in invalid]
        return [valid_indices[i] for i in torch.randperm(len(valid_indices))]