import torch
from torch.utils.data.sampler import Sampler

class NonRandomBatchSampler(Sampler):
    ############################################################
    ###### Samples batches sequentially, no replacement   ######
    ###### Items in batch are not shuffled                ######
    ###### data_source: Dataset to sample from            ######
    ############################################################

    def __init__(self, data_source, batch_size):
        self.data_len = len(data_source)
        self.batch_size = batch_size
        self.indices = self.find_valid_indices(set(data_source.video_metadata.values()))


    def __iter__(self):
        batch = []
        for i in self.indices:
            batch.append(i)
            if len(batch) == self.batch_size:
                yield batch
                batch = []


    def __len__(self):
        return len(self.indices)


    ## remove indices within batch_size of last frame of video
    ## this sampler counts forwards, removing sections of video < batch_size
    ## is equivalent to drop_last = True for each video
    def find_valid_indices(self, video_boundaries):
        invalid = set()
        video_boundaries = list(video_boundaries)
        video_boundaries.sort()
        for i in range(1, len(video_boundaries)):
            invalid_range = video_boundaries[i] % self.batch_size
            invalid |= {i for i in range(video_boundaries[i] - invalid_range, video_boundaries[i])}

        # computes drop_last for last video
        invalid |= {i for i in range(self.data_len - invalid_range, self.data_len)}

        valid_indices = [i for i in range(self.data_len) if i not in invalid]
        return valid_indices
