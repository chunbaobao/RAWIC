import numpy as np


class ChannelPartiteSampler:  # TODO: maybe change to different partite for every batch
    def __init__(self, partite_list: list, in_ch: int, min_ch: int = 1, sample_idx=False):

        self.partite_list = partite_list
        self.in_ch = in_ch
        self.min_ch = min_ch
        self.sample_idx = sample_idx

    def sample_partite(self):
        return np.random.choice(self.partite_list)

    def sample_ch(self, partite):

        res = self.in_ch - partite * self.min_ch
        assert res >= 0, "in_ch too small"
        if res == 0:
            return np.array([self.min_ch] * partite)
        else:
            cuts = np.random.choice(np.arange(1, res + partite), size=partite - 1, replace=False)
            cuts.sort()
            cuts = np.concatenate(([0], cuts, [res + partite]))
            y = np.diff(cuts) - 1
            x = y + self.min_ch
            assert sum(x) == self.in_ch and len(x) == partite and min(x) >= self.min_ch

            if self.sample_idx:

                return np.pad(x.cumsum(), (1, 0)).astype(int)  # return the index of channels

            return x.astype(int).tolist()

    def sample(self):
        partite = self.sample_partite()
        sample_ch = self.sample_ch(partite)
        return sample_ch


class ConstantSampler:
    def __init__(self, groups: list):
        self.partite = len(groups)
        self.groups = groups

    def sample(self):
        return self.groups
