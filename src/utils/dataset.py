import os
import random
import numpy as np
import torch

class CommutingODPairDataset(torch.utils.data.Dataset):
    """
        Dataset that treats every intra-area pair (i, j) as a sample.
        x: (F,)  y: scalar
        Each OD pair becomes one data point.
    """
    def __init__(self, root, areas, toy_flag=False):
        self.root = root
        self.areas = areas.copy()
        self.toy_flag = toy_flag

        self.samples = []
        for area in self.areas:
            demos, pois, dis, od = self._load_area_arrays(area)
            x = self._make_feature_tensor(demos, pois, dis)  # (N,N,F)
            y = od                                           # (N,N)
            N = x.shape[0]

            for i in range(N):
                for j in range(N):
                    y_ij = y[i, j]
                    self.samples.append({
                        "x": x[i, j],                   # shape (F,)
                        "y": torch.tensor(y_ij).float(),# scalar
                        "area": area,
                        "i": i,
                        "j": j
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "x": s["x"],     # Tensor(F,)
            "y": s["y"],     # scalar
            "area": s["area"],
            "i": s["i"],
            "j": s["j"]
        }

    def _load_area_arrays(self, area):
        prefix = os.path.join(self.root, area)
        demos = np.load(f"{prefix}/demos.npy")     # (N, D_d)
        pois  = np.load(f"{prefix}/pois.npy")      # (N, D_p)
        dis   = np.load(f"{prefix}/dis.npy")       # (N, N)
        od    = np.load(f"{prefix}/od.npy")        # (N, N)
        return demos, pois, dis, od

    def _make_feature_tensor(self, demos, pois, dis):
        """
            Build the feature tensor.
            :param demos: (N, D_d) demographic features
            :param pois: (N, D_p)  POI features
            :param dis: (N, N)     pairwise distance matrix
            :return: (N, N, F) feature tensor ready for modeling
        """
        if self.toy_flag:
            feat = demos[:, [0]] # (N,1)
        else:
            feat = np.concatenate([demos, pois], axis=1) # (N,F) 
            
        N = feat.shape[0]
        feat_o = feat[:, None, :]                      # (N,1,F)
        feat_d = feat[None, :, :]                      # (1,N,F)
        dis    = dis[..., None]                        # (N,N,1)
        x = np.concatenate([np.repeat(feat_o, N, axis=1),
                            np.repeat(feat_d, N, axis=0),
                            dis], axis=2)              # (N,N,2F+1)
        return torch.from_numpy(x).float()             # (N,N,F)
