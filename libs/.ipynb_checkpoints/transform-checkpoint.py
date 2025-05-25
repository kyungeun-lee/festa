from collections import defaultdict
import torch, random
import numpy as np       
from torch.distributions.uniform import Uniform


class Masking(object):
    def __init__(self, alpha, masking_constant=0., sampling_alpha=False):
        if sampling_alpha:
            self.mask_prob = np.random.beta(alpha, alpha)
        else:
            self.mask_prob = alpha
        if type(masking_constant) == str:
            self.masking_constant = eval(masking_constant)
        else:
            self.masking_constant = masking_constant
    
    def __call__(self, sample):
        img = sample['image']
        mask = np.random.uniform(0, 1, size=img.shape) < self.mask_prob    
        
        img[torch.tensor(mask).to(img.device)] = self.masking_constant
        return {'image': img, 'mask': torch.tensor(mask, device='cuda').requires_grad_(requires_grad=False)}

class Shuffling(object):
    def __init__(self, alpha=0.3, sampling_alpha=False):
        if sampling_alpha:
            self.mask_prob = np.random.beta(alpha, alpha)
        else:
            self.mask_prob = alpha
        self.seed = random.randint(0, 99999)
    
    def __call__(self, sample):
        img = sample['image'].to('cuda')
        mask = np.random.uniform(0, 1, size=img.shape) < self.mask_prob
        mask = torch.tensor(mask, device='cuda')
        
        permuted = torch.empty(size=img.size()).to('cuda')
        for f in range(img.size(1)):
            permuted[:, f] = img[torch.randperm(img.size(0)), f]

        return {'image': img * (1-mask.type(torch.int)) + permuted * mask.type(torch.int), 'mask': mask.requires_grad_(requires_grad=False)}
    
class NoiseMasking(object):
    def __init__(self, alpha=0.2, noise_level=0.1):
        self.mask_prob = alpha
        self.seed = random.randint(0, 99999)
        self.noise_level = noise_level
    
    def __call__(self, sample):
        img = sample['image'].to('cuda')
        mask = np.random.uniform(0, 1, size=img.shape) < self.mask_prob
        mask = torch.tensor(mask, device='cuda')
        
        noise = torch.normal(mean=0.0, std=self.noise_level, size=img.shape, device=img.device)
        permuted = img + noise
        
        for f in range(img.size(1)):
            permuted[:, f] = img[torch.randperm(img.size(0)), f]

        return {'image': img * (1-mask.type(torch.int)) + permuted * mask.type(torch.int), 'mask': mask.requires_grad_(requires_grad=False)}

    
# class BinShuffling(object):
#     def __init__(self, alpha, bin_interval, num_bins):
#         self.mask_prob = np.random.beta(alpha, alpha)
#         self.seed = random.randint(0, 99999)
#         self.bin_interval = bin_interval
#         self.num_bins = num_bins
        
#     def __call__(self, sample):
#         img = sample['image'].to('cuda')
#         mask = np.random.uniform(0, 1, size=img.shape) < self.mask_prob
#         mask = torch.tensor(mask, device='cuda')
        
#         permuted = torch.empty(size=img.size()).to('cuda')
        
#         if self.num_bins == 1:
#             for f in range(img.size(1)):
#                 permuted[:, f] = img[torch.randperm(img.size(0)), f]

#             return {'image': img * (1-mask.type(torch.int)) + permuted * mask.type(torch.int), 'mask': mask}
#         else:
#             ranks = sample['ranks'] // self.bin_interval
#             for f in range(img.size(1)):
#                 upper_resid = torch.where(ranks >= self.num_bins[f])
#                 for (x, y) in zip(upper_resid[0], upper_resid[1]):
#                     ranks[x, y] = (self.num_bins[f] - 1)
#                 for b in range(self.num_bins[f]):
#                     idx = torch.where(ranks[:, f] == b)[0]
#                     perm_idx = idx[torch.randperm(len(idx))]
#                     permuted[idx, f] = img[perm_idx, f]

#             return_img = img * (1 - mask.type(torch.int)) + permuted * mask.type(torch.int)

#             return {'image': img * (1 - mask.type(torch.int)) + permuted * mask.type(torch.int), 'mask': mask}
    
    
class ToTensor(object):
    def __init__(self):
        pass
    def __call__(self, sample):
        if isinstance(sample['image'], np.ndarray):
            return {'image': torch.from_numpy(sample['image']), 'mask': torch.from_numpy(sample['mask'])}
        else:
            return {'image': sample['image'], 'mask': sample['mask']}

class RandQuant(object):
    def __init__(self, num_bins,
                 collapse_to_val="inside_random", spacing="random", p_random_apply_rand_quant=0.3):
        self.num_bins = num_bins
        self.collapse_to_val = collapse_to_val
        self.spacing = spacing
        self.p_random_apply_rand_quant = p_random_apply_rand_quant
        self.transforms_like = False
        
    def get_params(self, x):
        C, _, _ = x.size() # one batch img
        min_val, max_val = x.view(C, -1).min(1)[0], x.view(C, -1).max(1)[0] # min, max over batch size, spatial dimension
        total_region_percentile_number = (torch.ones(C) * (self.num_bins - 1)).int()
        return min_val, max_val, total_region_percentile_number
        
    def __call__(self, sample):
        """
        x: (B, c, H, W) or (C, H, W)
        """
        x = sample['image']
        x = x.view(x.size(0), x.size(1), 1, 1)
        x = x.permute(2, 1, 0, 3)
        
        EPSILON = 1
        if self.p_random_apply_rand_quant != 1:
            x_orig = x
            
        B, c, H, W = x.shape
        C = B * c
        x = x.view(C, H, W)
        
        min_val, max_val, total_region_percentile_number_per_channel = self.get_params(x) # -> (C), (C), (C)
        
        # region percentiles for each channel
        if self.spacing == "random":
            region_percentiles = torch.rand(total_region_percentile_number_per_channel.sum(), device=x.device)
        elif self.spacing == "uniform":
            region_percentiles = torch.tile(torch.arange(1/(total_region_percentile_number_per_channel[0] + 1), 1, step=1/(total_region_percentile_number_per_channel[0]+1), device=x.device), [C])
        region_percentiles_per_channel = region_percentiles.reshape([-1, self.num_bins - 1])
        # ordered region ends
        region_percentiles_pos = (region_percentiles_per_channel * (max_val - min_val).view(C, 1) + min_val.view(C, 1)).view(C, -1, 1, 1)
        ordered_region_right_ends_for_checking = torch.cat([region_percentiles_pos, max_val.view(C, 1, 1, 1)+EPSILON], dim=1).sort(1)[0]
        ordered_region_right_ends = torch.cat([region_percentiles_pos, max_val.view(C, 1, 1, 1)+1e-6], dim=1).sort(1)[0]
        ordered_region_left_ends = torch.cat([min_val.view(C, 1, 1, 1), region_percentiles_pos], dim=1).sort(1)[0]
        # ordered middle points
        ordered_region_mid = (ordered_region_right_ends + ordered_region_left_ends) / 2

        # associate region id
        is_inside_each_region = (x.view(C, 1, H, W) < ordered_region_right_ends_for_checking) * (x.view(C, 1, H, W) >= ordered_region_left_ends) # -> (C, self.region_num, H, W); boolean
        assert (is_inside_each_region.sum(1) == 1).all()# sanity check: each pixel falls into one sub_range
        associated_region_id = torch.argmax(is_inside_each_region.int(), dim=1, keepdim=True)  # -> (C, 1, H, W)
        
        if self.collapse_to_val == 'middle':
            # middle points as the proxy for all values in corresponding regions
            proxy_vals = torch.gather(ordered_region_mid.expand([-1, -1, H, W]), 1, associated_region_id)[:,0]
            x = proxy_vals.type(x.dtype)
        elif self.collapse_to_val == 'inside_random':
            # random points inside each region as the proxy for all values in corresponding regions
            proxy_percentiles_per_region = torch.rand((total_region_percentile_number_per_channel + 1).sum(), device=x.device)
            proxy_percentiles_per_channel = proxy_percentiles_per_region.reshape([-1, self.num_bins])
            ordered_region_rand = ordered_region_left_ends + proxy_percentiles_per_channel.view(C, -1, 1, 1) * (ordered_region_right_ends - ordered_region_left_ends)
            proxy_vals = torch.gather(ordered_region_rand.expand([-1, -1, H, W]), 1, associated_region_id)[:, 0]
            x = proxy_vals.type(x.dtype)

        elif self.collapse_to_val == 'all_zeros':
            proxy_vals = torch.zeros_like(x, device=x.device)
            x = proxy_vals.type(x.dtype)
        else:
            raise NotImplementedError

        if not self.transforms_like:
#             x = x.view(B, c, H, W)
            x = x.view(1, x.size(0), x.size(1), x.size(2))
        
        if self.p_random_apply_rand_quant != 1:
            if not self.transforms_like:
                x = torch.where(torch.rand([B, 1, 1, 1], device=x.device) < self.p_random_apply_rand_quant, x, x_orig)
            else:
                x = torch.where(torch.rand([C, 1, 1], device=x.device) < self.p_random_apply_rand_quant, x, x_orig)
        
        return {'image': x.view(H, c), 'mask': None}
        
class scarfmasking(object):
    def __init__(self, features_low, features_high, masking_ratio=0.6):
        self.marginals = Uniform(torch.Tensor(features_low), torch.Tensor(features_high))
    def __call__(self, x):
        x = x["image"]
        corruption_mask = torch.rand_like(x, device=x.device) > masking_ratio
        x_random = self.marginals.sample(torch.Size((x.size(0),))).to(x.device)
        x_corrupted = torch.where(corruption_mask, x_random, x)
        return {'image': x_corrupted, 'mask': None}
        
class BinShuffling(object):
    def __init__(self, alpha, num_bins, 
                 num_features=[], boundarytype=0): #0: uniform, 1: random, 2: boundary random
        self.alpha = alpha
        self.mask_prob = alpha
        self.seed = random.randint(0, 99999)
        self.num_bins = num_bins
        self.boundarytype = boundarytype
        self.num_features = num_features
        
    def __call__(self, sample):
        img = sample['image']
        mask = np.random.uniform(0, 1, size=img.shape) < self.mask_prob
        mask = torch.tensor(mask, device=img.device)
        
        bin_boundaries = dict()
        for f in self.num_features:
            if self.boundarytype == 1:                
                num_points = 1000 * self.num_bins
                random_integers = torch.randperm(num_points-2, device=img.device)[:self.num_bins]
                random_boundaries = (random_integers+1).sort().values.float() / num_points
                bin_boundaries[f] = torch.cat([torch.tensor([0.0], device=img.device), random_boundaries, torch.tensor([1.0], device=img.device)])
            elif self.boundarytype == 0:                
                bin_boundaries[f] = torch.linspace(0, 1, self.num_bins + 1, device=img.device)
            elif self.boundarytype == 2:                
                bin_boundaries[f] = torch.linspace(0, 1, self.num_bins + 1, device=img.device)
                bin_boundaries[f] += torch.randn(bin_boundaries[f].size(), device=img.device) * 1e-3
        
        permuted = torch.empty(size=img.size()).to(img.device)
        
        if img.size(0) > 1:
            for f in range(img.size(1)):
                boundaries = bin_boundaries.get(f, None)
                if boundaries == None: ## categorical features
                    permuted[:, f] = img[:, f]
                elif len(boundaries) < 2: ## no bins
                    permuted[:, f] = img[torch.randperm(img.size(0)), f]
                else:
                    idx = torch.bucketize(img[:, f], boundaries, right=True) - 1
                    for bi in idx.unique():
                        sample_idx = torch.where(idx == bi)[0]
                        permuted[sample_idx, f] = img[sample_idx[torch.randperm(len(sample_idx))], f]
            return_img = img * (1 - mask.type(torch.int)) + permuted * mask.type(torch.int)
        else:
            return_img = img
        
        return {'image': return_img, 'mask': mask}
    

class BinSampling(object):
    def __init__(self, alpha, num_bins, num_features, boundarytype=0): #0: uniform, 1: random, 2: boundary random
        self.mask_prob = alpha
        self.seed = random.randint(0, 99999)
        self.num_bins = num_bins
        self.num_features = num_features
        self.boundarytype = boundarytype
        
    def __call__(self, sample):
        img = sample['image']
        mask = np.random.uniform(0, 1, size=img.shape) < self.mask_prob
        mask = torch.tensor(mask, device=img.device)
        
        bin_boundaries = dict()
        for f in self.num_features:
            if self.boundarytype == 1:                
                num_points = 1000 * self.num_bins
                random_integers = torch.randperm(num_points-2, device=img.device)[:self.num_bins]
                random_boundaries = (random_integers+1).sort().values.float() / num_points
                bin_boundaries[f] = torch.cat([torch.tensor([0.0], device=img.device), random_boundaries, torch.tensor([1.0], device=img.device)])
            elif self.boundarytype == 0:                
                bin_boundaries[f] = torch.linspace(0, 1, self.num_bins + 1, device=img.device)
            elif self.boundarytype == 2:                
                bin_boundaries[f] = torch.linspace(0, 1, self.num_bins + 1, device=img.device)
                bin_boundaries[f] += torch.randn(bin_boundaries[f].size(), device=img.device) * 1e-3
        
        permuted = torch.empty(size=img.size()).to(img.device)
        
        if img.size(0) > 1:
            for f in range(img.size(1)):
                boundaries = bin_boundaries.get(f, None)
                if boundaries == None: ## categorical features
                    permuted[:, f] = img[:, f]
                elif len(boundaries) < 2: ## no bins
                    minval = torch.min(img[:, f])
                    maxval = torch.max(img[:, f])
                    permuted[:, f] = minval + (maxval - minval) * torch.rand(img.size(0), device=img.device)
                else:
                    idx = torch.bucketize(img[:, f], bin_boundaries[f], right=True) - 1
                    for bi in torch.unique(idx):
                        sample_idx = torch.where(idx == bi)[0]
                        minval = bin_boundaries[f][bi]
                        if bi < (len(bin_boundaries[f])-1):
                            maxval = bin_boundaries[f][bi+1]
                            permuted[sample_idx, f] = minval + (maxval - minval) * torch.rand(len(sample_idx), device=img.device)
                        else:
                            permuted[sample_idx, f] = img[sample_idx, f]
            return_img = img * (1 - mask.type(torch.int)) + permuted * mask.type(torch.int)
        else:
            return_img = img
        
        return {'image': return_img, 'mask': mask}
    
    
class CutMix(object):
    def __init__(self, X_cat, X_num, lam=0.1):
        """
        CutMix transformation class for categorical and continuous inputs.
        
        Args:
        - lam (float): Lambda value for controlling the probability of applying noise.
        - noise_type (str): Type of noise to apply (currently only 'cutmix' supported).
        """
        self.lam = lam
        self.X_cat = X_cat.tolist()
        self.X_num = X_num.tolist()
    
    def __call__(self, x_categ, x_cont):
        """
        Apply the CutMix transformation to the categorical and continuous inputs.
        
        Args:
        - x_categ (torch.Tensor): Categorical input tensor.
        - x_cont (torch.Tensor): Continuous input tensor.
        
        Returns:
        - x_categ_corr (torch.Tensor): Transformed categorical input.
        - x_cont_corr (torch.Tensor): Transformed continuous input.
        """
        device = x_categ.device
        batch_size = x_categ.size(0)

        index = torch.randperm(batch_size).to(device)

        # Create random binary masks for categorical and continuous inputs
        cat_corr = torch.from_numpy(np.random.choice(2, x_categ.shape, p=[self.lam, 1 - self.lam])).to(device)
        con_corr = torch.from_numpy(np.random.choice(2, x_cont.shape, p=[self.lam, 1 - self.lam])).to(device)

        # Permute inputs based on generated indices
        x1, x2 = x_categ[index, :], x_cont[index, :]

        # Clone and detach the original inputs
        x_categ_corr, x_cont_corr = x_categ.clone().detach(), x_cont.clone().detach()

        # Apply CutMix by mixing with permuted inputs based on the masks
        x_categ_corr[cat_corr == 0] = x1[cat_corr == 0]
        x_cont_corr[con_corr == 0] = x2[con_corr == 0]

        x_corr = torch.cat((x_categ_corr, x_cont_corr), dim=1)
        all_indices = self.X_cat + self.X_num
        sort_order = torch.argsort(torch.tensor(all_indices))

        return x_corr[:, sort_order]