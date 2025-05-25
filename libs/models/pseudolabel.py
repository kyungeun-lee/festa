import torch, time
from tqdm import tqdm
import numpy as np
from libs.models.mlp import build_mlp
from libs.transform import *
from libs.utils import CosineAnnealingLR_Warmup

class pseudolabel(torch.nn.Module):
    def __init__(self, params, tasktype, device, weak_aug, strong_aug, train_transform=False, loss_u="regression",
                 unsup_weight=1, data_id=None, modelname=None, cat_features=[], num_views=3, T_sharp=0.5, weak_confidence=0,
                 increase_rate=0.05, aug_update_epoch=5, aug_update_threshold=(0.5, 0.9), X_num=None, X_cat=None):
        
        super(pseudolabel, self).__init__()
        
        self.tasktype = tasktype
        self.cat_features = cat_features
        self.device = device
        self.params = params
        self.data_id = data_id
        self.modelname = modelname
        self.num_views = num_views
        
        self.unsup_weight = unsup_weight
        self.weak_aug = weak_aug
        self.strong_aug = strong_aug
        self.train_transform = train_transform
        self.T_sharp = T_sharp
        self.weak_confidence = weak_confidence
        self.loss_u = loss_u
        self.increase_rate = increase_rate
        self.aug_update_epoch = aug_update_epoch
        self.aug_update_threshold = aug_update_threshold
        
        if self.modelname == "mlp":
            self.model = build_mlp(self.tasktype, params.get("input_dim", None), params.get("output_dim", None), 
                                   params['depth'], params['width'], params['dropout'], params['normalization'], params['activation'],
                                   params['optimizer'], params['learning_rate'], params['weight_decay'])            
            self.model.to(self.device)
    
    def fit(self, X_train, y_train):
        
        batch_size = 100
            
        optimizer = self.model.make_optimizer()
        if self.tasktype == "regression":
            loss_fn = torch.nn.functional.mse_loss
        elif self.tasktype == "binclass":
            loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
        else:
            loss_fn = torch.nn.functional.cross_entropy
            
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        if self.tasktype == "multiclass":
            ydim = y_train.size(1)
        del X_train, y_train
        
        if len(train_dataset) % batch_size == 1:
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True) ## prevent error for batchnorm
        else:
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        
        optimizer.zero_grad(); optimizer.step()
        
        if self.params["lr_scheduler"]:
            scheduler = CosineAnnealingLR_Warmup(optimizer, base_lr=self.params['learning_rate'], warmup_epochs=self.params.get('n_epochs')//10, 
                                                 T_max=self.params.get('n_epochs'), iter_per_epoch=len(train_loader), 
                                                 warmup_lr=1e-6, eta_min=0, last_epoch=-1)
        
        self.model = self.model.to(self.device)
        torch.autograd.set_detect_anomaly(True)
        
        loss_history = dict({"sup": [], "unsup": []})
        pbar = tqdm(range(1, self.params.get('n_epochs', 0) + 1))
        for epoch in pbar:
            pbar.set_description("EPOCH: %i" %epoch)
            
            for i, (x, y) in enumerate(train_loader):
                self.model.train(); optimizer.zero_grad()
                
                labeled_idx = torch.unique(torch.where(~torch.isnan(y))[0])
                unlabeled_idx = torch.unique(torch.where(torch.isnan(y))[0])
                
                labeled_sample = {'image': x[labeled_idx], 'mask': None}
                unlabeled_sample = {'image': x[unlabeled_idx], 'mask': None}
                
                ### supervised
                if len(labeled_idx) > 0:
                    weak_x = self.weak_aug(labeled_sample)
                    logit_x = self.model(weak_x["image"].to(self.device), cat_features=self.cat_features)
                else:
                    logit_x = torch.empty(y[labeled_idx].size(), device=self.device)
                
                ### unsupervised
#                 st = time.time()
                if len(unlabeled_idx) > 0:
                    weak_u = []
                    with torch.no_grad():
                        for _ in range(self.num_views):
                            weak_u.append(self.weak_aug(unlabeled_sample)["image"])
                        weak_u = torch.concatenate(weak_u, axis=0)
                        logit_w = self.model(weak_u.to(self.device), cat_features=self.cat_features)
                        
                    strong_u = []
                    for _ in range(self.num_views):
                        if isinstance(self.strong_aug, CutMix): 
                            strong_u.append(self.strong_aug(
                                unlabeled_sample["image"][:, self.params["cat_features"]], 
                                unlabeled_sample["image"][:, self.params["num_features"]]))
                        else:
                            strong_u.append(self.strong_aug(unlabeled_sample)["image"])
                    strong_u = torch.concatenate(strong_u, axis=0)
                    logit_s = self.model(strong_u.to(self.device), cat_features=self.cat_features)
                    
                else:
                    logit_x = torch.empty(y[unlabeled_idx].size(), device=self.device)
                    logit_s = torch.empty(y[unlabeled_idx].size(), device=self.device)
#                 print(time.time() - st)
                
                ## sharpening
                if self.tasktype == 'regression':
                    targets_u = logit_w.detach() ## no sharpening
                    mask = None
                elif self.tasktype == 'binclass':
                    pseudo_label = torch.sigmoid(logit_w.detach()/self.T_sharp)
                    pseudo_label = torch.cat([1-pseudo_label, pseudo_label], dim=-1)
                    max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                    mask = max_probs.ge(self.weak_confidence).float()
                else:
                    pseudo_label = torch.softmax(logit_w.detach()/self.T_sharp, dim=1)
                    max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                    targets_u = torch.nn.functional.one_hot(targets_u, num_classes=ydim)
                    mask = max_probs.ge(self.weak_confidence).float()
                    mask = mask.view(-1, 1).tile((1, ydim))
                
                Lx, Lu, w = SemiLoss()(
                    logit_x, y[labeled_idx],
                    logit_s, targets_u,
                    epoch+i/len(train_loader), self.tasktype, self.loss_u, self.unsup_weight, mask, T_max=self.params.get('n_epochs', 0))
                
                loss = Lx + w * Lu
                try:
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                except RuntimeError:
                    import IPython; IPython.embed()
                
                if self.params["lr_scheduler"]:
                    scheduler.step()                    
                    
                pbar.set_postfix_str(f'data_id: {self.data_id}, Model: {self.modelname}, Tr loss: {loss:.5f} ({Lx:.5f}, {Lu:.5f}, {w:.5f})')
                
            if (self.train_transform) & isinstance(self.strong_aug, BinShuffling) & (epoch % self.aug_update_epoch == 0):
                
                weak_rep = self.model(self.weak_aug({"image": x, "mask": None})["image"])
                current_boundaries = self.strong_aug.bin_boundaries
                
                for feature_idx in current_boundaries:
                    for _ in range(10):
                        try_boundaries = perform_action_data(current_boundaries, self.X_num, increase_rate=self.increase_rate)
                        try_aug = BinShuffling(self.strong_aug.alpha, try_boundaries)
                        try_rep = self.model(try_aug({"image": x, "mask": None})["image"])
                        
                        similarity = calculate_similarity("cosine", weak_rep, try_rep).mean()
                        if (similarity >= self.aug_update_threshold[0]) & (similarity < self.aug_update_threshold[1]):
                            self.strong_aug.bin_boundaries = try_boundaries
                
        self.model.eval()
    
    def predict(self, X_test):
        with torch.no_grad():            
            logits = self.model(X_test, self.cat_features)
            if self.tasktype == "binclass":
                return torch.sigmoid(logits).round().cpu().numpy()
            elif self.tasktype == "regression":
                return logits.cpu().numpy()
            else:
                return torch.argmax(logits, dim=1).cpu().numpy()
    
    def predict_proba(self, X_test, logit=False):
        with torch.no_grad():
            logits = self.model(X_test, self.cat_features)
                
            if logit:
                return logits.cpu().numpy()
            else:
                return torch.nn.functional.softmax(logits).cpu().numpy()
    
class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, 
                 tasktype_x, tasktype_u, lambda_u, mask, T_max=100):
        
        #sup loss first
        if len(outputs_x) == 0:
            Lx = 0.
        elif tasktype_x == "multiclass":
            Lx = -torch.mean(torch.sum(torch.nn.functional.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        elif tasktype_x == "binclass":
            Lx = torch.nn.functional.binary_cross_entropy(torch.nn.functional.sigmoid(outputs_x), targets_x.view(-1, 1))
        else:
            assert tasktype_x == "regression"
            Lx = torch.mean((outputs_x - targets_x)**2)
            
        #unsup loss
        if len(outputs_u) == 0:
            Lu = 0.
        elif tasktype_u == "multiclass":
            assert mask != None
            Lu = (torch.nn.functional.cross_entropy(logits_u, targets_u, reduction='none') * mask).mean()
        elif tasktype_u == "binclass":
            assert mask != None
            Lu = torch.nn.functional.binary_cross_entropy(torch.nn.functional.sigmoid(outputs_u), targets_u)
        else:
            assert tasktype_u == "regression"
            if tasktype_x == "multiclass":
                probs_u = torch.softmax(outputs_u, dim=1)
                Lu = torch.mean(((probs_u - targets_u)**2) * mask)
            elif tasktype_x == "binclass":
                probs_u = torch.nn.functional.sigmoid(outputs_u)
                Lu = torch.mean(((probs_u - targets_u)**2) * mask)
            else:
                Lu = torch.mean((outputs_u - targets_u)**2)
        
        return Lx, Lu, lambda_u * linear_rampup(epoch, T_max)

def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

import random
def perform_action(bin_boundaries, action, increase_rate=0.05):
    if action == "merge":
        if len(bin_boundaries) > 2:
            idx = random.randint(1, len(bin_boundaries) - 2)
            bin_boundaries = torch.cat((bin_boundaries[:idx], bin_boundaries[idx+1:]))
    elif action == "increase":
        if len(bin_boundaries) > 2:
            idx = random.randint(1, len(bin_boundaries) - 2)
            delta = increase_rate * (bin_boundaries[idx + 1] - bin_boundaries[idx])
            bin_boundaries[idx] -= delta
            bin_boundaries[idx + 1] += delta
    return bin_boundaries

def perform_action_cat(categories, action):
    if action == "merge":
        if len(categories) > 2:
            idx = random.randint(2, len(categories))
            bin_boundaries = np.delete(bin_boundaries, idx)
    return bin_boundaries

def perform_action_data(bin_boundaries, X_num, increase_rate=0.05):
    actions = ["merge", "increase", "stay"]
    for k, v in bin_boundaries.items():
        if k in X_num:
            action = random.choice(actions)
            bin_boundaries[k] = perform_action(v, action, increase_rate=increase_rate)
    return bin_boundaries
        
def calculate_similarity(metric, data1, data2):
    if metric == "cosine":
        data1_norm = torch.nn.functional.normalize(data1, p=2, dim=1)
        data2_norm = torch.nn.functional.normalize(data2, p=2, dim=1)
        return torch.sum(data1_norm * data2_norm, dim=1)