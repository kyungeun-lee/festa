import torch
from tqdm import tqdm
import numpy as np
from libs.models.supervised import CosineAnnealingLR_Warmup
from libs.models.mlp import build_mlp

class ICT(torch.nn.Module):
    def __init__(self, params, tasktype, device, data_id=None, modelname=None, cat_features=[]):
        
        super(ICT, self).__init__()
        
        self.tasktype = tasktype
        self.cat_features = cat_features
        self.device = device
        self.params = params
        self.data_id = data_id
        self.modelname = modelname
        
        if self.modelname == "mlp":
            self.model = build_mlp(
                self.tasktype, params.get("input_dim", None), params.get("output_dim", None), params['depth'], params['width'], params['dropout'], params['normalization'], params['activation'],
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
        del X_train, y_train
        
        if len(train_dataset) % batch_size == 1:
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True) ## prevent error for batchnorm
        else:
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        
        optimizer.zero_grad(); optimizer.step()
        
        if self.params["lr_scheduler"]:
            scheduler = CosineAnnealingLR_Warmup(optimizer, base_lr=self.params['learning_rate'], warmup_epochs=10, 
                                                 T_max=self.params.get('n_epochs'), iter_per_epoch=len(train_loader), 
                                                 warmup_lr=1e-6, eta_min=0, last_epoch=-1)
        
        loss_history = []
        pbar = tqdm(range(1, self.params.get('n_epochs', 0) + 1))
        for epoch in pbar:
            pbar.set_description("EPOCH: %i" %epoch)
            
            for i, (x, y) in enumerate(train_loader):
                self.model.train(); optimizer.zero_grad()
                
                labeled_idx = torch.unique(torch.where(~torch.isnan(y))[0])
                unlabeled_idx = torch.unique(torch.where(torch.isnan(y))[0])
                
                labeled_sample = x[labeled_idx]
                unlabeled_sample = x[unlabeled_idx]
                
                ### supervised
                if len(labeled_idx) > 0:
                    sup_yhat = self.model(labeled_sample, cat_features=self.cat_features)
                    if self.tasktype == "binclass":
                        sup_loss = loss_fn(sup_yhat, y[labeled_idx].to(self.device).view(-1, 1))
                    else:
                        sup_loss = loss_fn(sup_yhat, y[labeled_idx].to(self.device))
                else:
                    sup_loss = 0.
                
                ### unsupervised
                if (len(unlabeled_idx) > 0) & (len(labeled_idx) > 0):
                    num_unlabeled = len(unlabeled_idx)
                    num_labeled = len(labeled_idx)

                    if num_unlabeled > num_labeled:
                        idx = torch.randperm(num_unlabeled)[:num_labeled]
                        unlabeled_sample = unlabeled_sample[idx]
                        labeled_labels = y[labeled_idx]
                    else:
                        idx = torch.randperm(num_labeled)[:num_unlabeled]
                        labeled_sample = labeled_sample[idx]
                        labeled_labels = y[idx]

                    alpha = torch.rand(len(unlabeled_sample), 1, 1, device=self.device)
                    mixed_inputs = alpha * unlabeled_sample + (1 - alpha) * labeled_sample
                    mixed_labels = alpha * torch.zeros_like(labeled_labels, device=self.device) + (1 - alpha) * labeled_labels.float()

                    mixed_outputs = self.model(mixed_inputs, cat_features=self.cat_features)
                    consistency_loss = torch.nn.functional.mse_loss(mixed_outputs, mixed_labels)
                else:
                    consistency_loss = 0.
                
                loss = sup_loss + consistency_loss
                if len(labeled_idx) > 0: #unsup만 존재하는 경우 고려하지 않음 -- original paper
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                    if self.params["lr_scheduler"]:
                        scheduler.step()
                
                pbar.set_postfix_str(f'data_id: {self.data_id}, Model: {self.modelname}, Tr loss: {loss:.5f} ({sup_loss:.5f}, {consistency_loss:.5f})')
    
    def predict(self, X_test):
        with torch.no_grad():
            if (X_test.shape[0] > 10000):
                logits = []
                iters = X_test.shape[0] // 100 + 1
                for i in range(iters):
                    pred = self.model(X_test[100*i:100*(i+1)], self.cat_features)
                    logits.append(pred)
                    del pred
                logits = torch.concatenate(logits, dim=0)
            else:
                logits = self.model(X_test, self.cat_features)
            if self.tasktype == "binclass":
                return torch.sigmoid(logits).round().cpu().numpy()
            elif self.tasktype == "regression":
                return logits.cpu().numpy()
            else:
                return torch.argmax(logits, dim=1).cpu().numpy()
    
    def predict_proba(self, X_test, logit=False):
        with torch.no_grad():
            if (X_test.shape[0] > 10000) or (X_test.shape[1] > 240):
                logits = []
                iters = X_test.shape[0] // 100 + 1
                for i in range(iters):
                    pred = self.model(X_test[100*i:100*(i+1)], self.cat_features)
                    logits.append(pred)
                    del pred
                logits = torch.concatenate(logits, dim=0)
            else:
                logits = self.model(X_test, self.cat_features)
                
            if logit:
                return logits.cpu().numpy()
            else:
                return torch.nn.functional.softmax(logits).cpu().numpy()