import torch
from tqdm import tqdm
import numpy as np
from libs.models.mlp import build_mlp
from libs.models.supervised import CosineAnnealingLR_Warmup

class vime(torch.nn.Module):
    def __init__(self, params, tasktype, device, transform_func, unsup_weight=1, data_id=None, modelname=None, cat_features=[], num_views=3):
        
        super(vime, self).__init__()
        
        self.tasktype = tasktype
        self.cat_features = cat_features
        self.device = device
        self.params = params
        self.data_id = data_id
        self.modelname = modelname
        self.num_views = num_views
        
        self.unsup_weight = unsup_weight
        self.transform_func = transform_func
        
        if self.modelname == "mlp":
            self.model = build_mlp(self.tasktype, params.get("input_dim", None), params.get("output_dim", None), 
                                   params['depth'], params['width'], params['dropout'], params['normalization'], params['activation'],
                                   params['optimizer'], params['learning_rate'], params['weight_decay'])            
            self.model.to(self.device)
    
    def fit(self, X_train, y_train):
        
        batch_size = 128
            
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
        
        n_epochs = (self.params.get('n_steps', 0) // len(train_loader)) + 1
        if self.params["lr_scheduler"]:
            scheduler = CosineAnnealingLR_Warmup(optimizer, base_lr=self.params['learning_rate'], warmup_epochs=10, 
                                                 T_max=n_epochs, iter_per_epoch=len(train_loader), 
                                                 warmup_lr=1e-6, eta_min=0, last_epoch=-1)
        
        self.model = self.model.to(self.device)
        torch.autograd.set_detect_anomaly(True)
        
        loss_history = dict({"sup": [], "unsup": []})
        pbar = tqdm(range(1, n_epochs + 1))
        for epoch in pbar:
            pbar.set_description("EPOCH: %i" %epoch)
            
            for i, (x, y) in enumerate(train_loader):
                self.model.train(); optimizer.zero_grad()
                
                labeled_idx = torch.unique(torch.where(~torch.isnan(y))[0])
                unlabeled_idx = torch.unique(torch.where(torch.isnan(y))[0])
                
                unlabeled_sample = {'image': x[unlabeled_idx], 'mask': None}
                
                ### supervised
                if len(labeled_idx) > 0:
                    labeled_yhat = self.model(x[labeled_idx].to(self.device), cat_features=self.cat_features)
                    if self.tasktype == "binclass":
                        sup_loss = loss_fn(labeled_yhat, y[labeled_idx].to(self.device).view(-1, 1))
                    else:
                        sup_loss = loss_fn(labeled_yhat, y[labeled_idx].to(self.device))
                else:
                    sup_loss = 0.
                
                ### unsupervised
                if len(unlabeled_idx) > 0:
                    unlabeled_views = []
                    for _ in range(self.num_views):
                        unlabeled_views.append(self.transform_func(unlabeled_sample)["image"])
                    unlabeled_views = torch.concatenate(unlabeled_views, axis=0)

                    unlabeled_yhat = self.model(unlabeled_views.to(self.device), cat_features=self.cat_features) 
                    unsup_loss = torch.var(unlabeled_yhat, dim=0).mean() / self.num_views
                else:
                    unsup_loss = 0.
                
                
                loss = sup_loss + self.unsup_weight * unsup_loss
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                
                if self.params["lr_scheduler"]:
                    scheduler.step()
                
                pbar.set_postfix_str(f'data_id: {self.data_id}, Model: {self.modelname}, Tr loss: {loss:.5f} ({sup_loss:.5f}, {unsup_loss:.5f})')
                
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