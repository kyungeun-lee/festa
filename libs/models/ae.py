import torch
from tqdm import tqdm
import numpy as np
from libs.models.mlp import build_mlp
from libs.models.supervised import CosineAnnealingLR_Warmup
from itertools import chain

class ae(torch.nn.Module):
    def __init__(self, params, tasktype, device, transform_func, unsup_loss_func="mse_recon", unsup_weight=1, data_id=None, modelname=None, cat_features=[]):
        
        super(ae, self).__init__()
        
        self.tasktype = tasktype
        self.cat_features = cat_features
        self.device = device
        self.params = params
        self.data_id = data_id
        self.modelname = modelname
        
        self.unsup_loss_func = unsup_loss_func
        self.unsup_weight = unsup_weight
        self.transform_func = transform_func
        
        if self.modelname == "mlp":
            self.encoder = build_mlp(self.tasktype, params.get("input_dim", None), params.get("width", None), 
                                     params['depth'], params['width'], params['dropout'], params['normalization'], params['activation'],
                                     params['optimizer'], params['learning_rate'], params['weight_decay'])
            self.decoder = build_mlp(self.tasktype, params.get("width", None), params.get("input_dim", None), 
                                     params['depth'], params['width'], params['dropout'], params['normalization'], params['activation'],
                                     params['optimizer'], params['learning_rate'], params['weight_decay'])
            self.predictor = build_mlp(self.tasktype, params.get("width", None), params.get("output_dim", None), 
                                       1, params['width'], params['dropout'], params['normalization'], params['activation'],
                                       params['optimizer'], params['learning_rate'], params['weight_decay'])
            self.encoder.to(self.device)
            self.decoder.to(self.device)
            self.predictor.to(self.device)
    
    def fit(self, X_train, y_train):
        
        batch_size = 100
            
        optimizer = torch.optim.AdamW(
            chain(self.encoder.parameters(), self.decoder.parameters(), self.predictor.parameters()), 
            lr=self.params['learning_rate'], 
            weight_decay=self.params['weight_decay']
        )
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
            
        torch.autograd.set_detect_anomaly(True)
        
        loss_history = dict({"sup": [], "unsup": []})
        pbar = tqdm(range(1, self.params.get('n_epochs', 0) + 1))
        for epoch in pbar:
            pbar.set_description("EPOCH: %i" %epoch)
            
            for i, (x, y) in enumerate(train_loader):
                self.encoder.train(); self.decoder.train(); self.predictor.train(); optimizer.zero_grad()
                
                labeled_idx = torch.unique(torch.where(~torch.isnan(y))[0])
                unlabeled_idx = torch.unique(torch.where(torch.isnan(y))[0])
                
                unlabeled_sample = {'image': x[unlabeled_idx], 'mask': None}
                
                ### supervised
                if len(labeled_idx) > 0:
                    labeled_z = self.encoder(x[labeled_idx].to(self.device), cat_features=self.cat_features)
                    labeled_yhat = self.predictor(labeled_z, cat_features=None)
                    if self.tasktype == "binclass":
                        sup_loss = loss_fn(labeled_yhat, y[labeled_idx].to(self.device).view(-1, 1))
                    else:
                        sup_loss = loss_fn(labeled_yhat, y[labeled_idx].to(self.device))
                else:
                    sup_loss = 0.
                
                ### unsupervised
                if len(unlabeled_idx) > 0:
                    unlabeled_x = self.transform_func(unlabeled_sample)
                    unlabeled_z = self.encoder(unlabeled_x["image"], cat_features=self.cat_features)
                    unlabeled_xhat = self.decoder(unlabeled_z)
                    
                    if self.unsup_loss_func == "mse_recon":
                        unsup_loss = torch.nn.functional.mse_loss(unlabeled_xhat, unlabeled_sample["image"].to(self.device))
                    else:
                        unsup_loss = torch.nn.functional.binary_cross_entropy_with_logits(unlabeled_xhat, unlabeled_x['mask'].to(torch.float32))
                else:
                    unsup_loss = 0.
                
                loss = sup_loss + self.unsup_weight * unsup_loss
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                
                if self.params["lr_scheduler"]:
                    scheduler.step()
                
                pbar.set_postfix_str(f'data_id: {self.data_id}, Model: {self.modelname}, Tr loss: {loss:.5f} ({sup_loss:.5f}, {unsup_loss:.5f})')
    
    def predict(self, X_test):
        self.encoder.eval(); self.decoder.eval(); self.predictor.eval()
        with torch.no_grad():
            if (X_test.shape[0] > 10000):
                logits = []
                iters = X_test.shape[0] // 100 + 1
                for i in range(iters):
                    pred = self.predictor(self.encoder(X_test[100*i:100*(i+1)], self.cat_features))
                    logits.append(pred)
                    del pred
                logits = torch.concatenate(logits, dim=0)
            else:
                logits = self.predictor(self.encoder(X_test, self.cat_features))
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
                    pred = self.predictor(self.encoder(X_test[100*i:100*(i+1)], self.cat_features))
                    logits.append(pred)
                    del pred
                logits = torch.concatenate(logits, dim=0)
            else:
                logits = self.predictor(self.encoder(X_test, self.cat_features))
                
            if logit:
                return logits.cpu().numpy()
            else:
                return torch.nn.functional.softmax(logits).cpu().numpy()