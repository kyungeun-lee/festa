import torch
from tqdm import tqdm
import numpy as np
from libs.models.mlp import build_mlp

class meanteacher(torch.nn.Module):
    def __init__(self, params, tasktype, device, alpha=0.99, data_id=None, modelname=None, cat_features=[]):
        
        super(meanteacher, self).__init__()
        
        self.tasktype = tasktype
        self.cat_features = cat_features
        self.device = device
        self.params = params
        self.data_id = data_id
        self.modelname = modelname
        
        self.alpha = alpha
        
        if self.modelname == "mlp":
            self.student = build_mlp(
                self.tasktype, params.get("input_dim", None), params.get("output_dim", None), params['depth'], params['width'], params['dropout'], params['normalization'], params['activation'],
                params['optimizer'], params['learning_rate'], params['weight_decay'])            
            self.teacher = build_mlp(
                self.tasktype, params.get("input_dim", None), params.get("output_dim", None), params['depth'], params['width'], params['dropout'], params['normalization'], params['activation'],
                params['optimizer'], params['learning_rate'], params['weight_decay'])          
            self.teacher.load_state_dict(self.student.state_dict())
            self.student.to(self.device)
            self.teacher.to(self.device)
    
    def update_ema_variables(self, global_step):
        alpha = min(1 - 1 / (global_step + 1), self.alpha)
        for ema_param, param in zip(self.teacher.parameters(), self.student.parameters()):
            ema_param.data.mul_(self.alpha).add_(param.data, alpha=1 - self.alpha)
    
    def fit(self, X_train, y_train):
        
        batch_size = 100
            
        optimizer = self.student.make_optimizer()
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
                self.student.train(); self.teacher.train(); optimizer.zero_grad()
                
                labeled_idx = torch.unique(torch.where(~torch.isnan(y))[0])
                unlabeled_idx = torch.unique(torch.where(torch.isnan(y))[0])
                
                labeled_sample = x[labeled_idx]
                unlabeled_sample = x[unlabeled_idx]
                
                ### supervised
                if len(labeled_idx) > 0:
                    labeled_student = self.student(labeled_sample, cat_features=self.cat_features)
                    if self.tasktype == "binclass":
                        sup_loss = loss_fn(labeled_student, y[labeled_idx].to(self.device).view(-1, 1))
                    else:
                        sup_loss = loss_fn(labeled_student, y[labeled_idx].to(self.device))
                else:
                    sup_loss = 0.
                
                ### unsupervised
                if len(unlabeled_idx) > 0:
                    unlabeled_student = self.student(unlabeled_sample, cat_features=self.cat_features)
                    with torch.no_grad():
                        unlabeled_teacher = self.teacher(unlabeled_sample, cat_features=self.cat_features)
                    unsup_loss = torch.nn.functional.mse_loss(unlabeled_student, unlabeled_teacher)
                else:
                    unsup_loss = 0.
                
                loss = sup_loss + unsup_loss
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                
                if self.params["lr_scheduler"]:
                    scheduler.step()
                
                global_step = epoch * len(train_loader) + i
                self.update_ema_variables(global_step)
                
                pbar.set_postfix_str(f'data_id: {self.data_id}, Model: {self.modelname}, Tr loss: {loss:.5f} ({sup_loss:.5f}, {unsup_loss:.5f})')
                
        self.student.eval()
    
    def predict(self, X_test, cat_features=[]):
        with torch.no_grad():
            if (X_test.shape[0] > 10000):
                logits = []
                iters = X_test.shape[0] // 100 + 1
                for i in range(iters):
                    pred = self.student(X_test[100*i:100*(i+1)], cat_features)
                    logits.append(pred)
                    del pred
                logits = torch.concatenate(logits, dim=0)
            else:
                logits = self.student(X_test, cat_features)
            if self.tasktype == "binclass":
                return torch.sigmoid(logits).round().cpu().numpy()
            elif self.tasktype == "regression":
                return logits.cpu().numpy()
            else:
                return torch.argmax(logits, dim=1).cpu().numpy()
    
    def predict_proba(self, X_test, cat_features=[], logit=False):
        with torch.no_grad():
            if (X_test.shape[0] > 10000) or (X_test.shape[1] > 240):
                logits = []
                iters = X_test.shape[0] // 100 + 1
                for i in range(iters):
                    pred = self.student(X_test[100*i:100*(i+1)], cat_features)
                    logits.append(pred)
                    del pred
                logits = torch.concatenate(logits, dim=0)
            else:
                logits = self.student(X_test, cat_features)
                
            if logit:
                return logits.cpu().numpy()
            else:
                return torch.nn.functional.softmax(logits).cpu().numpy()

    

class CosineAnnealingLR_Warmup(object):
    def __init__(self, optimizer, warmup_epochs, T_max, iter_per_epoch, base_lr, warmup_lr, eta_min, last_epoch=-1):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.T_max = T_max
        self.iter_per_epoch = iter_per_epoch
        self.base_lr = base_lr
        self.warmup_lr = warmup_lr
        self.eta_min = eta_min
        self.last_epoch = last_epoch

        self.warmup_iter = self.iter_per_epoch * self.warmup_epochs
        self.cosine_iter = self.iter_per_epoch * (self.T_max - self.warmup_epochs)
        self.current_iter = (self.last_epoch + 1) * self.iter_per_epoch

        self.step()

    def get_current_lr(self):
        if self.current_iter < self.warmup_iter:
            current_lr = (self.base_lr - self.warmup_lr) / self.warmup_iter * self.current_iter + self.warmup_lr
        else:
            current_lr = self.eta_min + (self.base_lr - self.eta_min) * (1 + np.cos(np.pi * (self.current_iter-self.warmup_iter) / self.cosine_iter)) / 2
        return current_lr

    def step(self):
        current_lr = self.get_current_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr
        self.current_iter += 1


def CosineAnnealingParam(warmup_epochs, T_max, iter_per_epoch, current_iter, base_value, 
                         warmup_value=1e-8, eta_min=0):
    warmup_iter = iter_per_epoch * warmup_epochs
    cosine_iter = iter_per_epoch * (T_max - warmup_epochs)
    
    if current_iter < warmup_iter:
        return (base_value - warmup_value) / warmup_iter * current_iter + warmup_value
    else:
        return eta_min + (base_value - eta_min) * (1 + np.cos(np.pi * (current_iter - warmup_iter) / cosine_iter)) / 2
