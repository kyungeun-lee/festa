import torch
from tqdm import tqdm
import numpy as np

class supmodel(torch.nn.Module):
    def __init__(self, params, tasktype, device, data_id=None, modelname=None, cat_features=[]):
        
        super(supmodel, self).__init__()
        
        self.tasktype = tasktype
        self.cat_features = cat_features
        self.device = device
        self.params = params
        self.data_id = data_id
        self.modelname = modelname
    
    def fit(self, X_train, y_train):
        
        print(self.device)
        if y_train.ndim == 2:
            X_train = X_train[~torch.isnan(y_train[:, 0]), :]
            y_train = y_train[~torch.isnan(y_train[:, 0])]
        else:
            X_train = X_train[~torch.isnan(y_train), :]
            y_train = y_train[~torch.isnan(y_train)]
            
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
        
        self.model.to(self.device)
        
        loss_history = []
        pbar = tqdm(range(1, self.params.get('n_epochs', 0) + 1))
        for epoch in pbar:
            pbar.set_description("EPOCH: %i" %epoch)
            
            for i, (x, y) in enumerate(train_loader):
                self.model.train(); optimizer.zero_grad()
                
                out = self.model(x.to(self.device), self.cat_features)
                if out.size() != y.size():
                    out = out.view(y.size())
                loss = loss_fn(out, y.to(self.device))
                loss_history.append(loss.item())
                
                loss.backward()
                optimizer.step() 
                if self.params["lr_scheduler"]:
                    scheduler.step()
                
                pbar.set_postfix_str(f'data_id: {self.data_id}, Model: {self.modelname}, Tr loss: {loss:.5f}')
                
        self.model.eval()
    
    def predict(self, X_test, cat_features=[]):
        with torch.no_grad():
#             import IPython; IPython.embed()
            if (X_test.shape[0] > 10000):
                logits = []
                iters = X_test.shape[0] // 100 + 1
                for i in range(iters):
                    pred = self.model(X_test[100*i:100*(i+1)], cat_features)
                    logits.append(pred)
                    del pred
                logits = torch.concatenate(logits, dim=0)
            else:
                logits = self.model(X_test, cat_features)
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
                    pred = self.model(X_test[100*i:100*(i+1)], cat_features)
                    logits.append(pred)
                    del pred
                logits = torch.concatenate(logits, dim=0)
            else:
                logits = self.model(X_test, cat_features)
                
            if logit:
                return logits
            elif self.tasktype == "binclass":
                return torch.sigmoid(logits).cpu().numpy()
            elif self.tasktype == "multiclass":
                return torch.nn.functional.softmax(logits).cpu().numpy()
            else:
                return logits.cpu().numpy()
    
    
    
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
    
import inspect
def filter_params(func, params):
    signature = inspect.signature(func)
    valid_params = signature.parameters.keys()

    filtered_params = {key: value for key, value in params.items() if key in valid_params}
    return filtered_params