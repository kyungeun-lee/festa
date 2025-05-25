import torch
import numpy as np
from tabpfn import TabPFNClassifier
# For tabpfn, you should install tabpfn library as 1.**
# For tabpfnv2, you should install tabpfn library as 2.**

class tabpfn(torch.nn.Module):
    def __init__(self, params, tasktype, input_dim=0, output_dim=0, device='cuda', data_id=None, modelname="tabpfn"):
        
        super(tabpfn, self).__init__()
        self.tasktype = tasktype
        if modelname == "tabpfn":
            self.model = TabPFNClassifier(device=device, N_ensemble_configurations=32)
        elif modelname == "tabpfnv2":
            self.model = TabPFNClassifier(device=device, ignore_pretraining_limits=True)
    
    def fit(self, X_train, y_train):
        labeled_flag = torch.unique(torch.where(~torch.isnan(y_train))[0])
        label_X_train = X_train[labeled_flag]
        label_y_train = y_train[labeled_flag]
        
        if self.tasktype == "multiclass":
            label_y_train = torch.argmax(label_y_train, dim=1)
        try:
            self.model.fit(label_X_train.cpu().numpy(), label_y_train.cpu().numpy())
            self.exception = False
        except ValueError:
            self.exception = True
            
    def predict(self, X_test):
        if self.exception:
            return None
        else:
            return self.model.predict(X_test.cpu().numpy())
        
    def predict_proba(self, X_test, logit=False):
        if self.exception:
            return None
        else:
            return self.model.predict_proba(X_test.cpu().numpy())
