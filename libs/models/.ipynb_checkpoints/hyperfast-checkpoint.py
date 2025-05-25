import torch
import numpy as np
from hyperfast import HyperFastClassifier

class hyperfast(torch.nn.Module):
    def __init__(self, params, tasktype, input_dim=0, output_dim=0, device='cuda', data_id=None, modelname="hyperfast"):
        
        super(hyperfast, self).__init__()
        self.tasktype = tasktype
        self.model = HyperFastClassifier(device=device)
    
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
