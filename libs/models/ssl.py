import torch, os
from tqdm import tqdm
from libs.data import Binning
import numpy as np
from libs.models.mlp import build_mlp
from itertools import chain
from libs.utils import CosineAnnealingLR_Warmup
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=3, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = np.Inf
        self.early_stop = False
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
        if self.counter >= self.patience:
            self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'Validation loss decreased ({self.best_loss:.6f} —> {val_loss:.6f}). Saving model…')
        torch.save(model.state_dict(), self.path)

class sslmodel(torch.nn.Module):
    def __init__(self, params, tasktype, device, transform_func, ssl_loss="mse_binning", data_id=None, modelname=None, cat_features=[]):
        
        super(sslmodel, self).__init__()
        
        self.tasktype = tasktype
        self.cat_features = cat_features
        self.device = device
        self.params = params
        self.data_id = data_id
        self.modelname = modelname
        
        self.ssl_loss = ssl_loss
        self.transform_func = transform_func
        
        if self.modelname == "mlp":
            self.encoder = build_mlp(self.tasktype, params.get("input_dim", None), params.get("width", None), 
                                     params['depth'], params['width'], params['dropout'], params['normalization'], params['activation'],
                                     params['optimizer'], params['ssl_learning_rate'], params['ssl_weight_decay'])
            self.encoder.to(self.device)
            if self.ssl_loss == "vime":
                self.decoder1 = build_mlp(self.tasktype, params.get("width", None), params.get("input_dim", None), 
                                          params['depth'], params['width'], params['dropout'], params['normalization'], params['activation'],
                                          params['optimizer'], params['ssl_learning_rate'], params['ssl_weight_decay'])
                self.decoder2 = build_mlp(self.tasktype, params.get("width", None), params.get("input_dim", None), 
                                          params['depth'], params['width'], params['dropout'], params['normalization'], params['activation'],
                                          params['optimizer'], params['ssl_learning_rate'], params['ssl_weight_decay'])
                self.decoder1.to(self.device); self.decoder2.to(self.device)
            else:
                self.decoder = build_mlp(self.tasktype, params.get("width", None), params.get("input_dim", None), 
                                         params['depth'], params['width'], params['dropout'], params['normalization'], params['activation'],
                                         params['optimizer'], params['ssl_learning_rate'], params['ssl_weight_decay'])
                self.decoder.to(self.device)
            
            self.eval_lineareval = build_mlp(self.tasktype, params.get("width", None), params.get("output_dim", None), 
                                             1, params['width'], params['dropout'], params['normalization'], params['activation'],
                                             params['optimizer'], params['ssl_learning_rate'], params['ssl_weight_decay'])
            self.eval_lineareval.to(self.device)
            
            self.eval_finetuning = build_mlp(self.tasktype, params.get("width", None), params.get("output_dim", None), 
                                             1, params['width'], params['dropout'], params['normalization'], params['activation'],
                                             params['optimizer'], params['ssl_learning_rate'], params['ssl_weight_decay'])
            self.eval_finetuning.to(self.device)
            self.eval_lr = LogisticRegression()            
            self.eval_knn = KNeighborsClassifier(n_neighbors=params["k"])
    
    def fit(self, X_train, y_train):
            
        if self.ssl_loss == "vime":
            batch_size = 128
            ssl_optimizer = torch.optim.RMSprop(
                chain(self.encoder.parameters(), self.decoder1.parameters(), self.decoder2.parameters()), 
                lr=self.params['ssl_learning_rate'], weight_decay=self.params['ssl_weight_decay']
            )
        else:
            batch_size = 100
            ssl_optimizer = torch.optim.AdamW(
                chain(self.encoder.parameters(), self.decoder.parameters()), 
                lr=self.params['ssl_learning_rate'], weight_decay=self.params['ssl_weight_decay']
            )
        
        ft_optimizer = torch.optim.AdamW(
            chain(self.encoder.parameters(), self.eval_finetuning.parameters()), 
            lr=self.params['ft_learning_rate'], weight_decay=self.params['ft_weight_decay']
        )
        le_optimizer = torch.optim.AdamW(
            self.eval_lineareval.parameters(), 
            lr=self.params['le_learning_rate'], weight_decay=self.params['le_weight_decay']
        )
        
        if not os.path.exists(f'ssl-history/{self.ssl_loss}/{self.data_id}'):
            os.makedirs(f'ssl-history/{self.ssl_loss}/{self.data_id}')
        
        n_samples = len(X_train)
        train_idx = np.random.choice(n_samples, int(0.9*n_samples), replace=False)        
        if self.ssl_loss == "mse_binning":
            try: 
                X_bin_train = torch.load(f'/home/SemiTab/binning_data/{self.data_id}.pt').to(self.device)
            except FileNotFoundError:
                X_bin_train = Binning(X_train, num_bins=20, device=self.device, binning_reg=True)
                torch.save(X_bin_train, f'/home/SemiTab/binning_data/{self.data_id}.pt')
            train_dataset = torch.utils.data.TensorDataset(X_train[train_idx], X_bin_train[train_idx], y_train[train_idx])
            val_dataset = torch.utils.data.TensorDataset(X_train[~train_idx], X_bin_train[~train_idx], y_train[~train_idx])
        else:
            train_dataset = torch.utils.data.TensorDataset(X_train[train_idx], X_train[train_idx], y_train[train_idx])
            val_dataset = torch.utils.data.TensorDataset(X_train[~train_idx], X_train[~train_idx], y_train[~train_idx])
        
        early_stopping = EarlyStopping(patience=3, verbose=False, path=f'ssl-history/{self.ssl_loss}/{self.data_id}/logs.pt')
        
        labeled_flag = torch.unique(torch.where(~torch.isnan(y_train))[0])
        label_X_train = X_train[labeled_flag]
        label_y_train = y_train[labeled_flag]
        ft_dataset = torch.utils.data.TensorDataset(label_X_train, label_y_train)
        ft_batch_size = 100
        del X_train, y_train
        
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        ft_loader = torch.utils.data.DataLoader(dataset=ft_dataset, batch_size=ft_batch_size, shuffle=True)
        
        ssl_optimizer.zero_grad(); ssl_optimizer.step()
        
        if self.params["ssl_lr_scheduler"] & (len(train_loader) > 0):
            ssl_scheduler = CosineAnnealingLR_Warmup(ssl_optimizer, base_lr=self.params['ssl_learning_rate'], warmup_epochs=0, 
                                                     T_max=self.params.get('ssl_epochs'), iter_per_epoch=len(train_loader), warmup_lr=1e-6, eta_min=0, last_epoch=-1)
        if self.params.get("le_lr_scheduler", False):
            le_scheduler = CosineAnnealingLR_Warmup(le_optimizer, base_lr=self.params['le_learning_rate'], warmup_epochs=0, 
                                                    T_max=self.params.get('le_epochs'), iter_per_epoch=len(ft_loader), 
                                                    warmup_lr=1e-6, eta_min=0, last_epoch=-1)
        if self.params.get("ft_lr_scheduler", False):
            ft_scheduler = CosineAnnealingLR_Warmup(ft_optimizer, base_lr=self.params['ft_learning_rate'], warmup_epochs=0, 
                                                    T_max=self.params.get('ft_epochs'), iter_per_epoch=len(ft_loader), 
                                                    warmup_lr=1e-6, eta_min=0, last_epoch=-1)
        
        ## ssl first
        pbar = tqdm(range(1, self.params.get('ssl_epochs', 0) + 1))
        for epoch in pbar:
            pbar.set_description("EPOCH: %i" %epoch)
            
            for i, (x, xbin, y) in enumerate(train_loader):
                x_ = self.transform_func({'image': x, 'mask': None})
                
                if self.ssl_loss == "vime":
                    self.encoder.train(); self.decoder1.train(); self.decoder2.train(); ssl_optimizer.zero_grad()
                    xhat1 = self.decoder1(self.encoder(x_["image"], cat_features=self.cat_features))
                    xhat2 = self.decoder2(self.encoder(x_["image"], cat_features=self.cat_features))
                else:
                    self.encoder.train(); self.decoder.train(); ssl_optimizer.zero_grad()
                    xhat = self.decoder(self.encoder(x_["image"], cat_features=self.cat_features))
                    
                if self.ssl_loss == "mse_recon":
                    ssl_loss = torch.nn.functional.mse_loss(xhat, x.to(self.device))
                elif self.ssl_loss == "clf_mask":
                    ssl_loss = torch.nn.functional.binary_cross_entropy_with_logits(xhat, x_['mask'].to(torch.float32))
                elif self.ssl_loss == "mse_binning":
                    ssl_loss = torch.nn.MSELoss()(xhat, xbin.to(torch.float32))
                elif self.ssl_loss == "vime":
                    ssl_loss = torch.nn.functional.binary_cross_entropy_with_logits(xhat1, x_['mask'].to(torch.float32)) + 2*torch.nn.functional.mse_loss(xhat2, x.to(self.device))
                
                ssl_optimizer.zero_grad(); ssl_loss.backward(); ssl_optimizer.step()
                
                if self.params["ssl_lr_scheduler"]:
                    ssl_scheduler.step()
                
                pbar.set_postfix_str(f'data_id: {self.data_id}, Model: {self.modelname}, Tr loss: {ssl_loss:.5f}')
            
            self.encoder.eval(); self.decoder.eval()
            eval_loss = 0.0
            with torch.no_grad():
                for (x, xbin, y) in val_loader:
                    x_ = self.transform_func({'image': x, 'mask': None})
                    xhat = self.decoder(self.encoder(x_["image"], cat_features=self.cat_features))
                    if self.ssl_loss == "mse_recon":
                        val_loss = torch.nn.functional.mse_loss(xhat, x.to(self.device))
                    elif self.ssl_loss == "clf_mask":
                        val_loss = torch.nn.functional.binary_cross_entropy_with_logits(xhat, x_['mask'].to(torch.float32))
                    elif self.ssl_loss == "mse_binning":
                        val_loss = torch.nn.MSELoss()(xhat, xbin.to(torch.float32))
                    elif self.ssl_loss == "vime":
                        val_loss = torch.nn.functional.binary_cross_entropy_with_logits(xhat1, x_['mask'].to(torch.float32)) + 2*torch.nn.functional.mse_loss(xhat2, x.to(self.device))
                    eval_loss += val_loss.item()
            eval_loss /= len(val_loader.dataset)
            early_stopping(eval_loss, self.encoder)

            if early_stopping.early_stop:
                print("Early stopped at %i" %epoch)
                best_weights = torch.load(f'ssl-history/{self.ssl_loss}/{self.data_id}/logs.pt')
                self.encoder.load_state_dict(best_weights)
                self.stop_epoch = epoch
                break
        
        print("SSL training is completed! Start evaluation.")
        
        if self.tasktype == "regression":
            loss_fn = torch.nn.functional.mse_loss
        elif self.tasktype == "binclass":
            loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
        else:
            loss_fn = torch.nn.functional.cross_entropy
        
        ## logistic regression, knn
        if self.tasktype != "regression":
            with torch.no_grad():
                z = self.encoder(label_X_train, cat_features=self.cat_features)
                y = label_y_train.cpu().numpy()
                if self.tasktype == "multiclass":
                    y = np.argmax(y, axis=1)
                self.eval_lr.fit(z.cpu().numpy(), y)
                self.eval_knn.fit(z.cpu().numpy(), y)
        
        ## linear eval
        print("Linear evaluation")
        for epoch in tqdm(range(1, self.params.get('le_epochs', 0) + 1)):
            pbar.set_description("Linear eval. EPOCH: %i" %epoch)
            for i, (x, y) in enumerate(ft_loader):
                self.encoder.eval(); self.eval_lineareval.train(); le_optimizer.zero_grad()
                with torch.no_grad():
                    z = self.encoder(x, cat_features=self.cat_features)
                yhat = self.eval_lineareval(z)
                if self.tasktype == "binclass":
                    le_loss = loss_fn(y.to(self.device).view(-1, 1), yhat)
                else:
                    le_loss = loss_fn(y.to(self.device), yhat)
                le_optimizer.zero_grad(); le_loss.backward(); le_optimizer.step()
                if self.params["le_lr_scheduler"]:
                    le_scheduler.step()
                pbar.set_postfix_str(f'data_id: {self.data_id}, Model: {self.modelname}, Tr loss: {le_loss:.5f}')

        ## finetuning
        for epoch in tqdm(range(1, self.params.get('ft_epochs', 0) + 1)):
            pbar.set_description("Finetuning EPOCH: %i" %epoch)

            for i, (x, y) in enumerate(ft_loader):
                self.encoder.train(); self.eval_finetuning.train()
                z = self.encoder(x, cat_features=self.cat_features)
                yhat = self.eval_finetuning(z)
                if self.tasktype == "binclass":
                    ft_loss = loss_fn(y.to(self.device).view(-1, 1), yhat)
                else:
                    ft_loss = loss_fn(y.to(self.device), yhat)
                ft_optimizer.zero_grad(); ft_loss.backward(); ft_optimizer.step(); ft_optimizer.zero_grad()
                if self.params["ft_lr_scheduler"]:
                    ft_scheduler.step()
                pbar.set_postfix_str(f'data_id: {self.data_id}, Model: {self.modelname}, Tr loss: {ft_loss:.5f}')

    def predict(self, X_test):
        
        self.encoder.eval(); self.eval_lineareval.eval(); self.eval_finetuning.eval()
        with torch.no_grad():
            z = self.encoder(X_test, cat_features=self.cat_features)
            if self.tasktype != "regression":
                pred_lr = self.eval_lr.predict(z.cpu().numpy())
                pred_knn = self.eval_knn.predict(z.cpu().numpy())
            else:
                pred_lr, pred_knn = None, None
            pred_le = self.eval_lineareval(z)
            pred_ft = self.eval_finetuning(z)
            if self.tasktype == "binclass":
                pred_le = torch.sigmoid(pred_le).round()
                pred_ft = torch.sigmoid(pred_ft).round()
            elif self.tasktype == "multiclass":
                pred_le = torch.argmax(pred_le, dim=1)
                pred_ft = torch.argmax(pred_ft, dim=1)
        
        return pred_lr, pred_knn, pred_le.cpu().numpy(), pred_ft.cpu().numpy()
            
    def predict_proba(self, X_test, logit=False):
        with torch.no_grad():
            z = self.encoder(X_test, cat_features=self.cat_features)
            pred_lr = self.eval_lr.predict_proba(z.cpu().numpy())
            pred_knn = self.eval_knn.predict_proba(z.cpu().numpy())
            pred_le = self.eval_lineareval(z)
            pred_ft = self.eval_finetuning(z)

            if logit or (self.tasktype == "regression"):
                return pred_lr, pred_knn, pred_le.cpu().numpy(), pred_ft.cpu().numpy()
            else:
                return pred_lr, pred_knn, torch.nn.functional.softmax(pred_le).cpu().numpy(), torch.nn.functional.softmax(pred_ft).cpu().numpy() 