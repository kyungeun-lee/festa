# # Reproduction code of SCARF. Reference: original repository

import torch, os
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions.uniform import Uniform
from tqdm import tqdm
import numpy as np
from libs.models.mlp import build_mlp
from itertools import chain, combinations
from libs.utils import CosineAnnealingLR_Warmup
from libs.transform import NoiseMasking
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from libs.transform import *

### scarf original code
class NTXent(nn.Module):
    def __init__(self, temperature: float = 1.0) -> None:
        """NT-Xent loss for contrastive learning using cosine distance as similarity metric as used in [SimCLR](https://arxiv.org/abs/2002.05709).
        Implementation adapted from https://theaisummer.com/simclr/#simclr-loss-implementation

        Args:
            temperature (float, optional): scaling factor of the similarity metric. Defaults to 1.0.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: Tensor, z_j: Tensor) -> Tensor:
        """Compute NT-Xent loss using only anchor and positive batches of samples. Negative samples are the 2*(N-1) samples in the batch

        Args:
            z_i (torch.tensor): anchor batch of samples
            z_j (torch.tensor): positive batch of samples

        Returns:
            float: loss
        """
        batch_size = z_i.size(0)

        # compute similarity between the sample's embedding and its corrupted view
        z = torch.cat([z_i, z_j], dim=0)
        similarity = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity, batch_size)
        sim_ji = torch.diag(similarity, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool, device=z_i.device)).float()
        numerator = torch.exp(positives / self.temperature)
        denominator = mask * torch.exp(similarity / self.temperature)

        all_losses = -torch.log(numerator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * batch_size)

        return loss

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
    
class scarf_model(torch.nn.Module):
    def __init__(self, params, tasktype, device, data_id=None, modelname="mlp", cat_features=[]):
        
        super().__init__()
        
        self.tasktype = tasktype
        self.cat_features = cat_features
        self.device = device
        self.params = params
        self.data_id = data_id
        self.modelname = modelname
        self.transform = params["transform"]
        
        # uniform disstribution over marginal distributions of dataset's features
        epsilon = 1e-10
        params["features_high"] = np.where(params["features_low"] == params["features_high"], params["features_high"] + epsilon, params["features_high"])
        self.marginals = Uniform(torch.Tensor(params["features_low"]), torch.Tensor(params["features_high"]))            
        
        self.corruption_rate = 0.6
        
        if self.modelname == "mlp":
            self.encoder = build_mlp(self.tasktype, params.get("input_dim", None), params.get("width", None), 
                                     params['depth'], params['width'], params['dropout'], params['normalization'], params['activation'],
                                     params['optimizer'], params['ssl_learning_rate'], params['ssl_weight_decay'])
            self.encoder.to(self.device)
            self.head = build_mlp(self.tasktype, params.get("width", None), params.get("width", None), 
                                  params['depth'], params['width'], params['dropout'], params['normalization'], params['activation'],
                                  params['optimizer'], params['ssl_learning_rate'], params['ssl_weight_decay'])
            self.head.to(self.device)
    
    def forward(self, x):
        batch_size, _ = x.size()
        
        if self.params.get("transform_level", None) is None:
            p = 10 if self.transform == "rq" else 0.3
        else:
            p = self.params.get("transform_level", None)
        
        
        if self.transform is None:
            aug = ToTensor()
        elif self.transform == "binshuffling":
            aug = BinShuffling(p, 4, self.params["num_features"], 0)
        elif self.transform.startswith("binshuffling-"):
            aug = BinShuffling(p, eval(self.transform.split("-")[-1]), self.params["num_features"], 0)
        elif self.transform == "masking":
            aug = Masking(p)
        elif self.transform == "shuffling":
            aug = Shuffling(p)
        elif self.transform == "noisemasking":
            aug = NoiseMasking(p)
        elif self.transform == "rq":
            aug = RandQuant(p)
        elif self.transform == "scarf":
            pass
        elif self.transform == "binsampling":
            aug = BinSampling(p, 4, self.params["num_features"], 0)
        elif self.transform.startswith("binsampling-"):
            aug = BinSampling(p, eval(self.transform.split("-")[-1]), self.params["num_features"], 0)
        else:
            raise ValueError
        
        if self.transform == "scarf":
            corruption_mask = torch.rand_like(x, device=x.device) > 1 - p
            x_random = self.marginals.sample(torch.Size((batch_size,))).to(x.device)
            x_corrupted = torch.where(corruption_mask, x_random, x)
        else:
            x_corrupted = aug({"image": x, "mask": None})["image"]

        # get embeddings
        embeddings = self.head(self.encoder(x))
        embeddings_corrupted = self.head(self.encoder(x_corrupted))

        return embeddings, embeddings_corrupted
    
    @torch.inference_mode()
    def get_embeddings(self, x: Tensor) -> Tensor:
        return self.encoder(x)
    
    
class SCARF(torch.nn.Module):
    def __init__(self, params, tasktype, device, data_id=None, modelname=None, cat_features=[]):
        
        super(SCARF, self).__init__()
        
        self.tasktype = tasktype
        self.cat_features = cat_features
        self.device = device
        self.params = params
        self.data_id = data_id
        self.modelname = modelname
        self.model = scarf_model(params, tasktype, device, data_id=data_id, modelname="mlp", cat_features=cat_features)
        self.eval_lineareval = build_mlp(self.tasktype, params.get("width", None), params.get("output_dim", None), 
                                         1, params['width'], params['dropout'], params['normalization'], params['activation'],
                                         params['optimizer'], params['le_learning_rate'], params['le_weight_decay'])
        self.eval_lineareval.to(self.device)

        self.eval_finetuning = build_mlp(self.tasktype, params.get("width", None), params.get("output_dim", None), 
                                         1, params['width'], params['dropout'], params['normalization'], params['activation'],
                                         params['optimizer'], params['ft_learning_rate'], params['ft_weight_decay'])
        self.eval_finetuning.to(self.device)
        self.eval_lr = LogisticRegression()            
        self.eval_knn = KNeighborsClassifier(n_neighbors=params["k"])

        self.loss_fn = NTXent()
        
    def fit(self, X_train, y_train):
            
        batch_size = 128
        ssl_optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.params['ssl_learning_rate'], weight_decay=self.params['ssl_weight_decay']
        )
        
        ft_optimizer = torch.optim.AdamW(
            chain(self.model.encoder.parameters(), self.eval_finetuning.parameters()), 
            lr=self.params['ft_learning_rate'], weight_decay=self.params['ft_weight_decay']
        )
        le_optimizer = torch.optim.AdamW(
            self.eval_lineareval.parameters(), 
            lr=self.params['le_learning_rate'], weight_decay=self.params['le_weight_decay']
        )
        
        if not os.path.exists(f'scarf-history/{self.params["transform"]}/{self.data_id}'):
            os.makedirs(f'scarf-history/{self.params["transform"]}/{self.data_id}')
            
        early_stopping = EarlyStopping(patience=3, verbose=False, path=f'scarf-history/{self.params["transform"]}/{self.data_id}/logs.pt')
        
        n_samples = len(X_train)
        train_idx = np.random.choice(n_samples, int(0.9*n_samples), replace=False)
        train_dataset = torch.utils.data.TensorDataset(X_train[train_idx], y_train[train_idx])
        val_dataset = torch.utils.data.TensorDataset(X_train[~train_idx], y_train[~train_idx])
        
        labeled_flag = torch.unique(torch.where(~torch.isnan(y_train))[0])
        label_X_train = X_train[labeled_flag]
        label_y_train = y_train[labeled_flag]
        ft_dataset = torch.utils.data.TensorDataset(label_X_train, label_y_train)
        ft_batch_size = 100
        del X_train, y_train
        
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        ft_loader = torch.utils.data.DataLoader(dataset=ft_dataset, batch_size=ft_batch_size, shuffle=True, drop_last=True)
        
        ssl_optimizer.zero_grad(); ssl_optimizer.step()
        
        if self.params["ssl_lr_scheduler"]:
            ssl_scheduler = CosineAnnealingLR_Warmup(ssl_optimizer, base_lr=self.params['ssl_learning_rate'], warmup_epochs=0, 
                                                     T_max=self.params.get('ssl_epochs'), iter_per_epoch=len(train_loader), warmup_lr=1e-6, eta_min=0, last_epoch=-1)
        if self.params.get("le_lr_scheduler", False) & (len(ft_loader) > 0):
            le_scheduler = CosineAnnealingLR_Warmup(le_optimizer, base_lr=self.params['le_learning_rate'], warmup_epochs=0, 
                                                    T_max=self.params.get('le_epochs'), iter_per_epoch=len(ft_loader), 
                                                    warmup_lr=1e-6, eta_min=0, last_epoch=-1)
        if self.params.get("ft_lr_scheduler", False) & (len(ft_loader) > 0):
            ft_scheduler = CosineAnnealingLR_Warmup(ft_optimizer, base_lr=self.params['ft_learning_rate'], warmup_epochs=0, 
                                                    T_max=self.params.get('ft_epochs'), iter_per_epoch=len(ft_loader), 
                                                    warmup_lr=1e-6, eta_min=0, last_epoch=-1)
        
        ## ssl first
        pbar = tqdm(range(1, self.params.get('ssl_epochs', 0) + 1))
        for epoch in pbar:
            pbar.set_description("EPOCH: %i" %epoch)
            
            for i, (x, y) in enumerate(train_loader):                
                
                self.model.train(); ssl_optimizer.zero_grad()
                emb_anchor, emb_positive = self.model(x.to(self.device))
                ssl_loss = self.loss_fn(emb_anchor, emb_positive)
                
                ssl_loss.backward(); ssl_optimizer.step(); ssl_optimizer.zero_grad()
                
                if self.params["ssl_lr_scheduler"]:
                    ssl_scheduler.step()
                
                pbar.set_postfix_str(f'data_id: {self.data_id}, Model: {self.modelname}, Tr loss: {ssl_loss:.5f}')
            
            self.model.eval()
            eval_loss = 0.0
            with torch.no_grad():
                for (x, y) in val_loader:
                    emb_anchor, emb_positive = self.model(x.to(self.device))
                    val_loss = self.loss_fn(emb_anchor, emb_positive)
                    eval_loss += val_loss.item()
            eval_loss /= len(val_loader.dataset)
            early_stopping(eval_loss, self.model.encoder)

            if early_stopping.early_stop:
                print("Early stopped at %i" %epoch)
                best_weights = torch.load(f'scarf-history/{self.params["transform"]}/{self.data_id}/logs.pt')
                self.model.encoder.load_state_dict(best_weights)
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
                z = self.model.get_embeddings(label_X_train.to(self.device))

                label_y_train = label_y_train.cpu().numpy()
                if self.tasktype == "multiclass":
                    label_y_train = np.argmax(label_y_train, axis=1)
                self.eval_lr.fit(z.cpu().numpy(), label_y_train)
                self.eval_knn.fit(z.cpu().numpy(), label_y_train)
        
        ## linear eval
        print("Linear evaluation")
        for epoch in tqdm(range(1, self.params.get('le_epochs', 0) + 1)):
            pbar.set_description("Linear eval. EPOCH: %i" %epoch)
            for i, (x, y) in enumerate(ft_loader):
                self.model.eval(); self.eval_lineareval.train(); le_optimizer.zero_grad()
                with torch.no_grad():
                    z = self.model.get_embeddings(x.to(self.device))
                try:
                    yhat = self.eval_lineareval(z)
                except RuntimeError:
                    yhat = self.eval_lineareval(z.clone())
                    
                if self.tasktype == "binclass":
                    le_loss = loss_fn(y.to(self.device).view(-1, 1), yhat)
                else:
                    le_loss = loss_fn(y.to(self.device), yhat)
                le_loss.backward(); le_optimizer.step()
                if self.params["le_lr_scheduler"] & (len(ft_loader) > 0):
                    le_scheduler.step()
                pbar.set_postfix_str(f'data_id: {self.data_id}, Model: {self.modelname}, Tr loss: {le_loss:.5f}')

        ## finetuning
        for epoch in tqdm(range(1, self.params.get('ft_epochs', 0) + 1)):
            pbar.set_description("Finetuning EPOCH: %i" %epoch)

            for i, (x, y) in enumerate(ft_loader):
                self.model.train(); self.eval_finetuning.train(); ft_optimizer.zero_grad()
                z = self.model.get_embeddings(x.to(self.device))                
                try:
                    yhat = self.eval_lineareval(z)
                except RuntimeError:
                    yhat = self.eval_lineareval(z.clone())
                if self.tasktype == "binclass":
                    ft_loss = loss_fn(y.to(self.device).view(-1, 1), yhat)
                else:
                    ft_loss = loss_fn(y.to(self.device), yhat)
                ft_loss.backward(); ft_optimizer.step()
                if self.params["ft_lr_scheduler"] & (len(ft_loader) > 0):
                    ft_scheduler.step()
                pbar.set_postfix_str(f'data_id: {self.data_id}, Model: {self.modelname}, Tr loss: {ft_loss:.5f}')

    def predict(self, X_test):
        
        self.model.eval(); self.eval_lineareval.eval(); self.eval_finetuning.eval()
        with torch.no_grad():
            N = X_test.size(0)
            z = self.model.get_embeddings(X_test.to(self.device))       
            
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
            elif self.tasktype == "regression":
                pass
        
        return pred_lr, pred_knn, pred_le.cpu().numpy(), pred_ft.cpu().numpy()
            
    def predict_proba(self, X_test, logit=False):
        with torch.no_grad():
            z = self.model.get_embeddings(X_test.to(self.device))
                
            pred_lr = self.eval_lr.predict_proba(z.cpu().numpy())
            pred_knn = self.eval_knn.predict_proba(z.cpu().numpy())
            pred_le = self.eval_lineareval(z)
            pred_ft = self.eval_finetuning(z)

            if logit or (self.tasktype == "regression"):
                return pred_lr, pred_knn, pred_le.cpu().numpy(), pred_ft.cpu().numpy()
            else:
                return pred_lr, pred_knn, torch.nn.functional.softmax(pred_le).cpu().numpy(), torch.nn.functional.softmax(pred_ft).cpu().numpy() 
