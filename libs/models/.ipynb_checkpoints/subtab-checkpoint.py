## Reproduction code of SubTab. Because we do not admit HPO to maintain the semi-supervised setup, we use the same hyperparameter setups for all datasets as written in the original paper (Table A.1, Section C.4, Section G.2).

import torch
from tqdm import tqdm
import numpy as np
from libs.models.mlp import build_mlp
from itertools import chain, combinations
from libs.utils import CosineAnnealingLR_Warmup
from libs.transform import NoiseMasking
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

class JointLoss(torch.nn.Module):
    def __init__(self,
                 n_subsets: int,
                 use_contrastive: bool = False,
                 use_distance: bool = True,
                 use_cosine_similarity: bool = False
        ) -> None:
        super(JointLoss, self).__init__()

        self.n_subsets = n_subsets
        self.use_cosine_similarity = use_cosine_similarity
        
        self.similarity_fn = self._cosine_simililarity if use_cosine_similarity else self._dot_simililarity
        self.mse_loss = torch.nn.MSELoss()
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        
        self.use_contrastive = use_contrastive
        self.use_distance = use_distance

    @staticmethod
    def _dot_simililarity(x, y):
        x = x.unsqueeze(1)
        y = y.T.unsqueeze(0)
        similarity = torch.tensordot(x, y, dims=2)
        return similarity

    def _cosine_simililarity(self, x, y):
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        return F.cosine_similarity(x, y, dim=-1)
    
    def get_anchor_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        batch_size = similarity.size(0)
        group_size = self.n_subsets
        
        identity_mask = torch.eye(
            batch_size, dtype=torch.bool, device=similarity.device
        )

        group_indices = torch.arange(batch_size, device=similarity.device) // group_size
        group_mask = group_indices.unsqueeze(0) == group_indices.unsqueeze(1)

        positives_mask = group_mask & ~identity_mask
        negatives_mask = ~group_mask

        pos_sum = torch.sum(torch.exp(similarity) * positives_mask.float(), dim=1)
        neg_sum = torch.sum(torch.exp(similarity) * negatives_mask.float(), dim=1)

        pos_sum = torch.clamp(pos_sum, min=1e-10)
        anchor_loss = -torch.log(pos_sum / (pos_sum + neg_sum))

        return anchor_loss
        
    def XNegloss(self, projections: torch.FloatTensor) -> torch.Tensor:
        similarity = self.similarity_fn(projections, projections)
        anchor_losses = self.get_anchor_loss(similarity)
        return anchor_losses.mean()

    def forward(self, projections, xrecon, xorig):
        recon_loss = self.mse_loss(xrecon, xorig)
        closs, dist_loss = None, None
        loss = recon_loss

        if self.use_contrastive:
            closs = self.XNegloss(projections)
            loss += closs

        if self.use_distance:
            combi = np.array(list(combinations(range(self.n_subsets), 2)))
            left = combi[:, 0]
            right = combi[:, 1]
            
            indices = torch.arange(len(projections)).view(-1, self.n_subsets)
            left_indices = indices[:, left].reshape(-1)
            right_indices = indices[:, right].reshape(-1)
            
            dist_loss = self.mse_loss(projections[left_indices], projections[right_indices])
            
            loss += dist_loss

        return loss

class subtab(torch.nn.Module):
    def __init__(self, params, tasktype, device, data_id=None, modelname=None, cat_features=[]):
        
        super(subtab, self).__init__()
        
        self.tasktype = tasktype
        self.cat_features = cat_features
        self.device = device
        self.params = params
        self.data_id = data_id
        self.modelname = modelname
        
        self.loss_fn = JointLoss(n_subsets=self.params["subsets"])
        
        self.n_column_subset = int(self.params["input_dim"] / self.params["subsets"])
        self.n_overlap = int(self.params["overlap"] * self.n_column_subset)
        self.column_idx = np.array(range(self.params["input_dim"]))
        
        if self.modelname == "mlp":
            self.encoder = build_mlp(self.tasktype, self.n_column_subset + self.n_overlap, params.get("width", None), 
                                     params['depth'], params['width'], params['dropout'], params['normalization'], params['activation'],
                                     params['optimizer'], params['ssl_learning_rate'], params['ssl_weight_decay'])
            self.encoder.to(self.device)
            self.decoder = build_mlp(self.tasktype, params.get("width", None), params.get("width", None), 
                                     params['depth'], params['width'], params['dropout'], params['normalization'], params['activation'],
                                     params['optimizer'], params['ssl_learning_rate'], params['ssl_weight_decay'])
            self.decoder.to(self.device)
            self.projector = build_mlp(self.tasktype, params.get("width", None), params.get("input_dim", None), 
                                       1, params['width'], params['dropout'], params['normalization'], params['activation'],
                                       params['optimizer'], params['ssl_learning_rate'], params['ssl_weight_decay'])
            self.projector.to(self.device)
            
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
    
    def subset_generator(self, x):
        
        transform_func = NoiseMasking(alpha=self.params["mask_ratio"], noise_level=self.params["noise"])
        permuted_order = np.arange(self.params["subsets"])
        subset_column_indice_list = [self.column_idx[:self.n_column_subset + self.n_overlap]]
        subset_column_indice_list.extend([self.column_idx[range(k * self.n_column_subset - self.n_overlap, (k + 1) * self.n_column_subset)] for k in range(self.params["subsets"])])

        subset_column_indice = np.array(subset_column_indice_list)
        subset_column_indice = subset_column_indice[permuted_order]

        if len(subset_column_indice) == 1:
            subset_column_indice = np.concatenate([subset_column_indice, subset_column_indice])

        x_ = []
        for i in range(self.params["subsets"]):
            x_.append(transform_func({"image": x[:, subset_column_indice[i]], "mask": None})["image"])
            
        return torch.concat(x_)
    
    def fit(self, X_train, y_train):
            
        batch_size = 256 ## original paper (Table A1) -- (frequently) best parameter
        ssl_optimizer = torch.optim.AdamW(
            chain(self.encoder.parameters(), self.decoder.parameters(), self.projector.parameters()), 
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
        
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        
        labeled_flag = torch.unique(torch.where(~torch.isnan(y_train))[0])
        label_X_train = X_train[labeled_flag]
        label_y_train = y_train[labeled_flag]
        ft_dataset = torch.utils.data.TensorDataset(label_X_train, label_y_train)
        ft_batch_size = 100
        del X_train, y_train
        
        if len(train_dataset) % batch_size == 1:
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True) ## prevent error for batchnorm
        else:
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            
        if len(ft_dataset) % ft_batch_size == 1:
            ft_loader = torch.utils.data.DataLoader(dataset=ft_dataset, batch_size=ft_batch_size, shuffle=True, drop_last=True) ## prevent error for batchnorm
        else:
            ft_loader = torch.utils.data.DataLoader(dataset=ft_dataset, batch_size=ft_batch_size, shuffle=True)
        
        ssl_optimizer.zero_grad(); ssl_optimizer.step()
        
        if self.params["ssl_lr_scheduler"]:
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
            
            for i, (x, y) in enumerate(train_loader):                
                
                x_ = self.subset_generator(x)
                
                self.encoder.train(); self.decoder.train(); ssl_optimizer.zero_grad()
                projections = self.decoder(self.encoder(x_))
                x_recons = self.projector(projections)
                
                if self.params["agg"] == "mean":
                    x_recons = x_recons.view(x.size(0), x.size(1), self.params["subsets"])
                    x_recons = x_recons.mean(-1)
                
                ssl_loss = self.loss_fn(projections, x_recons, x)                
                ssl_optimizer.zero_grad(); ssl_loss.backward(); ssl_optimizer.step()
                
                if self.params["ssl_lr_scheduler"]:
                    ssl_scheduler.step()
                
                pbar.set_postfix_str(f'data_id: {self.data_id}, Model: {self.modelname}, Tr loss: {ssl_loss:.5f}')
        
        print("SSL training is completed! Start evaluation.")
        
        if self.tasktype == "regression":
            loss_fn = torch.nn.functional.mse_loss
        elif self.tasktype == "binclass":
            loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
        else:
            loss_fn = torch.nn.functional.cross_entropy
        
        ## logistic regression, knn
        with torch.no_grad():
            N = label_X_train.size(0)
            label_X_train = self.subset_generator(label_X_train)
            z = self.encoder(label_X_train, cat_features=self.cat_features)
            
            z = z.view(N, z.size(1), self.params["subsets"])
            z = z.mean(-1)
            
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
                self.encoder.eval(); self.eval_lineareval.train(); le_optimizer.zero_grad()
                with torch.no_grad():
                    x_ = self.subset_generator(x)
                    z = self.encoder(x_, cat_features=self.cat_features)
                    z = z.view(x.size(0), z.size(1), self.params["subsets"])
                    z = z.mean(-1)
                    
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
                self.encoder.train(); self.eval_finetuning.train(); ft_optimizer.zero_grad()
                x_ = self.subset_generator(x)
                z = self.encoder(x_, cat_features=self.cat_features)
                z = z.view(x.size(0), z.size(1), self.params["subsets"])
                z = z.mean(-1)
                
                yhat = self.eval_finetuning(z)
                if self.tasktype == "binclass":
                    ft_loss = loss_fn(y.to(self.device).view(-1, 1), yhat)
                else:
                    ft_loss = loss_fn(y.to(self.device), yhat)
                ft_optimizer.zero_grad(); ft_loss.backward(); ft_optimizer.step()
                if self.params["ft_lr_scheduler"]:
                    ft_scheduler.step()
                pbar.set_postfix_str(f'data_id: {self.data_id}, Model: {self.modelname}, Tr loss: {ft_loss:.5f}')

    def predict(self, X_test):
        
        self.encoder.eval(); self.eval_lineareval.eval(); self.eval_finetuning.eval()
        with torch.no_grad():
            N = X_test.size(0)
            X_test = self.subset_generator(X_test)
            z = self.encoder(X_test, cat_features=self.cat_features)
            z = z.view(N, z.size(1), self.params["subsets"])
            z = z.mean(-1)
            
            pred_lr = self.eval_lr.predict(z.cpu().numpy())
            pred_knn = self.eval_knn.predict(z.cpu().numpy())
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
            N = X_test.size(0)
            X_test = self.subset_generator(X_test)
            z = self.encoder(X_test, cat_features=self.cat_features)
            z = z.view(N, z.size(1), self.params["subsets"])
            z = z.mean(-1)
                
            pred_lr = self.eval_lr.predict_proba(z.cpu().numpy())
            pred_knn = self.eval_knn.predict_proba(z.cpu().numpy())
            pred_le = self.eval_lineareval(z)
            pred_ft = self.eval_finetuning(z)

            if logit:
                return pred_lr, pred_knn, pred_le.cpu().numpy(), pred_ft.cpu().numpy()
            else:
                return pred_lr, pred_knn, torch.nn.functional.softmax(pred_le).cpu().numpy(), torch.nn.functional.softmax(pred_ft).cpu().numpy() 
