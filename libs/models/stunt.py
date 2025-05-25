# # Reproduction code of SubTab. Because we do not admit HPO to maintain the semi-supervised setup, we use the same hyperparameter setups for all datasets as written in the original paper (Table A.1, Section C.4, Section G.2).

import torch
from tqdm import tqdm
import numpy as np
from libs.models.mlp import build_mlp
from itertools import chain, combinations

from re import I
import os
import copy
# import faiss



### from torchmeta library
def get_num_samples(targets, num_classes, dtype=None):
    batch_size = targets.size(0)
    with torch.no_grad():
        ones = torch.ones_like(targets, dtype=dtype)
        num_samples = ones.new_zeros((batch_size, num_classes))
        num_samples.scatter_add_(1, targets, ones)
    return num_samples


### from torchmeta library
def get_prototypes(embeddings, targets, num_classes):
    """Compute the prototypes (the mean vector of the embedded training/support 
    points belonging to its class) for each classes in the task.

    Parameters
    ----------
    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the support points. This tensor 
        has shape `(batch_size, num_examples, embedding_size)`.

    targets : `torch.LongTensor` instance
        A tensor containing the targets of the support points. This tensor has 
        shape `(batch_size, num_examples)`.

    num_classes : int
        Number of classes in the task.

    Returns
    -------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape
        `(batch_size, num_classes, embedding_size)`.
    """
    batch_size, embedding_size = embeddings.size(0), embeddings.size(-1)
    
    num_samples = get_num_samples(targets, num_classes, dtype=embeddings.dtype)
    num_samples.unsqueeze_(-1)
    num_samples = torch.max(num_samples, torch.ones_like(num_samples))

    prototypes = embeddings.new_zeros((batch_size, num_classes, embedding_size))
    indices = targets.unsqueeze(-1).expand_as(embeddings)
    prototypes.scatter_add_(1, indices, embeddings).div_(num_samples)

    return prototypes

def test_classifier(model, loader, criterion, device, proba=False):
    model.eval()
    
    steps = loader.val_x.shape[0] // 100 + 1
    result = []
    for _ in range(steps):
        batch = loader.get_batch()
        train_inputs, train_targets = batch['train']
        
        num_ways = len(set(list(train_targets[0].numpy())))            
        train_inputs = train_inputs.to(device)
        train_targets = train_targets.to(device)
        with torch.no_grad():
            train_embeddings = model(train_inputs)

        test_inputs, test_targets = batch['test']
        test_inputs = test_inputs.to(device)
        test_targets = test_targets.to(device)
        with torch.no_grad():
            test_embeddings = model(test_inputs)
        
        prototypes = get_prototypes(train_embeddings, train_targets, num_ways)

        squared_distances = torch.sum((prototypes.unsqueeze(2)
                                    - test_embeddings.unsqueeze(1)) ** 2, dim=-1)
        loss = criterion(-squared_distances, test_targets)
        
        result.append(get_accuracy(prototypes, test_embeddings, test_targets, proba=proba).item())
    
    if proba:
        return torch.stack(result, axis=0)
    else:
        return np.mean(result)

def get_accuracy(prototypes, embeddings, targets, proba=False):
        
    sq_distances = torch.sum((prototypes.unsqueeze(1)
        - embeddings.unsqueeze(2)) ** 2, dim=-1)
    _, predictions = torch.min(sq_distances, dim=-1)

    if proba:
        return predictions
    else:
        return torch.mean(predictions.eq(targets).float())

def generate_pseudo_val(X_train, y_train):
        
    ## following the official code (https://github.com/jaehyun513/STUNT/blob/main/data/income/generate_pseudo_val.py)
    torch.manual_seed(0)

    num_train = int(len(X_train) * 0.8)
    idx = torch.randperm(len(X_train))
    train_idx = idx[:num_train]
    val_idx = idx[num_train:]

    train_x = X_train[train_idx]
    val_x = X_train[val_idx]

    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=2)
    model.fit(val_x)
    labels = model.predict(val_x)

    return train_x, val_x, labels

def kmeans_pytorch(X, num_clusters, num_iters=20):
    
    ## to prevent duplicated values
    epsilon = 1e-6
    X = X + epsilon * torch.randn_like(X)
    
    centroids = X[torch.randint(0, len(X), (num_clusters,))].clone()
    for i in range(num_iters):
        distances = torch.cdist(X, centroids, p=2)
        cluster_assignments = torch.argmin(distances, dim=1)
        new_centroids = torch.stack([
            X[cluster_assignments == j].mean(dim=0) if (cluster_assignments == j).sum() > 0 else centroids[j]
            for j in range(num_clusters)
        ])
        if torch.all(new_centroids == centroids):
            break
        unique_centroids = torch.unique(new_centroids, dim=0)
        if len(unique_centroids) < num_clusters:
            missing_centroids = num_clusters - len(unique_centroids)
            new_random_centroids = X[torch.randint(0, len(X), (missing_centroids,))]
            new_centroids = torch.cat([unique_centroids, new_random_centroids], dim=0)
        centroids = new_centroids

    return centroids, cluster_assignments

class stuntdataset(object):
    def __init__(self, train_x, train_y, 
                 tabular_size, seed, source, shot, tasks_per_batch, test_num_way, query):
        super().__init__()
        self.num_classes = 2
        self.tabular_size = tabular_size
        self.source = source
        self.shot = shot
        self.query = query
        self.tasks_per_batch = tasks_per_batch
        train_x, val_x, val_y = generate_pseudo_val(train_x, train_y)
        self.unlabeled_x = train_x #np.load('./data/income/train_x.npy')
        self.val_x = val_x #np.load('./data/income/val_x.npy')
        self.val_y = val_y #np.load('./data/income/val_y.npy') # val_y is given from pseudo-validaiton scheme with STUNT
        self.test_num_way = test_num_way
        self.test_rng = np.random.RandomState(seed)
        self.val_rng = np.random.RandomState(seed)
        
    def __next__(self):
        return self.get_batch()

    def __iter__(self):
        return self

    def get_batch(self):
        xs, ys, xq, yq = [], [], [], []
        if self.source == 'train':
            x = self.unlabeled_x
            num_way = self.test_num_way

        elif self.source == 'val':
            x = self.val_x
            y = self.val_y
            class_list,_ = np.unique(y, return_counts=True) 
            num_val_shot = 1
            num_way = 2
        
        for _ in range(self.tasks_per_batch):
            
            support_set = []
            query_set = []
            support_sety = []
            query_sety = []

            if self.source == 'val':

                classes = np.random.choice(class_list, num_way, replace = False)
                support_idx = []
                query_idx = []
                for k in classes:
                    k_idx = np.where(y == k)[0]
                    permutation = np.random.permutation(len(k_idx))
                    k_idx = k_idx[permutation]
                    support_idx.append(k_idx[:num_val_shot])
                    query_idx.append(k_idx[num_val_shot:num_val_shot+30])
                support_idx = np.concatenate(support_idx)
                query_idx = np.concatenate(query_idx)
                
                support_x = x[support_idx]
                query_x = x[query_idx]
                s_y = y[support_idx]
                q_y = y[query_idx]
                support_y = copy.deepcopy(s_y)
                query_y = copy.deepcopy(q_y)

                i = 0
                for k in classes:
                    support_y[s_y == k] = i
                    query_y[q_y == k] = i
                    i+=1

                support_set.append(support_x)
                support_sety.append(support_y)
                query_set.append(query_x)
                query_sety.append(query_y)

            elif self.source == 'train':
                tmp_x = copy.deepcopy(x)
                min_count = 0
                while min_count < (self.shot + self.query):
                    min_col = np.max([1, int(x.shape[1] * 0.2)])  # prevent output 0
                    max_col = int(x.shape[1] * 0.5)
                    if min_col == max_col:
                        col = min_col
                    else:
                        col = np.random.choice(range(min_col, max_col), 1, replace=False)[0]
                    task_idx = np.random.choice([i for i in range(x.shape[1])], col, replace=False)
                    masked_x = torch.tensor(x[:, task_idx], dtype=torch.float32).to('cuda') ## randomly selected subset of full columns

                    centroids, cluster_assignments = kmeans_pytorch(masked_x, num_way, num_iters=20) ## implement clustering

                    y = cluster_assignments.cpu().numpy()
                    class_list = np.arange(num_way)
                    counts = np.bincount(y, minlength=len(class_list))
#                     class_list, counts = np.unique(y, return_counts=True)
                    min_count = min(counts)
                
                num_to_permute = x.shape[0]
                for t_idx in task_idx:
                    rand_perm = np.random.permutation(num_to_permute)
                    tmp_x[:, t_idx] = tmp_x[:, t_idx][rand_perm] 
            
                classes = np.arange(num_way) #np.random.choice(class_list, num_way, replace = False)
                    
                support_idx = []
                query_idx = []
                for k in classes:
                    k_idx = np.where(y == k)[0]
                    permutation = np.random.permutation(len(k_idx))
                    k_idx = k_idx[permutation]
                    support_idx.append(k_idx[:self.shot])
                    query_idx.append(k_idx[self.shot:self.shot + self.query])
                    
                support_idx = np.concatenate(support_idx)
                query_idx = np.concatenate(query_idx)
                
                support_x = tmp_x[support_idx]
                query_x = tmp_x[query_idx]
                s_y = y[support_idx]
                q_y = y[query_idx]
                support_y = copy.deepcopy(s_y)
                query_y = copy.deepcopy(q_y)
            
#                 i = 0
#                 for k in classes:
#                     support_y[s_y == k] = i
#                     query_y[q_y == k] = i
#                     i+=1

                support_set.append(support_x)
                support_sety.append(support_y)
                query_set.append(query_x)
                query_sety.append(query_y)
                    
            xs_k = np.concatenate(support_set, 0)
            xq_k = np.concatenate(query_set, 0)
            ys_k = np.concatenate(support_sety, 0)
            yq_k = np.concatenate(query_sety, 0)
            
            if xs_k.shape[0] != (self.shot * num_way):
                import IPython; IPython.embed()
            
            xs.append(xs_k)
            xq.append(xq_k)
            ys.append(ys_k)
            yq.append(yq_k)
        
        xs, ys = np.stack(xs, 0), np.stack(ys, 0)
        xq, yq = np.stack(xq, 0), np.stack(yq, 0)          

        if self.source == 'val':
            xs = np.reshape(
                xs,
                [self.tasks_per_batch, num_way * num_val_shot, self.tabular_size])
        else:
            xs = np.reshape(
                xs,
                [self.tasks_per_batch, num_way * self.shot, self.tabular_size])

        if self.source == 'val':
            xq = np.reshape(
                xq,
                [self.tasks_per_batch, -1, self.tabular_size])
        else:
            xq = np.reshape(
                xq,
                [self.tasks_per_batch, -1, self.tabular_size])

        xs = xs.astype(np.float32)
        xq = xq.astype(np.float32)
        ys = ys.astype(np.float32)
        yq = yq.astype(np.float32)

        xs = torch.from_numpy(xs).type(torch.FloatTensor)
        xq = torch.from_numpy(xq).type(torch.FloatTensor)

        ys = torch.from_numpy(ys).type(torch.LongTensor)
        yq = torch.from_numpy(yq).type(torch.LongTensor)         

        batch = {'train': (xs, ys), 'test': (xq, yq)}

        return batch

class stunt(torch.nn.Module):
    def __init__(self, params, tasktype, device, data_id=None, modelname=None, cat_features=[]):
        
        super(stunt, self).__init__()
        
        self.tasktype = tasktype
        self.cat_features = cat_features
        self.device = device
        self.params = params
        self.data_id = data_id
        self.modelname = modelname
        
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
        if self.modelname == "mlp":
            self.model = build_mlp(self.tasktype, params.get("input_dim", None), params.get("width", None), 
                                   params['depth'], params['width'], params['dropout'], params['normalization'], params['activation'],
                                   params['optimizer'], params['learning_rate'], params['weight_decay'])
            self.model.to(self.device)        
    
    def fit(self, X_train, y_train):
            
        optimizer = self.model.make_optimizer()
        device = X_train.device
        num_way = 3 if X_train.shape[0] < 1000 else 10
        
        meta_train_dataset = stuntdataset(X_train.cpu().numpy(), y_train.cpu().numpy(),
                                          tabular_size=self.params["input_dim"], seed=self.params["seed"], source="train", shot=self.params["shots"], 
                                          tasks_per_batch=4, test_num_way=num_way, query=15) ##original paper: Appendix B
        meta_val_dataset = stuntdataset(X_train.cpu().numpy(), y_train.cpu().numpy(),
                                        tabular_size=self.params["input_dim"], seed=self.params["seed"], source="val", shot=1, tasks_per_batch=np.min([100, X_train.size(0)]), 
                                        test_num_way=2, query=30)
        
        optimizer.zero_grad(); optimizer.step()
        
        pbar = tqdm(range(1, self.params["n_steps"] + 1))
        best = 0
        for step in pbar:
            pbar.set_description("STEP: %i" %step)
            
            train_batch = next(meta_train_dataset)
            train_inputs, train_targets = train_batch["train"]
            test_inputs, test_targets = train_batch["test"]
            
            train_inputs = train_inputs.to(device)
            train_targets = train_targets.to(device)
            mask_prob = np.random.uniform(0.2, 0.5)
            mask = np.random.uniform(0, 1, size=train_inputs.shape) < mask_prob
            train_inputs[torch.tensor(mask).to(train_inputs.device)] = 0.
            train_embeddings = self.model(train_inputs)
            
            test_inputs = test_inputs.to(device)
            test_targets = test_targets.to(device)
            test_embeddings = self.model(test_inputs)
            
            prototypes = get_prototypes(train_embeddings, train_targets, 10)
    
            squared_distances = torch.sum((prototypes.unsqueeze(2) - test_embeddings.unsqueeze(1)) ** 2, dim=-1)
            loss = self.loss_fn(-squared_distances, test_targets)

            """ outer gradient step """
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = get_accuracy(prototypes, test_embeddings, test_targets).item()
            pbar.set_postfix_str(f'data_id: {self.data_id}, Model: {self.modelname}, Tr loss: {loss:.5f}, Train Acc: {acc:.5f}, Val Acc: {best:.5f}')
            
            if step % self.params["eval_step"] == 0:
                val_acc = test_classifier(self.model, meta_val_dataset, torch.nn.CrossEntropyLoss(), device)
                if best > val_acc:
                    self.stop_epoch = step
                    break
                else:
                    best = val_acc
    
    def evaluate(self, train_x, train_y, test_x, test_y):
        
        output_size = 2 if self.tasktype == "binclass" else train_y.size(1)
        if self.tasktype == "multiclass":
            train_y = torch.argmax(train_y, dim=1)
            test_y = torch.argmax(test_y, dim=1)
        
        train_idx = torch.unique(torch.where(~torch.isnan(train_y))[0])
        
        few_train = self.model(torch.tensor(train_x[train_idx].to(self.device)).float())
        support_x = few_train.cpu().detach().numpy()
        support_y = train_y[train_idx].cpu().detach().numpy()
        few_test = self.model(torch.tensor(test_x).to(self.device).float())
        query_x = few_test.cpu().detach().numpy()
        query_y = test_y.cpu().detach().numpy()

        def get_accuracy(prototypes, embeddings, targets):

            sq_distances = torch.sum((prototypes.unsqueeze(1)
                - embeddings.unsqueeze(2)) ** 2, dim=-1)
            _, predictions = torch.min(sq_distances, dim=-1)
            return torch.mean(predictions.eq(targets).float()).item()

        train_x = torch.tensor(support_x.astype(np.float32)).unsqueeze(0)
        train_y = torch.tensor(support_y.astype(np.int64)).unsqueeze(0).type(torch.LongTensor)
        val_x = torch.tensor(query_x.astype(np.float32)).unsqueeze(0)
        val_y = torch.tensor(query_y.astype(np.int64)).unsqueeze(0).type(torch.LongTensor)
        
        prototypes = get_prototypes(train_x, train_y, output_size)
        acc = get_accuracy(prototypes, val_x, val_y)
        
        return acc       
        