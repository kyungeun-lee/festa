# # Reference
# # SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training
# # https://github.com/somepago/saint/blob/main/models/model.py

import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch import nn, einsum
import numpy as np
from einops import rearrange
from itertools import chain, combinations
from libs.models.mlp import build_mlp
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from libs.utils import CosineAnnealingLR_Warmup

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def ff_encodings(x,B):
    x_proj = (2. * np.pi * x.unsqueeze(-1)) @ B.t()
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

# classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# attention

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 16,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out)


class RowColTransformer(nn.Module):
    def __init__(self, num_tokens, dim, nfeats, depth, heads, dim_head, attn_dropout, ff_dropout,style='col'):
        super().__init__()
        self.embeds = nn.Embedding(num_tokens, dim)
        self.layers = nn.ModuleList([])
        self.mask_embed =  nn.Embedding(nfeats, dim)
        self.style = style
        for _ in range(depth):
            if self.style == 'colrow':
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, Residual(Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout))),
                    PreNorm(dim, Residual(FeedForward(dim, dropout = ff_dropout))),
                    PreNorm(dim*nfeats, Residual(Attention(dim*nfeats, heads = heads, dim_head = 64, dropout = attn_dropout))),
                    PreNorm(dim*nfeats, Residual(FeedForward(dim*nfeats, dropout = ff_dropout))),
                ]))
            else:
                self.layers.append(nn.ModuleList([
                    PreNorm(dim*nfeats, Residual(Attention(dim*nfeats, heads = heads, dim_head = 64, dropout = attn_dropout))),
                    PreNorm(dim*nfeats, Residual(FeedForward(dim*nfeats, dropout = ff_dropout))),
                ]))

    def forward(self, x, x_cont=None, mask = None):
        if x_cont is not None:
            x = torch.cat((x,x_cont),dim=1)
        _, n, _ = x.shape
        if self.style == 'colrow':
            for attn1, ff1, attn2, ff2 in self.layers: 
                x = attn1(x)
                x = ff1(x)
                x = rearrange(x, 'b n d -> 1 b (n d)')
                x = attn2(x)
                x = ff2(x)
                x = rearrange(x, '1 b (n d) -> b n d', n = n)
        else:
             for attn1, ff1 in self.layers:
                x = rearrange(x, 'b n d -> 1 b (n d)')
                x = attn1(x)
                x = ff1(x)
                x = rearrange(x, '1 b (n d) -> b n d', n = n)
        return x


# transformer
class Transformer(nn.Module):
    def __init__(self, num_tokens, dim, depth, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()
        self.layers = nn.ModuleList([])


        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Residual(Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout))),
                PreNorm(dim, Residual(FeedForward(dim, dropout = ff_dropout))),
            ]))

    def forward(self, x, x_cont=None):
        if x_cont is not None:
            x = torch.cat((x,x_cont),dim=1)
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x
    

#mlp
class MLP(nn.Module):
    def __init__(self, dims, act = None):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims) - 1)
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)

            if is_last:
                continue
            if act is not None:
                layers.append(act)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class simple_MLP(nn.Module):
    def __init__(self,dims):
        super(simple_MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2])
        )
        
    def forward(self, x):
        if len(x.shape)==1:
            x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x

# main class

class TabAttention(nn.Module):
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head = 16,
        dim_out = 1,
        mlp_hidden_mults = (4, 2),
        mlp_act = None,
        num_special_tokens = 1,
        continuous_mean_std = None,
        attn_dropout = 0.,
        ff_dropout = 0.,
        lastmlp_dropout = 0.,
        cont_embeddings = 'MLP',
        scalingfactor = 10,
        attentiontype = 'colrow'
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'

        # categories related calculations
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        self.total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table
        categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
        categories_offset = categories_offset.cumsum(dim = -1)[:-1]
        
        self.register_buffer('categories_offset', categories_offset)

        self.norm = nn.LayerNorm(num_continuous)
        self.num_continuous = num_continuous
        self.dim = dim
        self.cont_embeddings = cont_embeddings
        self.attentiontype = attentiontype

        if self.cont_embeddings == 'MLP':
            self.simple_MLP = nn.ModuleList([simple_MLP([1,100,self.dim]) for _ in range(self.num_continuous)])
            input_size = (dim * self.num_categories)  + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous
        else:
            print('Continous features are not passed through attention')
            input_size = (dim * self.num_categories) + num_continuous
            nfeats = self.num_categories 

        # transformer
        if attentiontype == 'col':
            self.transformer = Transformer(
                num_tokens = self.total_tokens,
                dim = dim,
                depth = depth,
                heads = heads,
                dim_head = dim_head,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout
            )
        elif attentiontype in ['row','colrow'] :
            self.transformer = RowColTransformer(
                num_tokens = self.total_tokens,
                dim = dim,
                nfeats= nfeats,
                depth = depth,
                heads = heads,
                dim_head = dim_head,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout,
                style = attentiontype
            )

        l = input_size // 8
        hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
        all_dimensions = [input_size, *hidden_dimensions, dim_out]
        
        self.mlp = MLP(all_dimensions, act = mlp_act)
        self.embeds = nn.Embedding(self.total_tokens, self.dim) #.to(device)

        cat_mask_offset = F.pad(torch.Tensor(self.num_categories).fill_(2).type(torch.int8), (1, 0), value = 0) 
        cat_mask_offset = cat_mask_offset.cumsum(dim = -1)[:-1]

        con_mask_offset = F.pad(torch.Tensor(self.num_continuous).fill_(2).type(torch.int8), (1, 0), value = 0) 
        con_mask_offset = con_mask_offset.cumsum(dim = -1)[:-1]

        self.register_buffer('cat_mask_offset', cat_mask_offset)
        self.register_buffer('con_mask_offset', con_mask_offset)

        self.mask_embeds_cat = nn.Embedding(self.num_categories*2, self.dim)
        self.mask_embeds_cont = nn.Embedding(self.num_continuous*2, self.dim)

    def forward(self, x_categ, x_cont,x_categ_enc,x_cont_enc):
        device = x_categ.device
        if self.attentiontype == 'justmlp':
            if x_categ.shape[-1] > 0:
                flat_categ = x_categ.flatten(1).to(device)
                x = torch.cat((flat_categ, x_cont.flatten(1).to(device)), dim = -1)
            else:
                x = x_cont.clone()
        else:
            if self.cont_embeddings == 'MLP':
                x = self.transformer(x_categ_enc,x_cont_enc.to(device))
            else:
                if x_categ.shape[-1] <= 0:
                    x = x_cont.clone()
                else: 
                    flat_categ = self.transformer(x_categ_enc).flatten(1)
                    x = torch.cat((flat_categ, x_cont), dim = -1)                    
        flat_x = x.flatten(1)
        return self.mlp(flat_x)


class sep_MLP(nn.Module):
    def __init__(self,dim,len_feats,categories):
        super(sep_MLP, self).__init__()
        self.len_feats = len_feats
        self.layers = nn.ModuleList([])
        for i in range(len_feats):
            self.layers.append(simple_MLP([dim,5*dim, categories[i]]))

        
    def forward(self, x):
        y_pred = list([])
        for i in range(self.len_feats):
            x_i = x[:,i,:]
            pred = self.layers[i](x_i)
            y_pred.append(pred)
        return y_pred

class SAINT(nn.Module):
    def __init__(
        self,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head = 16,
        dim_out = 1,
        mlp_hidden_mults = (4, 2),
        mlp_act = None,
        num_special_tokens = 1,
        attn_dropout = 0.,
        ff_dropout = 0.,
        cont_embeddings = 'MLP',
        scalingfactor = 10,
        attentiontype = 'col',
        final_mlp_style = 'common',
        y_dim = 2
        ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'

        # categories related calculations

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        self.total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table
        categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
        categories_offset = categories_offset.cumsum(dim = -1)[:-1]
        
        self.register_buffer('categories_offset', categories_offset)
        
        self.norm = nn.LayerNorm(num_continuous)
        self.num_continuous = num_continuous
        self.dim = dim
        self.cont_embeddings = cont_embeddings
        self.attentiontype = attentiontype
        self.final_mlp_style = final_mlp_style

        if self.cont_embeddings == 'MLP':
            self.simple_MLP = nn.ModuleList([simple_MLP([1,100,self.dim]) for _ in range(self.num_continuous)])
            input_size = (dim * self.num_categories)  + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous
        elif self.cont_embeddings == 'pos_singleMLP':
            self.simple_MLP = nn.ModuleList([simple_MLP([1,100,self.dim]) for _ in range(1)])
            input_size = (dim * self.num_categories)  + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous
        else:
            print('Continous features are not passed through attention')
            input_size = (dim * self.num_categories) + num_continuous
            nfeats = self.num_categories 

        # transformer
        if attentiontype == 'col':
            self.transformer = Transformer(
                num_tokens = self.total_tokens,
                dim = dim,
                depth = depth,
                heads = heads,
                dim_head = dim_head,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout
            )
        elif attentiontype in ['row','colrow'] :
            self.transformer = RowColTransformer(
                num_tokens = self.total_tokens,
                dim = dim,
                nfeats= nfeats,
                depth = depth,
                heads = heads,
                dim_head = dim_head,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout,
                style = attentiontype
            )

        l = input_size // 8
        hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
        all_dimensions = [input_size, *hidden_dimensions, dim_out]
        
        self.mlp = MLP(all_dimensions, act = mlp_act)
        self.embeds = nn.Embedding(self.total_tokens, self.dim) #.to(device)

        cat_mask_offset = F.pad(torch.Tensor(self.num_categories).fill_(2).type(torch.int8), (1, 0), value = 0) 
        cat_mask_offset = cat_mask_offset.cumsum(dim = -1)[:-1]

        con_mask_offset = F.pad(torch.Tensor(self.num_continuous).fill_(2).type(torch.int8), (1, 0), value = 0) 
        con_mask_offset = con_mask_offset.cumsum(dim = -1)[:-1]

        self.register_buffer('cat_mask_offset', cat_mask_offset)
        self.register_buffer('con_mask_offset', con_mask_offset)

        self.mask_embeds_cat = nn.Embedding(self.num_categories*2, self.dim)
        self.mask_embeds_cont = nn.Embedding(self.num_continuous*2, self.dim)
        self.single_mask = nn.Embedding(2, self.dim)
        self.pos_encodings = nn.Embedding(self.num_categories+ self.num_continuous, self.dim)
        
        if self.final_mlp_style == 'common':
            self.mlp1 = simple_MLP([dim,(self.total_tokens)*2, self.total_tokens])
            self.mlp2 = simple_MLP([dim ,(self.num_continuous), 1])

        else:
            self.mlp1 = sep_MLP(dim,self.num_categories,categories)
            self.mlp2 = sep_MLP(dim,self.num_continuous,np.ones(self.num_continuous).astype(int))


        self.mlpfory = simple_MLP([dim ,1000, y_dim])
        self.pt_mlp = simple_MLP([dim*(self.num_continuous+self.num_categories) ,6*dim*(self.num_continuous+self.num_categories)//5, dim*(self.num_continuous+self.num_categories)//2])
        self.pt_mlp2 = simple_MLP([dim*(self.num_continuous+self.num_categories) ,6*dim*(self.num_continuous+self.num_categories)//5, dim*(self.num_continuous+self.num_categories)//2])

        
    def forward(self, x_categ, x_cont):
        
        x = self.transformer(x_categ, x_cont)
        cat_outs = self.mlp1(x[:,:self.num_categories,:])
        con_outs = self.mlp2(x[:,self.num_categories:,:])
        return cat_outs, con_outs 
    
### added
from libs.models.supervised import supmodel

class build_saint(SAINT):
    def __init__(self, 
                 categories, num_continuous, 
                 dim=32, depth=6, heads=8, 
                 attn_dropout = 0.1, ff_dropout = 0.1, 
                 cont_embeddings = 'MLP', attentiontype = 'colrow', optimizer="AdamW", 
                 learning_rate=0.0001, weight_decay=5e-4, dim_out = 1, 
                 mlp_hidden_mults = (4, 2), mlp_act = None,
                 scalingfactor = 10,
                 final_mlp_style = 'sep', 
                 y_dim = 2, 
                 dim_head = 16, num_special_tokens = 0,
                 ):
            
        categories = torch.cat((torch.tensor([1]), categories))
        super().__init__(categories, num_continuous, dim, depth, heads)
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model = SAINT(categories, num_continuous, dim, depth, heads, dim_head, dim_out, 
                           mlp_hidden_mults, mlp_act,
                           num_special_tokens, attn_dropout, ff_dropout, cont_embeddings, scalingfactor,
                           attentiontype, final_mlp_style, y_dim)
    
    def forward(self, x_categ, x_cont):
        return self.model(x_categ, x_cont)

        
### augmentations
def embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model):
    device = x_cont.device
    if x_categ.size(1) > 1:
        x_categ = x_categ + model.categories_offset.type_as(x_categ)
    x_categ_enc = model.embeds(x_categ)
    n1,n2 = x_cont.shape
    _, n3 = x_categ.shape
    
    if model.cont_embeddings == 'MLP':
        x_cont_enc = torch.empty(n1,n2, model.dim)
        for i in range(model.num_continuous):
            x_cont_enc[:,i,:] = model.simple_MLP[i](x_cont[:,i])
    else:
        raise Exception('This case should not work!')    

    x_cont_enc = x_cont_enc.to(device)
    cat_mask_temp = cat_mask + model.cat_mask_offset.type_as(cat_mask)
    con_mask_temp = con_mask + model.con_mask_offset.type_as(con_mask)

    cat_mask_temp = model.mask_embeds_cat(cat_mask_temp)
    con_mask_temp = model.mask_embeds_cont(con_mask_temp)
    x_categ_enc[cat_mask == 0] = cat_mask_temp[cat_mask == 0]
    x_cont_enc[con_mask == 0] = con_mask_temp[con_mask == 0]

    return x_categ, x_categ_enc, x_cont_enc



def mixup_data(x1, x2 , lam=1.0, y=None, use_cuda=True):
    '''Returns mixed inputs, pairs of targets'''

    batch_size = x1.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x1 = lam * x1 + (1 - lam) * x1[index, :]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index, :]
    if y is not None:
        y_a, y_b = y, y[index]
        return mixed_x1, mixed_x2, y_a, y_b
    
    return mixed_x1, mixed_x2


def add_noise(x_categ, x_cont, noise_params = {'noise_type': ['cutmix'], 'lambda': 0.1}):
    lam = noise_params['lambda']
    device = x_categ.device
    batch_size = x_categ.size()[0]

    if 'cutmix' in noise_params['noise_type']:
        index = torch.randperm(batch_size)
        cat_corr = torch.from_numpy(np.random.choice(2,(x_categ.shape),p=[lam,1-lam])).to(device)
        con_corr = torch.from_numpy(np.random.choice(2,(x_cont.shape),p=[lam,1-lam])).to(device)
        x1, x2 =  x_categ[index,:], x_cont[index,:]
        x_categ_corr, x_cont_corr = x_categ.clone().detach() ,x_cont.clone().detach()
        x_categ_corr[cat_corr==0] = x1[cat_corr==0]
        x_cont_corr[con_corr==0] = x2[con_corr==0]
        return x_categ_corr, x_cont_corr
    elif noise_params['noise_type'] == 'missing':
        x_categ_mask = np.random.choice(2,(x_categ.shape),p=[lam,1-lam])
        x_cont_mask = np.random.choice(2,(x_cont.shape),p=[lam,1-lam])
        x_categ_mask = torch.from_numpy(x_categ_mask).to(device)
        x_cont_mask = torch.from_numpy(x_cont_mask).to(device)
        return torch.mul(x_categ,x_categ_mask), torch.mul(x_cont,x_cont_mask)
        
    else:
        print("yet to write this")


        
class main_saint(supmodel):
    def __init__(self, params, tasktype, device='cuda', data_id=None, 
                 modelname="saint", cat_features=[]):
        
        super().__init__(params, tasktype, device, data_id, modelname)
        
        self.model = build_saint(
            categories=params["categories"], num_continuous=len(params["num_features"]), 
            y_dim=params["output_dim"], dim=params["dim"]) 
        
        self.model = self.model.to(device)
        
        self.eval_lineareval = build_mlp(self.tasktype, params["dim"], params.get("output_dim", None), 
                                         1, params["dim"], 0.1, "batchnorm", "relu", "Adam", 0.01, 0.)
        self.eval_lineareval.to(self.device)

        self.eval_finetuning = build_mlp(self.tasktype, params["dim"], params.get("output_dim", None), 
                                         1, 32, 0.1, "batchnorm", "relu", "Adam", 0.001, 0.00001)
        self.eval_finetuning.to(self.device)
        self.eval_lr = LogisticRegression()            
        self.eval_knn = KNeighborsClassifier(n_neighbors=params["k"])
    
    def fit(self, X_train, y_train):
        
        batch_size = 256
        ssl_optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0001)
        
        ft_optimizer = torch.optim.AdamW(
            chain(self.model.parameters(), self.eval_finetuning.parameters()), 
            lr=self.params['ft_learning_rate'], weight_decay=self.params['ft_weight_decay']
        )
        le_optimizer = torch.optim.AdamW(
            self.eval_lineareval.parameters(), 
            lr=self.params['le_learning_rate'], weight_decay=self.params['le_weight_decay']
        )
        
        labeled_flag = torch.unique(torch.where(~torch.isnan(y_train))[0])
        label_X_train = X_train[labeled_flag]
        label_y_train = y_train[labeled_flag]
        
        if self.params.get("cat_features") is None:
            X_train_cat = torch.empty((X_train.size(0), 0), dtype=int, device=X_train.device)
            label_X_train_cat = torch.empty((label_X_train.size(0), 0), dtype=int, device=label_X_train.device)
        else:
            X_train_cat = X_train[:, self.params.get("cat_features")].type(torch.long)
            label_X_train_cat = label_X_train[:, self.params.get("cat_features")].type(torch.long)
        X_train_num = X_train[:, self.params.get("num_features")]
        label_X_train_num = label_X_train[:, self.params.get("num_features")]
        
        cls_token1 = torch.zeros((X_train.size(0), 1), dtype=torch.int, device=X_train.device)
        cls_token2 = torch.zeros((label_X_train.size(0), 1), dtype=torch.int, device=label_X_train.device)
        
        X_train_cat = torch.cat((cls_token1, X_train_cat), dim=1)
        label_X_train_cat = torch.cat((cls_token2, label_X_train_cat), dim=1)
        
        train_dataset = torch.utils.data.TensorDataset(X_train_cat, X_train_num, y_train)
        ft_dataset = torch.utils.data.TensorDataset(label_X_train_cat, label_X_train_num, label_y_train)
        ft_batch_size = 100
        del X_train, y_train
        
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        ft_loader = torch.utils.data.DataLoader(dataset=ft_dataset, batch_size=ft_batch_size, shuffle=True, drop_last=True)
        
        ssl_optimizer.zero_grad(); ssl_optimizer.step()
        
        ssl_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(ssl_optimizer, self.params.get('ssl_epochs', 0))
        if self.params.get("le_lr_scheduler", False) & (len(ft_loader) > 0):
            le_scheduler = CosineAnnealingLR_Warmup(le_optimizer, base_lr=self.params['le_learning_rate'], warmup_epochs=0, 
                                                    T_max=self.params.get('le_epochs'), iter_per_epoch=len(ft_loader), 
                                                    warmup_lr=1e-6, eta_min=0, last_epoch=-1)
        if self.params.get("ft_lr_scheduler", False) & (len(ft_loader) > 0):
            ft_scheduler = CosineAnnealingLR_Warmup(ft_optimizer, base_lr=self.params['ft_learning_rate'], warmup_epochs=0, 
                                                    T_max=self.params.get('ft_epochs'), iter_per_epoch=len(ft_loader), 
                                                    warmup_lr=1e-6, eta_min=0, last_epoch=-1)
        
        ## ssl first
        criterion1 = torch.nn.CrossEntropyLoss()
        criterion2 = torch.nn.MSELoss()
        
        pbar = tqdm(range(1, 51))
        for epoch in pbar:
            pbar.set_description("EPOCH: %i" %epoch)
            
            for i, (x_categ, x_cont, y) in enumerate(train_loader):                
                
#                 if self.params.get("cat_features") is None:
#                     x_categ = torch.empty((x.size(0), 0), dtype=int, device=x.device)
#                 else:
#                     x_categ = x[:, self.params.get("cat_features")].type(torch.long)
#                 x_cont = x[:, self.params.get("num_features")]
                
                self.model.train(); ssl_optimizer.zero_grad()
#                 cls_token = torch.zeros((x.size(0), 1), dtype=torch.int, device=x.device)
                cls_mask = torch.ones((y.size(0), 1), dtype=torch.int, device=y.device)
#                 x_categ = torch.cat((cls_token, x_categ), dim=1)
                
                x_categ_corr, x_cont_corr = add_noise(x_categ, x_cont, noise_params={'noise_type' : ["mixup", "cutmix"], 'lambda' : 0.1})
                
                cat_mask = torch.ones_like(x_categ, dtype=torch.int)
                con_mask = torch.ones_like(x_cont_corr, dtype=torch.int)
                
                _ , x_categ_enc_2, x_cont_enc_2 = embed_data_mask(
                    x_categ_corr, x_cont_corr, cat_mask, con_mask, self.model)
                _ , x_categ_enc, x_cont_enc = embed_data_mask(
                    x_categ.to(torch.long), x_cont, cat_mask, con_mask, self.model)
                
                x_categ_enc_2, x_cont_enc_2 = mixup_data(x_categ_enc_2, x_cont_enc_2 , lam=0.3)
                
                loss = 0
                
                ### contrastive
                aug_features_1  = self.model.transformer(x_categ_enc, x_cont_enc)
                aug_features_2 = self.model.transformer(x_categ_enc_2, x_cont_enc_2)
                aug_features_1 = (aug_features_1 / aug_features_1.norm(dim=-1, keepdim=True)).flatten(1,2)
                aug_features_2 = (aug_features_2 / aug_features_2.norm(dim=-1, keepdim=True)).flatten(1,2)
                #projhead = diff (Appendix E1)
                aug_features_1 = self.model.pt_mlp(aug_features_1)
                aug_features_2 = self.model.pt_mlp2(aug_features_2)
                
                logits_per_aug1 = aug_features_1 @ aug_features_2.t()/0.7
                logits_per_aug2 =  aug_features_2 @ aug_features_1.t()/0.7
                targets = torch.arange(logits_per_aug1.size(0)).to(logits_per_aug1.device)
                loss_1 = criterion1(logits_per_aug1, targets)
                loss_2 = criterion1(logits_per_aug2, targets)
                loss   = 0.5*(loss_1 + loss_2)/2
                
                ### denoising
                cat_outs, con_outs = self.model(x_categ_enc_2, x_cont_enc_2)
                if len(con_outs) > 0:
                    con_outs =  torch.cat(con_outs, dim=1)
                    l2 = criterion2(con_outs, x_cont)
                else:
                    l2 = 0
                l1 = 0
                # import ipdb; ipdb.set_trace()
                n_cat = x_categ.shape[-1]
                for j in range(1,n_cat):
                    l1+= criterion1(cat_outs[j],x_categ[:,j])
                loss += 1*l1 + 10*l2    
                
                loss.backward(); ssl_optimizer.step(); ssl_optimizer.zero_grad(); ssl_scheduler.step()
                pbar.set_postfix_str(f'data_id: {self.data_id}, Model: {self.modelname}, Tr loss: {loss:.5f}')
        
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
                cls_mask = torch.ones((label_X_train.size(0), 1), 
                                      dtype=torch.int, device=label_X_train.device)

                cat_mask = torch.ones_like(label_X_train_cat, dtype=torch.int)
                con_mask = torch.ones_like(label_X_train_num, dtype=torch.int)

                _ , x_categ_enc, x_cont_enc = embed_data_mask(
                    label_X_train_cat.to(torch.long), label_X_train_num, cat_mask, 
                    con_mask, self.model)

                z = self.model.transformer(x_categ_enc, x_cont_enc)[:, 0, :]

                label_y_train = label_y_train.cpu().numpy()
                if self.tasktype == "multiclass":
                    label_y_train = np.argmax(label_y_train, axis=1)
                self.eval_lr.fit(z.cpu().numpy(), label_y_train)
                self.eval_knn.fit(z.cpu().numpy(), label_y_train)
        
        ## linear eval
        print("Linear evaluation")
        for epoch in tqdm(range(1, self.params.get('le_epochs', 0) + 1)):
            pbar.set_description("Linear eval. EPOCH: %i" %epoch)
            for i, (x_categ, x_cont, y) in enumerate(ft_loader):
                self.model.eval(); self.eval_lineareval.train(); le_optimizer.zero_grad()
                with torch.no_grad():
#                     if self.params.get("cat_features") is None:
#                         x_categ = torch.empty((x.size(0), 0), dtype=int, device=label_X_train.device)
#                     else:
#                         x_categ = label_X_train[:, self.params.get("cat_features")]
#                     x_cont = label_X_train[:, self.params.get("num_features")]
#                     cls_token = torch.zeros((x.size(0), 1), dtype=torch.int, device=x.device)
                    cls_mask = torch.ones((y.size(0), 1), dtype=torch.int, device=y.device)
#                     x_categ = torch.cat((cls_token, x_categ), dim=1)

                    cat_mask = torch.ones_like(x_categ, dtype=torch.int)
                    con_mask = torch.ones_like(x_cont, dtype=torch.int)

                    _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ.to(torch.long), x_cont, cat_mask, con_mask, self.model)  

                    z = self.model.transformer(x_categ_enc, x_cont_enc)[:, 0, :]
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

            for i, (x_categ, x_cont, y) in enumerate(ft_loader):
                self.model.train(); self.eval_finetuning.train(); ft_optimizer.zero_grad()
#                 if self.params.get("cat_features") is None:
#                     x_categ = torch.empty((x.size(0), 0), dtype=int, device=label_X_train.device)
#                 else:
#                     x_categ = label_X_train[:, self.params.get("cat_features")]
#                 x_cont = label_X_train[:, self.params.get("num_features")]
#                 cls_token = torch.zeros((x.size(0), 1), dtype=torch.int, device=x.device)
                cls_mask = torch.ones((y.size(0), 1), dtype=torch.int, device=y.device)
#                 x_categ = torch.cat((cls_token, x_categ), dim=1)
                
                cat_mask = torch.ones_like(x_categ, dtype=torch.int)
                con_mask = torch.ones_like(x_cont, dtype=torch.int)

                _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ.to(torch.long), x_cont, cat_mask, con_mask, self.model)  

                z = self.model.transformer(x_categ_enc, x_cont_enc)[:, 0, :]
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
            if self.params.get("cat_features") is None:
                x_categ = torch.empty((X_test.size(0), 0), dtype=int, device=X_test.device)
            else:
                x_categ = X_test[:, self.params.get("cat_features")]
            x_cont = X_test[:, self.params.get("num_features")]
            cls_token = torch.zeros((X_test.size(0), 1), dtype=torch.int, device=X_test.device)
            cls_mask = torch.ones((X_test.size(0), 1), dtype=torch.int, device=X_test.device)
            x_categ = torch.cat((cls_token, x_categ), dim=1)
            
            cat_mask = torch.ones_like(x_categ, dtype=torch.int)
            con_mask = torch.ones_like(x_cont, dtype=torch.int)
            
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ.to(torch.long), x_cont, cat_mask, con_mask, self.model)  
            
            z = self.model.transformer(x_categ_enc, x_cont_enc)[:, 0, :]
            
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
            if self.params.get("cat_features") is None:
                x_categ = torch.empty((X_test.size(0), 0), dtype=int, device=X_test.device)
            else:
                x_categ = X_test[:, self.params.get("cat_features")]
            x_cont = X_test[:, self.params.get("num_features")]
            cls_token = torch.zeros((X_test.size(0), 1), dtype=torch.int, device=X_test.device)
            cls_mask = torch.ones((X_test.size(0), 1), dtype=torch.int, device=X_test.device)
            x_categ = torch.cat((cls_token, x_categ), dim=1)
            
            cat_mask = torch.ones_like(x_categ, dtype=torch.int)
            con_mask = torch.ones_like(x_cont, dtype=torch.int)
            
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ.to(torch.long), x_cont, cat_mask, con_mask, self.model)  
            
            z = self.model.transformer(x_categ_enc, x_cont_enc)[:, 0, :]
                
            pred_lr = self.eval_lr.predict_proba(z.cpu().numpy())
            pred_knn = self.eval_knn.predict_proba(z.cpu().numpy())
            pred_le = self.eval_lineareval(z)
            pred_ft = self.eval_finetuning(z)

            if logit or (self.tasktype == "regression"):
                return pred_lr, pred_knn, pred_le.cpu().numpy(), pred_ft.cpu().numpy()
            else:
                return pred_lr, pred_knn, torch.nn.functional.softmax(pred_le).cpu().numpy(), torch.nn.functional.softmax(pred_ft).cpu().numpy() 