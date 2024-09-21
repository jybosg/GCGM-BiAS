import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GATv2Conv, GINConv, global_mean_pool, global_max_pool, global_add_pool
import math

# reference: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SAGEConv

ACTIVATION = {'elu': nn.ELU(), 'relu': nn.ReLU(), 'prelu': nn.PReLU(), 'leaky_relu': nn.LeakyReLU()}

class GIN(nn.Module):
    def __init__(self, in_channels, out_channels, activation):
        super(GIN, self).__init__()

        self.activation = ACTIVATION[activation]
        self.nn = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            self.activation,
            nn.Linear(out_channels, out_channels)
        )
        self.conv = GINConv(self.nn)
        
        self.activation_ = activation
        
        self.init_weights()

    def init_weights(self):
        if self.activation_ == 'elu':
            alpha = 1
        elif self.activation_ == 'prelu':
            alpha = 0.25
        for m in self.nn.modules():
            if isinstance(m, nn.Linear):
                if self.activation_ == 'elu' or self.activation_ == 'prelu':
                    nn.init.kaiming_uniform_(m.weight, a=alpha, mode='fan_in', nonlinearity='leaky_relu')
                else:
                    nn.init.kaiming_uniform_(m.weight, nonlinearity=self.activation_)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return x
    
BASE_MODELS = {'GCNConv': GCNConv, 'SAGEConv': SAGEConv, 'GATConv': GATConv, 'GATv2Conv': GATv2Conv, 'GINConv': GIN}

class Encoder(nn.Module):
    def __init__(self, in_channels: int, inter_channels:int, out_channels: int, activation,
                 base_model='SAGEConv', aggr='mean', dropout=0.2, skip:bool = True, project:bool = False, k: int = 3, 
                 global_readout='mean', structure='Pre'):
        super(Encoder, self).__init__()
        self.base_model = BASE_MODELS[base_model]
        self.base_model_ = base_model
        # whether to add skip connection
        self.skip = skip    
        # whether to project the output of each GCN layer
        self.project = project    
        # activation function
        self.activation = ACTIVATION[activation]
        self.activation_ = activation
        self.structure = structure

        # project to lower dimension space
        if inter_channels == in_channels:
            self.fc1 = None
        else:
            self.fc1 = nn.Linear(in_channels, inter_channels)
        self.in_channels = inter_channels
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        
        # GCN layers
        # assert k >= 2
        self.k = k
        self.aggr = aggr
        self.dropout = dropout
        self.global_readout = global_readout
        
        if base_model == 'GCNConv':
            self.conv = [self.base_model(self.in_channels, self.inter_channels)]
            for _ in range(1, self.k):
                self.conv.append(self.base_model(self.inter_channels, self.inter_channels))
        elif base_model == 'SAGEConv':
            self.conv = [self.base_model(self.in_channels, self.inter_channels, self.aggr)]
            for _ in range(1, self.k):
                self.conv.append(self.base_model(self.inter_channels, self.inter_channels, self.aggr))
        elif base_model in ['GATConv', 'GATv2Conv']:
            self.conv = [self.base_model(self.in_channels, self.inter_channels, dropout=self.dropout)]
            for _ in range(1, self.k):
                self.conv.append(self.base_model(self.inter_channels, self.inter_channels, dropout=self.dropout))
        elif base_model == 'GINConv':
            self.conv = [self.base_model(self.in_channels, self.inter_channels, self.activation_)]
            for _ in range(1, self.k):
                self.conv.append(self.base_model(self.inter_channels, self.inter_channels, self.activation_))
        self.conv = nn.ModuleList(self.conv)
        
        if structure == 'Pre':
            # project concatenation of all GCN layers outputs as the final output of the node features
            self.fc2 = nn.Linear(self.k * self.inter_channels, 2 * self.out_channels)
            self.fc3 = nn.Linear(2 * self.out_channels, self.out_channels)
            # * for readout function
            self.fc4 = nn.Linear(self.k * self.inter_channels, 2 * self.out_channels)
            self.fc5 = nn.Linear(2 * self.out_channels, self.out_channels)
        else: # Cur
            # include the initial projection layer with the output of the last GCN layer
            self.fc2 = nn.Linear((1 + self.k) * self.inter_channels, 2 * self.out_channels)
            self.fc3 = nn.Linear(2 * self.out_channels, self.out_channels)
            # * for readout function
            self.fc4 = nn.Linear((1 + self.k) * self.inter_channels, 2 * self.out_channels)
            self.fc5 = nn.Linear(2 * self.out_channels, self.out_channels)
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        if self.activation_ == 'elu':
            alpha = 1
            gain = (2 / (1 + alpha**2))**0.5
        elif self.activation_ == 'prelu':
            initial_alpha = 0.25
            
        if self.base_model_ in ['GCNConv', 'GATConv', 'GATv2Conv']:
            for conv in self.conv:
                # He initialization
                if self.activation_ == 'elu':
                    # Initialize the weight tensor using Kaiming uniform initialization with the calculated gain for ELU
                    nn.init.kaiming_uniform_(conv.weight, a=gain, mode='fan_in', nonlinearity='leaky_relu')
                elif self.activation_ == 'prelu':
                    nn.init.kaiming_uniform_(conv.weight, a=initial_alpha, mode='fan_in', nonlinearity='leaky_relu')
                else:
                    nn.init.kaiming_uniform_(conv.weight, nonlinearity=self.activation_)
        elif self.base_model_ == 'SAGEConv':
            for conv in self.conv:
                # He initialization
                if self.activation_ == 'elu':
                    nn.init.kaiming_uniform_(conv.lin_l.weight, a=gain, mode='fan_in', nonlinearity='leaky_relu')
                    nn.init.kaiming_uniform_(conv.lin_r.weight, a=gain, mode='fan_in', nonlinearity='leaky_relu')
                elif self.activation_ == 'prelu':
                    nn.init.kaiming_uniform_(conv.lin_l.weight, a=initial_alpha, mode='fan_in', nonlinearity='leaky_relu')
                    nn.init.kaiming_uniform_(conv.lin_r.weight, a=initial_alpha, mode='fan_in', nonlinearity='leaky_relu')
                else:
                    nn.init.kaiming_uniform_(conv.lin_l.weight, nonlinearity=self.activation_)
                    nn.init.kaiming_uniform_(conv.lin_r.weight, nonlinearity=self.activation_)
            
        if self.fc1 is not None:
            if self.activation_ == 'elu':
                nn.init.kaiming_uniform_(self.fc1.weight, a=gain, mode='fan_in', nonlinearity='leaky_relu')
            elif self.activation_ == 'prelu':
                nn.init.kaiming_uniform_(self.fc1.weight, a=initial_alpha, mode='fan_in', nonlinearity='leaky_relu')
            else:
                nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity=self.activation_)
        
        if self.activation_ == 'elu':
            nn.init.kaiming_uniform_(self.fc2.weight, a=gain, mode='fan_in', nonlinearity='leaky_relu')
            nn.init.kaiming_uniform_(self.fc3.weight, a=gain, mode='fan_in', nonlinearity='leaky_relu')
            nn.init.kaiming_uniform_(self.fc4.weight, a=gain, mode='fan_in', nonlinearity='leaky_relu')
            nn.init.kaiming_uniform_(self.fc5.weight, a=gain, mode='fan_in', nonlinearity='leaky_relu')
        elif self.activation_ == 'prelu':
            nn.init.kaiming_uniform_(self.fc2.weight, a=initial_alpha, mode='fan_in', nonlinearity='leaky_relu')
            nn.init.kaiming_uniform_(self.fc3.weight, a=initial_alpha, mode='fan_in', nonlinearity='leaky_relu')
            nn.init.kaiming_uniform_(self.fc4.weight, a=initial_alpha, mode='fan_in', nonlinearity='leaky_relu')
            nn.init.kaiming_uniform_(self.fc5.weight, a=initial_alpha, mode='fan_in', nonlinearity='leaky_relu')
        else:
            nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity=self.activation_)
            nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity=self.activation_)
            nn.init.kaiming_uniform_(self.fc4.weight, nonlinearity=self.activation_)
            nn.init.kaiming_uniform_(self.fc5.weight, nonlinearity=self.activation_)
    
    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = self.activation(self.fc2(z))
        # * we could add activation function or leave it as it is
        return self.fc3(z)
    
    # * update the projection head for global mean pooling
    def readout(self, z: torch.Tensor, batch):
        if self.global_readout == 'mean':
            z = global_mean_pool(z, batch.batch)
        elif self.global_readout == 'max':
            z = global_max_pool(z, batch.batch)
        elif self.global_readout == 'add':
            z = global_add_pool(z, batch.batch)
        # z = F.relu(self.fc4(z))
        z = self.activation(self.fc4(z))
        return self.fc5(z)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch=None):
        # store residual before the linear transformation
        # residual = x
        if self.fc1:
            x = self.fc1(x)
        # store residual after the linear transformation
        # residual = x
        if self.structure == 'Pre':
            x_list = []
            for i in range(self.k):
                if self.skip and len(x_list) > 0:
                    # dimsion 0 could be the dimension of batch
                    x = self.activation(self.conv[i](x, edge_index) + torch.sum(torch.stack(x_list), dim=0))
                else:
                    x = self.activation(self.conv[i](x, edge_index))
                x_list.append(x)
        else: # Cur
            x_list = [x]  # * add initial projection to the input of each layer
            for i in range(self.k):
                if self.skip:
                    x = self.activation(self.conv[i](x, edge_index) + torch.sum(torch.stack(x_list), dim=0))
                x_list.append(x)

        # concatenate all GCN layers outputs
        x = torch.concat(x_list, dim=1)
        # project to lower dimension
        node_features = self.projection(x)
        # * add readout function for contrastive learning
        if batch is not None:
            # y = global_mean_pool(F.relu(self.fc4(x)), batch.batch)
            global_info = self.readout(x, batch)
            return node_features, global_info, x_list
        
        return node_features, x_list

# according to SimCLR paper, we could use nn.ReLU, and we should disable bias
class ProjectionHead(torch.nn.Module):
    def __init__(self, out_channels: int, projection_channels: int, bias=True, structure='GRACE'):
        super(ProjectionHead, self).__init__()
        self.structure = structure
        
        if structure == 'GRACE':
            self.model = torch.nn.Sequential(
                nn.Linear(out_channels, projection_channels, bias=bias),
                nn.ELU(),
                nn.Linear(projection_channels, out_channels, bias=bias)
                )
        else: # 'SimCLR'
            self.model = torch.nn.Sequential(
                nn.Linear(out_channels, out_channels, bias=bias),
                nn.ReLU(),
                nn.Linear(out_channels, projection_channels, bias=bias)
                )
        self.init_weights()
    
    def init_weights(self):
        
        if self.structure == 'GRACE':
            # ELU alpha hyperparameter
            alpha = 1
            # Calculate the gain for ELU
            gain = (2 / (1 + alpha**2))**0.5
            # Apply He initialization to the linear layers in the model
            for layer in self.model:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight, a=gain, mode='fan_in', nonlinearity='leaky_relu')
                    # Initialize the bias terms to zero
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        else: # 'SimCLR'
            # Apply He initialization to the linear layers in the model
            for layer in self.model:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                    # Initialize the bias terms to zero
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        
    def forward(self, x: torch.Tensor):
        return self.model(x)
    
class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            # assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        # * for query and key the output dimension is the same
        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h):
        
        if h is None:
            h = q  # compute self-attention

        # * graph_size is the size of the instance
        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)

        # * https://pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html
        # tensor.view returns a new tensor with the same data as the self tensor but of a different shape.
        hflat = h.contiguous().view(-1, input_dim) #################   reshape
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)   
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # * attention score
        attn = F.softmax(compatibility, dim=-1)
       
        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        # * concatenate the input query with the output as did in CGMN
        return torch.concat([q, out], dim=-1)