import torch
import torch.nn as nn
import torch.nn.functional as F
from data import datagen
import numpy as np
import matplotlib.pyplot as plt

class Highway(torch.nn.Module):
    """
    A `Highway layer <https://arxiv.org/abs/1505.00387>`_.
    Adopted from the AllenNLP implementation.
    """
    def __init__(self, input_dim: int, num_layers: int = 1):
        super(Highway, self).__init__()
        self.input_dim = input_dim
        self.layers = nn.ModuleList(
            [nn.Linear(input_dim, input_dim * 2) for _ in range(num_layers)]
        )
        self.activation = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            # As per comment in AllenNLP:
            # We should bias the highway layer to just carry its input forward. We do that by
            # setting the bias on `B(x)` to be positive, because that means `g` will be biased to
            # be high, so we will carry the input forward. The bias on `B(x)` is the second half
            # of the bias vector in each Linear layer.
            nn.init.constant_(layer.bias[self.input_dim:], 1)

            nn.init.constant_(layer.bias[: self.input_dim], 0)
            nn.init.xavier_normal_(layer.weight)

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            projection = layer(x)
            proj_x, gate = projection.chunk(2, dim=-1)
            proj_x = self.activation(proj_x)
            gate = torch.sigmoid(gate)
            x = gate * x + (gate.new_tensor([1]) - gate) * proj_x
            return x


class ByteCombineCNN(nn.Module):
    def __init__(self, embed_dim, output_dim, activation_fn='relu',
        filters=[(1, 64), (2, 128), (3, 192)] + [(i, 256) for i in range(4, 8)],
        highway_layers=2):

        # Pytorch will search for the most efficient convolution implementation
        torch.backends.cudnn.benchmark = True

        # TODO: increase filters once the byte fields went to 8
        super().__init__()

        #self.activation_fn = utils.get_activation_fn(activation_fn)

        self.convolutions = nn.ModuleList()
        for width, out_c in filters:
            self.convolutions.append(
                nn.Conv1d(embed_dim, out_c, kernel_size=width)
            )

        last_dim = sum(f[1] for f in filters)
        self.highway = Highway(last_dim, highway_layers) if highway_layers > 0 else None

        self.projection = nn.Linear(last_dim, output_dim)

    def forward(self, features):
        # features size: Batch x Seq x byte_len x Emb_dim
        B = features.size(0)
        T = features.size(1)
        byte_len = features.size(2)
        emb_dim = features.size(3)
        
        # BTC -> BCT, BTC: batch, sequence, embedding size
        features = features.transpose(2, 3).view(-1, emb_dim, byte_len)

        conv_result = []

        for conv in self.convolutions:
            x = conv(features)
            x, _ = torch.max(x, -1)
            x = F.relu(x)
            conv_result.append(x)

        x = torch.cat(conv_result, dim=-1)
        if self.highway is not None:
            x = self.highway(x)
            x = self.projection(x)
            x = x.view(B, T, -1)

        return x

class Model(nn.Module):
    def __init__(self, embed_dim, dim1, dim2, dim3, dim4, dim5):
        super().__init__()

        self.byte_emb = nn.Embedding(256, embed_dim)
        self.op_emb = nn.Embedding(4, 256)
        self.byte_combine = ByteCombineCNN(embed_dim, dim1)
        self.lin1 = nn.Linear(dim1*2+256, dim2)
        self.lin2 = nn.Linear(dim2, dim3)
        self.lin3 = nn.Linear(dim3, dim4)
        self.lin4 = nn.Linear(dim4, dim5)
        #self.softmax = nn.LogSoftmax(dim=1)
        self.act = nn.ReLU()

    def forward(self, tokens):
        # [batch_size, seq=2, byte_len=8] -> [batch_size, seq=2, byte_len=8, byte_embed_dim]
        embedded_bytes = self.byte_emb(tokens[0])

        # [batch_size, 1] -> [batch_size, op_embed_dim]
        embedded_op = self.op_emb(tokens[1]).squeeze(1)
        
        # [batch_size, seq, byte_len, byte_embed_dim] -> [batch_size, seq, dim1]
        x = self.byte_combine(embedded_bytes)
        
        # [batch_size, seq, dim1] -> [batch_size, seq*dim1]
        x = x.flatten(start_dim=1, end_dim=2)
        
        # [batch_size, seq*dim1] -> [batch_size, seq*dim1+op_embed_dim]
        x = torch.cat([x, embedded_op], dim=-1)
        
        # [batch_size, seq*dim1+op_embed_dim] -> [batch_size, dim2]
        x = self.lin1(x)
        x = self.act(x)
        
        # [batch_size, dim2] -> [batch_size, dim3]
        x = self.lin2(x)
        x = self.act(x)
        
        # [batch_size, dim3] -> [batch_size, 256*8]
        x = self.lin3(x)

        return x

#x = F.dropout(x)
#x = self.act(x)
#x = self.lin4(x)
#x = self.act(x)
#x = torch.cat([self.softmax(x) for _ in range(8)], dim=-1)
#print(torch.max(x))
#print(x.shape)

LR=1e-3
EPOCH=10000
BATCH_SIZE=128
MINI_BATCH=64

EMBED_DIM=512
DIM1=512
DIM2=512
DIM3=512
DIM4=2048
DIM5=512

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on GPU", torch.cuda.get_device_name(), torch.cuda.device_count())

model = Model(EMBED_DIM, DIM1, DIM2, DIM3, DIM4, DIM5).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def train(epochs):
    losses = []
    vlosses = []
    for epoch in range(epochs):
        mini_batch_loss = []
        bytes_train, op_train, t_train = datagen(BATCH_SIZE, device)

        for mini in range(MINI_BATCH):
            optimizer.zero_grad()
            
            y_train = model((bytes_train, op_train))
            loss = criterion(y_train, t_train)
            mini_batch_loss.append(loss)
            loss.backward()
            optimizer.step()
        
        bytes_valid, op_valid, t_valid = datagen(BATCH_SIZE, device)
        y_valid = model((bytes_valid, op_valid))
        loss_valid = criterion(y_valid, t_valid)
        
        losses.append((sum(mini_batch_loss) / len(mini_batch_loss)).item())
        vlosses.append(loss_valid)

        if(epoch % 1 == 0):
            print('train %d: %.10f\tvalid: %.10f' %(epoch, losses[len(losses)-1], loss_valid.item()))
    return losses

loss_list = train(EPOCH)

t = np.arange(len(loss_list))
loss_arr = np.array(loss_list)
plt.clf()
plt.plot(t, loss_arr)
plt.savefig('loss.png')
