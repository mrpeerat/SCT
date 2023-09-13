import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
import torch.distributed as dist
import torch.nn.functional as F

def batch_all_gather(x):
    x_list = FullGatherLayer.apply(x)
    return torch.cat(x_list, dim=0)

class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]

class MLP(nn.Module):
	def __init__(self, dim, projection_size, hidden_size):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(dim, hidden_size),
			# nn.BatchNorm1d(hidden_size),
			nn.ReLU(),
# 			nn.Linear(hidden_size, hidden_size),
# 			nn.ReLU(),
			nn.Linear(hidden_size, projection_size)
		)

	def forward(self, x):
		return self.net(x)

class VICRegLoss(nn.Module):
    
    def __init__(self, model, batch_size, embedding_size, sim_coeff, std_coeff, cov_coeff, mlp_args="8192-8192-8192"):
        """
        :param model: SentenceTransformerModel
        """
        super(VICRegLoss, self).__init__()
        self.model = model
        self.projector = Projector(mlp_args,embedding_size)
        self.batch_size = batch_size
        self.num_features = int(mlp_args.split("-")[-1])
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.online_predictor_1 = MLP(embedding_size, embedding_size, 10 * embedding_size)
        self.online_predictor_2 = MLP(embedding_size, embedding_size, 10 * embedding_size)
        self.online_predictor_3 = MLP(embedding_size, embedding_size, 10 * embedding_size)


    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        x = self.online_predictor_1(reps[0])
        x = self.online_predictor_2(x)
        x = self.online_predictor_3(x)
        
        y = self.online_predictor_1(torch.cat(reps[1:]))
        y = self.online_predictor_2(y)
        y = self.online_predictor_3(y)

        repr_loss = F.mse_loss(x, y)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.batch_size - 1)
        cov_y = (y.T @ y) / (self.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)
        

        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )
        return loss

def Projector(args, embedding):
    mlp_spec = f"{embedding}-{args}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()