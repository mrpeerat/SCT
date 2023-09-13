import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import losses
from typing import Iterable, Dict
import copy

class EMA():
	def __init__(self, beta):
		super().__init__()
		self.beta = beta

	def update_average(self, old, new):
		if old is None:
			return new
		return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
	for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
		old_weight, up_weight = ma_params.data, current_params.data
		ma_params.data = ema_updater.update_average(old_weight, up_weight)


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


# MLP for  predictor
class MLP(nn.Module):
	def __init__(self, dim, projection_size, hidden_size):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(dim, hidden_size),
			nn.BatchNorm1d(hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, projection_size)
		)

	def forward(self, x):
		return self.net(x)


class BARLOWLoss_V2(nn.Module):
    def __init__(self, model, embed_size, batch_size, lambda_weight=0.0005, w1=1, w2=1, device=None, moving_average_decay=0.999,projector=False):
        """
        param model: SentenceTransformerModel
        K:          queue size
        t:          temperature for student encoder
        temp:       distillation temperature
        """
        super(BARLOWLoss_V2, self).__init__()

        self.model = model
        self.lambda_weight = lambda_weight
        self.w1 = w1
        self.w2 = w2
        self.device = device
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.projector = projector

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(embed_size, affine=False)

        self.target_encoder = copy.deepcopy(self.model) # momentum
        self.target_ema_updater = EMA(moving_average_decay) 

        self.online_predictor_1 = MLP(embed_size, embed_size, 10 * embed_size) 
        self.online_predictor_2 = MLP(embed_size, embed_size, 10 * embed_size) 
        self.online_predictor_3 = MLP(embed_size, embed_size, 10 * embed_size) 

        # Use multi GPU
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

    def update_moving_average(self):
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.model)

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        # raise Exception(len(reps),"----",len(reps[0]),'---',self.embed_size, '----', self.batch_size)
        non_en_question = reps[0]
        en_doc = reps[2]

        if self.projector:
            non_en_question, en_doc = self.online_predictor_1(non_en_question), self.online_predictor_1(en_doc)
            non_en_question, en_doc = self.online_predictor_2(non_en_question), self.online_predictor_2(en_doc)
            non_en_question, en_doc = self.online_predictor_3(non_en_question), self.online_predictor_3(en_doc)

        with torch.no_grad():
            reps = [self.target_encoder(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
            en_question_m = reps[1]
        

        c_q_q = self.bn(non_en_question).T @ self.bn(en_question_m.detach()) # question-to-question
        c_d_q = self.bn(en_doc).T @ self.bn(en_question_m.detach()) # doc-to-question

        # sum the cross-correlation matrix between all gpus
        c_q_q.div_(self.batch_size)
        c_d_q.div_(self.batch_size)
        # torch.distributed.all_reduce(c_q_q)
        # torch.distributed.all_reduce(c_d_q)

        on_diag_qq = torch.diagonal(c_q_q).add_(-1).pow_(2).sum()
        on_diag_dq = torch.diagonal(c_d_q).add_(-1).pow_(2).sum()

        off_diag_qq = off_diagonal(c_q_q).pow_(2).sum()
        off_diag_dq = off_diagonal(c_d_q).pow_(2).sum()

        loss_w1 = on_diag_qq + self.lambda_weight * off_diag_qq
        loss_w2 = on_diag_dq + self.lambda_weight * off_diag_dq
        all_loss = (loss_w1*self.w1) + (loss_w1*self.w2)
        return all_loss
