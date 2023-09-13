import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import losses
from typing import Iterable, Dict
import copy
import random
import numpy as np
import math
from functools import wraps

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

class MLP(nn.Module):
	def __init__(self, dim, projection_size, hidden_size):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(dim, hidden_size),
# 			nn.BatchNorm1d(hidden_size),
			nn.ReLU(),
# 			nn.Linear(hidden_size, hidden_size),
# 			nn.ReLU(),
			nn.Linear(hidden_size, projection_size)
		)

	def forward(self, x):
		return self.net(x)

class ARL_Loss_NoMLP(nn.Module):
    def __init__(self, instanceQ, teacher_temp, model, sentence_embedding_dimension, moving_average_decay=0.999, student_temp=0.05, device=None):
        """
        param model: SentenceTransformerModel
        K:          queue size
        t:          temperature for student encoder
        temp:       distillation temperature
        """
        super(ARL_Loss_NoMLP, self).__init__()
        self.rep_instanceQ = instanceQ
        self.model = model
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.device = device
        self.target_encoder = copy.deepcopy(self.model)
        self.target_ema_updater = EMA(moving_average_decay)
        self.online_predictor_1 = MLP(sentence_embedding_dimension, sentence_embedding_dimension, 10 * sentence_embedding_dimension)
        self.online_predictor_2 = MLP(sentence_embedding_dimension, sentence_embedding_dimension, 10 * sentence_embedding_dimension)
        self.online_predictor_3 = MLP(sentence_embedding_dimension, sentence_embedding_dimension, 10 * sentence_embedding_dimension)
        self.loss_fnc = nn.KLDivLoss(reduction='batchmean')
        self.count=0
        
        # Use multi GPU
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
    
    def update_moving_average(self):
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.model)

    def forward(self, 
                sentence_features: Iterable[Dict[str, Tensor]],
                rep_sent_en_t: Tensor):

        # Batch-size
        sent_A, sent_B = sentence_features
        
     
        rep_sent_A = self.model(sent_A)['sentence_embedding']
        batch_size = rep_sent_A.shape[0]
#         rep_sent_A = self.online_predictor_1(rep_sent_A)
#         rep_sent_A = self.online_predictor_2(rep_sent_A)
#         rep_sent_A = self.online_predictor_3(rep_sent_A)

        rep_sent_A = F.normalize(rep_sent_A, p=2, dim=1)


        with torch.no_grad():
            rep_sent_momentum = self.target_encoder(sent_B)['sentence_embedding']
            rep_sent_momentum = F.normalize(rep_sent_momentum, p=2, dim=1).detach()

        # insert the current batch embedding from T
        Q = torch.cat((self.rep_instanceQ, rep_sent_momentum.detach()))
        
        
#         raise Exception(f"{rep_sent_A.shape}//{rep_sent_momentum.shape}//{len(sent_A)}//{len(sent_B)}//{len(Q)}//{rep_sent_A[0]}")
        
#         Q = self.rep_instanceQ
    
        # probability scores distribution for T, S: B X (N + 1)
        logit_momentum = torch.einsum('nc,ck->nk', rep_sent_momentum, Q.t().clone().detach())
        logit_sentence = torch.einsum('nc,ck->nk', rep_sent_A, Q.t().clone().detach())
        


        # Apply temperatures for soft-labels
        momentum_Dist = F.softmax(logit_momentum/self.teacher_temp, dim=1).detach()
        sentence_Dist = logit_sentence / self.student_temp

        # loss computation, use log_softmax for stable computation
        ARL_loss_score = self.loss_fnc(F.log_softmax(sentence_Dist, dim=1),momentum_Dist.detach()) 
#         ARL_loss_score = -torch.mul(momentum_Dist.detach(), F.log_softmax(sentence_Dist, dim=1)).sum() / batch_size

        # update the random sample queue
        self.rep_instanceQ = Q[batch_size:]
        self.count+=1
        
        return ARL_loss_score