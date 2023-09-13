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
			nn.BatchNorm1d(hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, projection_size)
		)

	def forward(self, x):
		return self.net(x)

class SEEDERLoss_v3(nn.Module):
    def __init__(self, instanceQ, model, sentence_embedding_dimension, moving_average_decay=0.999, student_temp=0.05, device=None, number_of_terms=2):
        """
        param model: SentenceTransformerModel
        K:          queue size
        t:          temperature for student encoder
        temp:       distillation temperature
        """
        super(SEEDERLoss_v3, self).__init__()
        self.rep_instanceQ = instanceQ
        self.model = model
        self.student_temp = student_temp
        self.device = device
        self.terms = number_of_terms
        self.target_encoder = copy.deepcopy(self.model)
        self.target_ema_updater = EMA(moving_average_decay)
        self.online_predictor_1 = MLP(sentence_embedding_dimension, sentence_embedding_dimension, 10 * sentence_embedding_dimension)
        self.online_predictor_2 = MLP(sentence_embedding_dimension, sentence_embedding_dimension, 10 * sentence_embedding_dimension)
        self.online_predictor_3 = MLP(sentence_embedding_dimension, sentence_embedding_dimension, 10 * sentence_embedding_dimension)
        self.loss_fnc = nn.KLDivLoss(reduction='batchmean')
        
        # Use multi GPU
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
    
    def update_moving_average(self):
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.model)

    def forward(self, 
                sentence_ne_features: Iterable[Dict[str, Tensor]],
                sentence_en_features: Iterable[Dict[str, Tensor]], 
                rep_sent_en_t: Tensor):

        # Batch-size
        batch_size = rep_sent_en_t.shape[0]

        rep_sent_ne_s = self.model(sentence_ne_features)['sentence_embedding']
        rep_sent_en_s = self.model(sentence_en_features)['sentence_embedding']

        online_pred_one, online_pred_two = self.online_predictor_1(rep_sent_en_s), self.online_predictor_1(rep_sent_ne_s)
        online_pred_one, online_pred_two = self.online_predictor_2(online_pred_one), self.online_predictor_2(online_pred_two)
#         online_pred_one, online_pred_two = self.online_predictor_3(online_pred_one), self.online_predictor_3(online_pred_two)

        rep_sent_ne_s = F.normalize(online_pred_two, p=2, dim=1)
        rep_sent_en_s = F.normalize(online_pred_one, p=2, dim=1)


        with torch.no_grad():
            rep_sent_en_s_momentum = self.target_encoder(sentence_en_features)['sentence_embedding']
            rep_sent_en_s_momentum = F.normalize(rep_sent_en_s_momentum, p=2, dim=1)

        # insert the current batch embedding from T
        Q = self.rep_instanceQ
    
        # probability scores distribution for T, S: B X (N + 1)
        logit_en_momentum = torch.einsum('nc,ck->nk', rep_sent_en_s_momentum, Q.t().clone().detach())
        logit_ne_s = torch.einsum('nc,ck->nk', rep_sent_ne_s, Q.t().clone().detach())
        logit_en_s = torch.einsum('nc,ck->nk', rep_sent_en_s, Q.t().clone().detach())


        # Apply temperatures for soft-labels
        en_momentum_Dist = F.softmax(logit_en_momentum/self.student_temp, dim=1)
        ne_S_Dist = logit_ne_s / self.student_temp
        en_S_Dist = logit_en_s / self.student_temp

        # loss computation, use log_softmax for stable computation
        loss_t_s_en = self.loss_fnc(F.log_softmax(en_S_Dist, dim=1),en_momentum_Dist.detach()) 
        loss_t_s_ne = self.loss_fnc(F.log_softmax(ne_S_Dist, dim=1),en_momentum_Dist.detach())  
        # print(f"Loss1:{loss_t_s_en:.4f} Loss2:{loss_t_s_ne:.4f} Loss3:{loss_s_s_ne:.4f}")

        if self.terms == 2:
            seeder_loss = loss_t_s_en + loss_t_s_ne
        elif self.terms == 1:
            seeder_loss = loss_t_s_en
        else:
            raise Exception('Term error')
        
        # update the random sample queue
        Q = torch.cat((Q, rep_sent_en_s_momentum))
        self.rep_instanceQ = Q[batch_size:]
  
        return seeder_loss