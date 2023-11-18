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


class MLP(nn.Module):
	def __init__(self, dim, projection_size, hidden_size):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(dim, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, projection_size)
		)

	def forward(self, x):
		return self.net(x)

class SCTLoss_distillation(nn.Module):
    def __init__(self, instanceQ_A, instanceQ_B, teacher_temp, model, sentence_embedding_dimension, student_temp=0.05, device=None,path_model='X'):
        """
        param model: SentenceTransformerModel
        K:          queue size
        t:          temperature for student encoder
        temp:       distillation temperature
        """
        super(SCTLoss_distillation, self).__init__()
        self.rep_instanceQ_A = instanceQ_A
        self.rep_instanceQ_B = instanceQ_B
        self.model = model
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.device = device
        self.online_predictor_1 = MLP(sentence_embedding_dimension, sentence_embedding_dimension, 10 * sentence_embedding_dimension)
        self.online_predictor_2 = MLP(sentence_embedding_dimension, sentence_embedding_dimension, 10 * sentence_embedding_dimension)
        self.online_predictor_3 = MLP(sentence_embedding_dimension, sentence_embedding_dimension, 10 * sentence_embedding_dimension)
        self.loss_fnc = nn.KLDivLoss(reduction='batchmean')
        
        self.loss_fct_mse = nn.MSELoss()
        
        # Use multi GPU
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

    def forward(self, 
                sentence_ne_features: Iterable[Dict[str, Tensor]],
                sentence_en_features: Iterable[Dict[str, Tensor]], 
                rep_sent_momentum_A: Tensor,
                rep_sent_momentum_B: Tensor):

        # Batch-size
        rep_sent_A = self.model(sentence_ne_features)['sentence_embedding']
        rep_sent_B = self.model(sentence_en_features)['sentence_embedding']
 
        batch_size = rep_sent_A.shape[0]

        rep_sent_A = F.normalize(rep_sent_A, p=2, dim=1)
        rep_sent_B = F.normalize(rep_sent_B, p=2, dim=1)
        

        # insert the current batch embedding from T
        Q_A = torch.cat((self.rep_instanceQ_A, rep_sent_momentum_A.detach()))
        Q_B = torch.cat((self.rep_instanceQ_B, rep_sent_momentum_B.detach()))
    
        # probability scores distribution for T, S: B X (N + 1)
        logit_momentum_A = torch.einsum('nc,ck->nk', rep_sent_momentum_A, Q_A.t().clone().detach())
        logit_sentence_A = torch.einsum('nc,ck->nk', rep_sent_B, Q_A.t().clone().detach())

        logit_momentum_B = torch.einsum('nc,ck->nk', rep_sent_momentum_B, Q_B.t().clone().detach())
        logit_sentence_B = torch.einsum('nc,ck->nk', rep_sent_A, Q_B.t().clone().detach())

        # Apply temperatures for soft-labels
        momentum_Dist_A = F.softmax(logit_momentum_A/self.teacher_temp, dim=1).detach()
        sentence_Dist_A = logit_sentence_A / self.student_temp

        momentum_Dist_B = F.softmax(logit_momentum_B/self.teacher_temp, dim=1).detach()
        sentence_Dist_B = logit_sentence_B / self.student_temp


        # loss computation, use log_softmax for stable computation
        ARL_loss_score_A = self.loss_fnc(F.log_softmax(sentence_Dist_A, dim=1),momentum_Dist_A.detach()) 
        ARL_loss_score_B = self.loss_fnc(F.log_softmax(sentence_Dist_B, dim=1),momentum_Dist_B.detach()) 

        Final_loss = (ARL_loss_score_A+ARL_loss_score_B).mean()
        # update the random sample queue
        self.rep_instanceQ_A = Q_A[batch_size:]
        self.rep_instanceQ_B = Q_B[batch_size:]
        
        return Final_loss