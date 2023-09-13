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
from .. import util

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

class MoKD_Distillation(nn.Module):
    def __init__(self, model_1, model_2, sentence_embedding_dimension, student_temp=0.05, device=None,path_model='X', moving_average_decay=0.999, similarity_fct = util.cos_sim):
        """
        param model: SentenceTransformerModel
        K:          queue size
        t:          temperature for student encoder
        temp:       distillation temperature
        """
        super(MoKD_Distillation, self).__init__()
        self.model_1 = model_1
        self.model_2 = model_2
        self.student_temp = student_temp
        self.device = device
        self.online_predictor_1 = MLP(sentence_embedding_dimension, sentence_embedding_dimension, 10 * sentence_embedding_dimension)
        self.online_predictor_2 = MLP(sentence_embedding_dimension, sentence_embedding_dimension, 10 * sentence_embedding_dimension)
        self.online_predictor_3 = MLP(sentence_embedding_dimension, sentence_embedding_dimension, 10 * sentence_embedding_dimension)

        self.target_encoder_1 = copy.deepcopy(self.model_1)
        self.target_ema_updater_1 = EMA(moving_average_decay)

        self.target_encoder_2 = copy.deepcopy(self.model_2)
        self.target_ema_updater_2 = EMA(moving_average_decay)

        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        
        # Use multi GPU
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
    
    def update_moving_average(self):
        update_moving_average(self.target_ema_updater_1, self.target_encoder_1, self.model_1) 
        # update_moving_average(self.target_ema_updater_2, self.target_encoder_2, self.model_2) 

    def forward(self, 
                sentence_features,
                rep_sent_en_t: Tensor):

        # Batch-size
        target_sentence_features = copy.deepcopy(sentence_features)

        rep_one, rep_two = [self.model_1(sentence_feature) for sentence_feature in sentence_features]
        # _, rep_two = [self.model_2(sentence_feature) for sentence_feature in sentence_features]

        rep_sent_student_1 = rep_one['sentence_embedding']
        rep_sent_student_2 = rep_two['sentence_embedding']

        online_pred_one = self.online_predictor_1(rep_sent_student_1)
        online_pred_one = self.online_predictor_2(online_pred_one)
        rep_sent_student_1 = self.online_predictor_3(online_pred_one)

        online_pred_one = self.online_predictor_1(rep_sent_student_2)
        online_pred_one = self.online_predictor_2(online_pred_one)
        rep_sent_student_2 = self.online_predictor_3(online_pred_one)

        rep_sent_student_1 = F.normalize(rep_sent_student_1, p=2, dim=1)
        rep_sent_student_2 = F.normalize(rep_sent_student_2, p=2, dim=1)

        with torch.no_grad():
            rep_one, rep_two = [self.target_encoder_1(sentence_feature) for sentence_feature in target_sentence_features]
            # _, rep_two = [self.target_encoder_2(sentence_feature) for sentence_feature in target_sentence_features]

            rep_sent_teacher_1 = rep_two['sentence_embedding']
            rep_sent_teacher_2 = rep_one['sentence_embedding']

            rep_sent_teacher_1 = F.normalize(rep_sent_teacher_1, p=2, dim=1)
            rep_sent_teacher_2 = F.normalize(rep_sent_teacher_2, p=2, dim=1)

        scores = self.similarity_fct(rep_sent_teacher_1, rep_sent_student_1)/self.student_temp
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)
        ssl_loss_1 = self.cross_entropy_loss(scores, labels)

        scores = self.similarity_fct(rep_sent_teacher_2, rep_sent_student_2)/self.student_temp
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)
        ssl_loss_2 = self.cross_entropy_loss(scores, labels)

        scores = self.similarity_fct(rep_sent_teacher_1, rep_sent_student_2)/self.student_temp
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)
        cross_loss_1 = self.cross_entropy_loss(scores, labels)

        scores = self.similarity_fct(rep_sent_teacher_2, rep_sent_student_1)/self.student_temp
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)
        cross_loss_2 = self.cross_entropy_loss(scores, labels)

        ssl_loss = ssl_loss_1 + (0.1*cross_loss_2) #(cross_loss_2*self.weight_1)
        cross_loss = ssl_loss_2 + (1*cross_loss_1) #(cross_loss_1*self.weight_2)
        final_loss = ssl_loss + cross_loss
        return final_loss