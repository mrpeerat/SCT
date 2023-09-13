import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import losses
from typing import Iterable, Dict
import numpy as np

class BCRLoss(nn.Module):
    def __init__(self, instanceQ, model, teacher_temp=0.01, student_temp=0.2, device=None):
        """
        param model: SentenceTransformerModel
        K:          queue size
        t:          temperature for student encoder
        temp:       distillation temperature
        """
        super(BCRLoss, self).__init__()
        self.rep_instanceQ = instanceQ
        self.model = model
        self.device = device
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        
        # Use multi GPU
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

    def forward(self, 
                sentence_ne_features: Iterable[Dict[str, Tensor]],
                sentence_en_features: Iterable[Dict[str, Tensor]], 
                rep_sent_en_t: Tensor):

        # Batch-size
        batch_size = rep_sent_en_t.shape[0]

        # rep_sent_ne_s = self.model(sentence_ne_features)['sentence_embedding']
        rep_sent_en_s = self.model(sentence_en_features)['sentence_embedding']
 
        # rep_sent_ne_s = F.normalize(rep_sent_ne_s, p=2, dim=1)
        rep_sent_en_s = F.normalize(rep_sent_en_s, p=2, dim=1)

        # insert the current batch embedding from T
        rep_instanceQ = self.rep_instanceQ
        Q = torch.cat((rep_instanceQ, rep_sent_en_t))
    

        logit_t_s = self.cos(rep_sent_en_t, rep_sent_en_s)
        rep_instance_queue=rep_instanceQ[torch.randint(len(rep_instanceQ),(batch_size,))]
        logit_t_q_s = self.cos(rep_instance_queue, rep_sent_en_s)

        logit_t_s = torch.log(logit_t_s)
        logit_t_q_s = 1-torch.log(1-logit_t_q_s)
        logit_t_q_s = torch.matmul(batch_size,logit_t_q_s)    
        logits = logit_t_s+logit_t_q_s
        all_lost = torch.sum(logits)
        
        # update the random sample queue
        self.rep_instanceQ = Q[:batch_size]
  
        return all_lost