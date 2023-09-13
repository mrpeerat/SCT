import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import losses
from typing import Iterable, Dict
from .. import util

class DISCOLoss(nn.Module):
    def __init__(self, instanceQ, model, teacher_temp=0.01, student_temp=0.2, device=None, number_of_terms=3):
        """
        param model: SentenceTransformerModel
        K:          queue size
        t:          temperature for student encoder
        temp:       distillation temperature
        """
        super(DISCOLoss, self).__init__()
        self.rep_instanceQ = instanceQ
        self.model = model
        self.device = device
        self.terms = number_of_terms
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.similarity_fct = util.cos_sim
        self.scale = (1/student_temp)
        # Use multi GPU
        if torch.cuda.device_count() > 1: 
            self.model = nn.DataParallel(self.model)

    def forward(self, 
                sentence_ne_features: Iterable[Dict[str, Tensor]],
                sentence_en_features: Iterable[Dict[str, Tensor]], 
                rep_sent_en_t: Tensor):

        # Batch-size
        batch_size = rep_sent_en_t.shape[0]

        rep_sent_en_s = self.model(sentence_en_features)['sentence_embedding']
        rep_sent_en_s = F.normalize(rep_sent_en_s, p=2, dim=1)

        # insert the current batch embedding from T
        rep_instanceQ = self.rep_instanceQ

        # probability scores distribution for T, S: B X (N + 1)
        all_negative = torch.cat((rep_sent_en_t, rep_instanceQ))
        scores = self.similarity_fct(rep_sent_en_s, all_negative) * self.scale
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)  # Example a[i] should match with b[i]

        # update the random sample queue
        Q = torch.cat((rep_instanceQ, rep_sent_en_t))
        self.rep_instanceQ = Q[batch_size:]
  
        return self.cross_entropy_loss(scores, labels)