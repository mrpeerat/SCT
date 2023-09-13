import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import losses
from typing import Iterable, Dict


class SEEDERLoss_QA(nn.Module):
    def __init__(self, instanceQ, model, teacher_temp=0.05, student_temp=0.07, device=None, number_of_terms=2):
        """
        param model: SentenceTransformerModel
        K:          queue size
        t:          temperature for student encoder
        temp:       distillation temperature
        """
        super(SEEDERLoss_QA, self).__init__()
        self.rep_instanceQ = instanceQ
        self.model = model
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.device = device
        self.terms = number_of_terms
        
        # Use multi GPU
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], rep_sent_teacher: Tensor):

        # Batch-size
        batch_size = rep_sent_teacher.shape[0]

        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        # raise Exception(len(reps),"----",len(reps[0]),'---',self.embed_size, '----', self.batch_size)
        rep_sent_non_en_question = reps[0]
        rep_sent_en_doc = reps[2]
 
        rep_sent_non_en_question = F.normalize(rep_sent_non_en_question, p=2, dim=1)
        rep_sent_en_doc = F.normalize(rep_sent_en_doc, p=2, dim=1)


        # insert the current batch embedding from T
        rep_instanceQ = self.rep_instanceQ
        Q = torch.cat((rep_instanceQ, rep_sent_teacher))
    
        # probability scores distribution for T, S: B X (N + 1)
        logit_en_t = torch.einsum('nc,ck->nk', rep_sent_teacher, Q.t().clone().detach())
        logit_ne_s = torch.einsum('nc,ck->nk', rep_sent_non_en_question, Q.t().clone().detach())
        logit_en_s = torch.einsum('nc,ck->nk', rep_sent_en_doc, Q.t().clone().detach())


        # Apply temperatures for soft-labels
        en_T_Dist = F.softmax(logit_en_t/self.teacher_temp, dim=1)
        ne_S_Dist = logit_ne_s / self.student_temp
        en_S_Dist = logit_en_s / self.student_temp
        

        # loss computation, use log_softmax for stable computation
        loss_t_s_en = -torch.mul(en_T_Dist, F.log_softmax(en_S_Dist, dim=1)).sum() / batch_size
        loss_t_s_ne = -torch.mul(en_T_Dist, F.log_softmax(ne_S_Dist, dim=1)).sum() / batch_size

        seeder_loss = (loss_t_s_en + loss_t_s_ne)/2
        
        # update the random sample queue
        self.rep_instanceQ = Q[batch_size:]
  
        return seeder_loss