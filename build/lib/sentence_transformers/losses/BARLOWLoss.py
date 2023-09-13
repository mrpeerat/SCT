import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import losses
from typing import Iterable, Dict


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class BARLOWLoss(nn.Module):
    def __init__(self, model, embed_size, batch_size, lambda_weight=0.0005, w1=1, w2=1, w3=1, device=None, projector='4096-4096-4096'):
        """
        param model: SentenceTransformerModel
        K:          queue size
        t:          temperature for student encoder
        temp:       distillation temperature
        """
        super(BARLOWLoss, self).__init__()

        self.model = model
        self.lambda_weight = lambda_weight
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.device = device
        self.batch_size = batch_size
        self.embed_size = embed_size
        
        # sizes = [embed_size] + list(map(int, projector.split('-')))
        # layers = []
        # for i in range(len(sizes) - 2):
        #     layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
        #     layers.append(nn.BatchNorm1d(sizes[i + 1]))
        #     layers.append(nn.ReLU(inplace=True))
        # layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        # self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(embed_size, affine=False)

        # Use multi GPU
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], teacher_reps: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        # raise Exception(len(reps),"----",len(reps[0]),'---',self.embed_size, '----', self.batch_size)
        non_en_question = self.projector(reps[0])
        en_question = self.projector(reps[1])
        en_doc = self.projector(reps[2])

        c_q_q = self.bn(non_en_question).T @ self.bn(en_question) # question-to-question
        c_d_q = self.bn(en_doc).T @ self.bn(en_question) # doc-to-question

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