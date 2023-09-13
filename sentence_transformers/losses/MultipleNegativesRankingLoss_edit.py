import torch
from torch import nn, Tensor
from typing import Iterable, Dict
from ..SentenceTransformer import SentenceTransformer
from .. import util
import torch.nn.functional as F

class MultipleNegativesRankingLoss_edit(nn.Module):
    """
        This loss expects as input a batch consisting of sentence pairs (a_1, p_1), (a_2, p_2)..., (a_n, p_n)
        where we assume that (a_i, p_i) are a positive pair and (a_i, p_j) for i!=j a negative pair.

        For each a_i, it uses all other p_j as negative samples, i.e., for a_i, we have 1 positive example (p_i) and
        n-1 negative examples (p_j). It then minimizes the negative log-likehood for softmax normalized scores.

        This loss function works great to train embeddings for retrieval setups where you have positive pairs (e.g. (query, relevant_doc))
        as it will sample in each batch n-1 negative docs randomly.

        The performance usually increases with increasing batch sizes.

        For more information, see: https://arxiv.org/pdf/1705.00652.pdf
        (Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4)

        You can also provide one or multiple hard negatives per anchor-positive pair by structering the data like this:
        (a_1, p_1, n_1), (a_2, p_2, n_2)

        Here, n_1 is a hard negative for (a_1, p_1). The loss will use for the pair (a_i, p_i) all p_j (j!=i) and all n_j as negatives.

        Example::

            from sentence_transformers import SentenceTransformer, losses, InputExample
            from torch.utils.data import DataLoader

            model = SentenceTransformer('distilbert-base-uncased')
            train_examples = [InputExample(texts=['Anchor 1', 'Positive 1']),
                InputExample(texts=['Anchor 2', 'Positive 2'])]
            train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
            train_loss = losses.MultipleNegativesRankingLoss(model=model)
    """
    def __init__(self, model: SentenceTransformer, teacher_model, weight_one, weight_two, scale: float = 20.0, similarity_fct = util.cos_sim):
        """
        :param model: SentenceTransformer model
        :param scale: Output of similarity function is multiplied by scale value
        :param similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to dot product (and then set scale to 1)
        """
        super(MultipleNegativesRankingLoss_edit, self).__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.teacher_model = teacher_model
        self.mse_fct = nn.MSELoss()
        self.weight_one = weight_one
        self.weight_two = weight_two


    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        student_reps1, student_reps2, student_reps3 = [self.model(sentence_feature) for sentence_feature in sentence_features]
        embeddings_ne,embeddings_en,embeddings_doc = student_reps1['sentence_embedding'], student_reps2['sentence_embedding'], student_reps3['sentence_embedding']
        embeddings_ne = F.normalize(embeddings_ne, p=2, dim=1)
        embeddings_en = F.normalize(embeddings_en, p=2, dim=1)
        embeddings_doc = F.normalize(embeddings_doc, p=2, dim=1)
        
        with torch.no_grad():
            teacher_reps1, teacher_reps2, teacher_reps3 = [self.teacher_model(sentence_feature) for sentence_feature in sentence_features]
            embeddings_ne_teacher,embeddings_en_teacher,embeddings_doc_teacher = teacher_reps1['sentence_embedding'], teacher_reps2['sentence_embedding'], teacher_reps3['sentence_embedding']
            embeddings_ne_teacher = F.normalize(embeddings_ne_teacher, p=2, dim=1)
            embeddings_en_teacher = F.normalize(embeddings_en_teacher, p=2, dim=1)
            embeddings_doc_teacher = F.normalize(embeddings_doc_teacher, p=2, dim=1)
                            
        
        # LKT
        lkt_term_one = self.mse_fct(embeddings_en_teacher.detach(), embeddings_ne)
        lkt_term_two = self.mse_fct(embeddings_doc_teacher.detach(), embeddings_doc)
        lkt_term_three = self.mse_fct(embeddings_doc_teacher.detach(), embeddings_ne)
        lkt_term_four = self.mse_fct(embeddings_en_teacher.detach(), embeddings_en)
        
        lkt_loss = ((lkt_term_one+lkt_term_two+lkt_term_three+lkt_term_four)*1000)/4
        
        
        
        # contrastive learning
        scores_one = self.similarity_fct(embeddings_en_teacher.detach(), embeddings_ne) * self.scale
        labels_one = torch.tensor(range(len(scores_one)), dtype=torch.long, device=scores_one.device)  # Example a[i] should match with b[i]
        CL_one = self.cross_entropy_loss(scores_one, labels_one)
        
        scores_two = self.similarity_fct(embeddings_doc_teacher.detach(), embeddings_ne) * self.scale
        labels_two = torch.tensor(range(len(scores_two)), dtype=torch.long, device=scores_two.device)  # Example a[i] should match with b[i]
        CL_two = self.cross_entropy_loss(scores_two, labels_two)
        
        contrastive_loss = (CL_one + CL_two)/2
        
        
        all_loss = self.weight_one*lkt_loss + contrastive_loss*self.weight_two

        return all_loss
        

    def get_config_dict(self):
        return {'scale': self.scale, 'similarity_fct': self.similarity_fct.__name__}





