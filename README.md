# SCT
Implementation of [An Efficient Self-Supervised Cross-View Training For Sentence Embedding (TACL 2023)](https://arxiv.org/abs/2311.03228).

## Citation
```
@article{10.1162/tacl_a_00620,
    author = {Limkonchotiwat, Peerat and Ponwitayarat, Wuttikorn and Lowphansirikul, Lalita and Udomcharoenchaikit, Can and Chuangsuwanich, Ekapol and Nutanong, Sarana},
    title = "{An Efficient Self-Supervised Cross-View Training For Sentence Embedding}",
    journal = {Transactions of the Association for Computational Linguistics},
    volume = {11},
    pages = {1572-1587},
    year = {2023},
    month = {12},
    issn = {2307-387X},
    doi = {10.1162/tacl_a_00620},
    url = {https://doi.org/10.1162/tacl\_a\_00620},
    eprint = {https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl\_a\_00620/2196817/tacl\_a\_00620.pdf},
}
```

## Installation
```
git clone https://github.com/mrpeerat/SCT
cd SCT
pip install -e .
``` 

## Our models (Huggingface)
### Self-supervised
- [SCT-BERT-Tiny](https://huggingface.co/mrp/SCT_BERT_Tiny)
- [SCT-BERT-Mini](https://huggingface.co/mrp/SCT_BERT_Mini)
- [SCT-BERT-Small](https://huggingface.co/mrp/SCT_BERT_Small)
- [SCT-BERT-Base](https://huggingface.co/mrp/SCT_BERT_Base)
- [SCT-BERT-Large](https://huggingface.co/mrp/SCT_BERT_Large)
### Distillation
- [SCT-Distillation-BERT-Tiny](https://huggingface.co/mrp/SCT_Distillation_BERT_Small)
- [SCT-Distillation-BERT-Mini](https://huggingface.co/mrp/SCT_Distillation_BERT_Mini)
- [SCT-Distillation-BERT-Small](https://huggingface.co/mrp/SCT_Distillation_BERT_Small)
- [SCT-Distillation-BERT-Base](https://huggingface.co/mrp/SCT_Distillation_BERT_Base)


## Usage
### Training data
We use the training data from [BSL's paper](https://aclanthology.org/2021.acl-long.402.pdf): [here](https://drive.google.com/file/d/1HeqsEChDr7i_kxbdJvmVaRMSFKDRnFBY/view?usp=sharing).

### Development data
We use sts-b development set from [sentence transformer](https://sbert.net/datasets/stsbenchmark.tsv.gz).

### Parameters
Self-supervised:
| Models  | Reference Temp | Student Temp | Queue Size | Learning Rate |   
| --------------------- | ----- | ----- | -----| ----|
|BERT-Tiny              | 0.03  | 0.04  | 131072| 5e-4|
|BERT-Mini              | 0.01  | 0.03  | 131072| 3e-4|  
|BERT-Small             | 0.02  | 0.03  | 65536| 3e-4|  
|BERT-Base              | 0.04  | 0.05  | 65536| 5e-4| 
|BERT-Large             | 0.04  | 0.05  | 16384| 5e-4| 

Distillation:
| Models  | Reference Temp | Student Temp | Queue Size | Learning Rate |   
| --------------------- | ----- | ----- | -----| ----|
|BERT-Tiny              | 0.03  | 0.04  | 131072| 5e-4|
|BERT-Mini              | 0.04  | 0.05  | 65536| 1e-4|  
|BERT-Small             | 0.04  | 0.05  | 131072| 1e-4|  
|BERT-Base              | 0.04  | 0.05  | 65536| 1e-4| 

### Train your own model
Please set the model's parameter before training.
```bash
>> bash Running_distillation_script.sh
>> bash Running_script.sh
```

For finetuning model parameters: 
```
learning_rate_all=(1e-4 3e-4 5e-4)
queue_sizes=(131072 65536 16384)
teacher_temps=(0.01 0.02 0.03 0.04 0.05 0.06 0.07)
student_temps=(0.01 0.02 0.03 0.04 0.05 0.06 0.07)
```

# Evaluation
Our evaluation code for sentence embeddings is based on a modified version of [SentEval](https://github.com/facebookresearch/SentEval) and [SimCSE](https://github.com/princeton-nlp/SimCSE).

Before evaluation, please download the evaluation datasets by running
```bash
cd SentEval
pip install -e .
cd data/downstream/
bash download_dataset.sh
```

## Evaluation - Notebook
Please see this [notebooks](https://github.com/mrpeerat/SCT/blob/main/notebook/STS_eval.ipynb).

## Evaluation - Python
```bash
python evaluation.py \
    --model_name_or_path "your-model-path" \
    --task_set sts \
    --mode test
```

## Main results - STS
Self-supervised:
| Models  | STS (Avg.) | 
| --------------------- | ----- |
|[SCT-BERT-Tiny](https://huggingface.co/mrp/SCT_BERT_Tiny)              | 69.73  | 
|[SCT-BERT-Mini](https://huggingface.co/mrp/SCT_BERT_Mini)              | 69.59  | 
|[SCT-BERT-Small](https://huggingface.co/mrp/SCT_BERT_Small)             | 72.56  | 
|[SCT-BERT-Base](https://huggingface.co/mrp/SCT_BERT_Base)              | 75.55  | 
|[SCT-BERT-Large](https://huggingface.co/mrp/SCT_BERT_Large)             | 78.16  | 

Distillation:
| Models  | STS (Avg.) | 
| --------------------- | ----- |
|[SCT-Distillation-BERT-Tiny](https://huggingface.co/mrp/SCT_Distillation_BERT_Small)             | 76.43  | 
|[SCT-Distillation-BERT-Mini](https://huggingface.co/mrp/SCT_Distillation_BERT_Mini)              | 77.58  | 
|[SCT-Distillation-BERT-Small](https://huggingface.co/mrp/SCT_Distillation_BERT_Small)             | 78.16  | 
|[SCT-Distillation-BERT-Base](https://huggingface.co/mrp/SCT_Distillation_BERT_Base)              | 79.58  | 

## Downstream tasks - Reranking and NLI
- For the reranking evaluation code, we use [USEB](https://github.com/UKPLab/useb)
- For the NLI evaluation code, we use [SentEval](https://github.com/facebookresearch/SentEval)

Self-supervised:
| Models  | Reranking (Avg.) | NLI (Avg.) |
| --------------------- | ----- | ----- |
|[SCT-BERT-Tiny](https://huggingface.co/mrp/SCT_BERT_Tiny)             | 55.29  | 71.89  |
|[SCT-BERT-Small](https://huggingface.co/mrp/SCT_BERT_Small)             | 58.59  | 75.70  |
|[SCT-BERT-Base](https://huggingface.co/mrp/SCT_BERT_Base)              | 60.97  | 77.93  |
|[SCT-BERT-Large](https://huggingface.co/mrp/SCT_BERT_Large)             | 63.02  | 79.55  |

Distillation:
| Models  | Reranking (Avg.) | NLI (Avg.) |
| --------------------- | ----- | ----- |
|[SCT-Distillation-BERT-Tiny](https://huggingface.co/mrp/SCT_Distillation_BERT_Small)             | 61.14  | 78.53  |
|[SCT-Distillation-BERT-Small](https://huggingface.co/mrp/SCT_Distillation_BERT_Small)             | 61.94  | 80.44  |
|[SCT-Distillation-BERT-Base](https://huggingface.co/mrp/SCT_Distillation_BERT_Base)              | 64.63  | 80.97  |
