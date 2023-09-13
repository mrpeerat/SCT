# SCT
Implementation of [An Efficient Self-Supervised Cross-View Training For Sentence Embedding (TACL 2023)]().

## Citation
```
@inproceedings{limkonchotiwat-etal-2023-sct,
    title = "An Efficient Self-Supervised Cross-View Training For Sentence Embedding",
    author = "Limkonchotiwat, Peerat  and
      Ponwitayarat, Wuttikorn  and
      Lowphansirikul, Lalita and
      Udomcharoenchaikit, Can  and
      Chuangsuwanich, Ekapol  and
      Nutanong, Sarana",
    journal = "Transactions of the Association for Computational Linguistics",
    year = "2023",
    address = "Cambridge, MA",
    publisher = "MIT Press",
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
>> bash Runing_distillation_script.sh
>> bash Runing_script.sh
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
cd SentEval/data/downstream/
bash download_dataset.sh
```

## Evaluation - Notebook
Please see [notebooks]().

## Evaluation - Python
Then come back to the root directory, you can evaluate any `sentence transformers` models using SimCSE evaluation code. For example,
```bash
python evaluation.py \
    --model_name_or_path "your-model-path" \
    --task_set sts \
    --mode test
```

## Main results - STS
XXXXXXXX

## Downstream tasks - Reranking and NLI
XXXXXXXX
