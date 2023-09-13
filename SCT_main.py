import argparse 

parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--model_save_path",
					type=str,
					default=None,
					required=True,
					help="The output directory where the model checkpoints will be written.")
parser.add_argument("--train_dataset_path",
					type=str,
					default=None,
					required=True,
					help="The directory of train dataset.")
parser.add_argument("--dev_dataset_path",
					type=str,
					default=None,
					required=True,
					help="The directory of dev dataset.")
parser.add_argument("--student_model_name_or_path",
					type=str,
					default=None,
					required=True,
					help="The student model checkpoint for weights initialization.")
parser.add_argument("--train_batch_size", 
					type=int, 
					default=32,
					help="Batch size for training.")
parser.add_argument("--eval_batch_size", 
					type=int, 
					default=32,
					help="Batch size for evaluation.")
parser.add_argument("--inference_batch_size",
					type=int,
					default=16,
					help="Batch size at inference.")
parser.add_argument("--max_seq_length",
					type=int,
					default=32,
					help="Student model max. lengths for inputs (number of word pieces).")
parser.add_argument("--num_epochs",
					type=int,
					default=3,
					help="Total number of training epochs to perform.")
parser.add_argument("--learning_rate",
					type=float,
					default=5e-5,
					help="The initial learning rate for Adam.")
parser.add_argument("--teacher_temp",
					type=float,
					default=0.01,
					help="Distillation temperature.")
parser.add_argument("--student_temp",
					type=float,
					default=0.2,
					help="Temperature for student encoder.")
parser.add_argument("--queue_size",
					type=int,
					default=1000,
					help="The size of instance queue")
parser.add_argument("--gpu_device",
					type=int,
					default=0,
					help="gpu device number")
parser.add_argument("--early_stopping_patience",
					type=int,
					default=7,
					help="Early stopping criteria: patience") 
parser.add_argument("--seed",
					type=int,
					default=1000,
					help="The seed value")					
parser.add_argument("--queue_random",
					type=int,
					default=0,
					help="Random instance queue or not")


args = parser.parse_args()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device)
print(f"GPU:{args.gpu_device}")


import logging
from datetime import datetime
import io
import math
import numpy as np
from glob import glob 
import pickle

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SequentialEvaluator
from sentence_transformers.datasets import ParallelSentencesDatasetForSEEDER
from utils import set_seed

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)



def main():
	set_seed(args.seed)
	print(f"Seed:{args.seed}")

	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Output:{args.model_save_path}")
    
    
    # Prepare data
	logging.info("Preparing training dataset")
	train_data = open(args.train_dataset_path, mode="rt", encoding="utf-8").readlines()
	train_data = [sample.strip().split('\t') for sample in train_data]
	en_texts = [sample[0] for sample in train_data]
	non_en_texts = [sample[1] for sample in train_data]

	train_samples = []
	for en_text, non_en_text in zip(en_texts, non_en_texts): 
		train_samples.append(InputExample(texts=[en_text, non_en_text]))
	
	train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.train_batch_size)
	print(f"Train batch size:{args.train_batch_size}")

	logging.info(f"Create student model from scratch:{args.student_model_name_or_path}")
	student_word_embedding_model = models.Transformer(args.student_model_name_or_path, max_seq_length=args.max_seq_length)
	student_dimension = student_word_embedding_model.get_word_embedding_dimension()
	student_pooling_model = models.Pooling(student_dimension)
	student_model = SentenceTransformer(modules=[student_word_embedding_model, student_pooling_model])

	logging.info(f"Create instance queue")
	print('Random Queue')
	rep_instance_queue_edited_A = torch.randn(args.queue_size, student_dimension).to(device)
	rep_instance_queue_edited_A = F.normalize(rep_instance_queue_edited_A, p=2, dim=1)

	rep_instance_queue_edited_B = torch.randn(args.queue_size, student_dimension).to(device)
	rep_instance_queue_edited_B = F.normalize(rep_instance_queue_edited_B, p=2, dim=1)

	training_loss = losses.SCTLoss(instanceQ_A=rep_instance_queue_edited_A,  
									instanceQ_B=rep_instance_queue_edited_B, 
									model=student_model,
									student_temp=args.student_temp, 
									teacher_temp=args.teacher_temp, 
									device=device,
									sentence_embedding_dimension=student_dimension,
									path_model=args.model_save_path)

	warmup_steps = math.ceil(len(train_dataloader) * args.num_epochs * 0.1)  # 10% of train data for warm-up
	evaluation_steps = 64
	
	logger.info("Load evaluator for STSBenchmark") 
	dev_samples = []
	with io.open(args.dev_dataset_path, "r", encoding="utf-8") as f:
		for line in f:
			text = line.strip().split("\t")
			if text[0] == 'dev':
				sentence1 = text[6]
				sentence2 = text[7]
				score = float(text[5]) / 5.0  #Normalize score to range 0 ... 1
				dev_samples.append(InputExample(texts=[sentence1, sentence2], label=score))
	
	dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=args.eval_batch_size, name='sts-dev')
	
	logger.info("Start training")
	start = datetime.now()
	student_model.fit(train_objectives=[(train_dataloader, training_loss)],
			evaluator=dev_evaluator,
			epochs=args.num_epochs,
			warmup_steps=warmup_steps,
			evaluation_steps=evaluation_steps,
			output_path=args.model_save_path,
			optimizer_params={"lr": args.learning_rate, 'eps': 1e-6, 'correct_bias': False},
			use_amp=True,
			early_stopping_patience=args.early_stopping_patience)
	stop = datetime.now()
	run_time = stop - start
	logger.info("Training time: " + str(run_time) + " s")

if __name__ == "__main__":
	main()