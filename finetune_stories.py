from transformers import Trainer, TrainingArguments, AutoModelForCausalLM
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
import torch
import numpy as np
import math
import os
from datasets import Dataset
import sys
sys.path.append('/work/09192/jianing/ls6/Memory_generation')
from utils import model_to_path_dict
import os
import argparse
from datetime import datetime
import pickle


# Get the current date and format it as MMDD
current_date = datetime.now().strftime("%m%d%H%M")
os.environ["WANDB_PROJECT"]="stories_finetuning"

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     logits = torch.tensor(logits)  # Convert to tensor for CE loss
#     labels = torch.tensor(labels) 
#     predictions = torch.argmax(logits, axis=-1)
#     with torch.no_grad():  
#         shift_logits = logits[..., :-1, :].reshape(-1, logits.shape[-1])
#         shift_labels = labels[..., 1:].reshape(-1)
#         loss_fct = torch.nn.CrossEntropyLoss()
#         loss = loss_fct(shift_logits, shift_labels)
#         perplexity = math.exp(loss.item())
#     return {"perplexity": perplexity}

def main(args):
    if args.all_moth:
        with open(args.moth_story_names_path,'rb')as f:
            target_stories = pickle.load(f)
    else:
        target_stories = ['pieman','alternateithicatom', 'avatar', 'howtodraw', 'legacy', 
            'life', 'myfirstdaywiththeyankees', 'naked', 
            'odetostepfather', 'souls', 'undertheinfluence',
            'stagefright', 'tildeath', 'sloth', 'exorcism', 'haveyoumethimyet', 
        'adollshouse', 'inamoment', 'theclosetthatateeverything', 'adventuresinsayingyes',
        'buck', 'swimmingwithastronauts', 'thatthingonmyarm', 'eyespy', 'itsabox', 'hangtime',
        'fromboyhoodtofatherhood',
        'wheretheressmoke']
    
    # prepare txt
    file_paths = [os.path.join(args.story_transcript_dir, fname) for fname in os.listdir(args.story_transcript_dir) if fname.split('.')[0] in target_stories]
    texts = []
    for file_path in file_paths[1:]:
        with open(file_path, "r", encoding="utf-8") as f:
            story = f.read()
        texts.append({"text": story})
    dataset = Dataset.from_list(texts)

    model_hf_name = model_to_path_dict[args.model]['hf_name']
    tokenizer = AutoTokenizer.from_pretrained(model_hf_name)
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], return_special_tokens_mask=True, truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    if args.use_checkpoint is None:
        model = AutoModelForCausalLM.from_pretrained(model_hf_name,device_map='auto',torch_dtype=torch.bfloat16,)
    else:
        checkpoint_dir = os.path.join(args.training_dir,'results',args.use_checkpoint)
        model = AutoModelForCausalLM.from_pretrained(checkpoint_dir,device_map='auto',torch_dtype=torch.bfloat16,)
    tokenizer.pad_token = tokenizer.eos_token

    # Training arguments with evaluation after each epoch
    training_args = TrainingArguments(
        output_dir=os.path.join(args.training_dir,'results'),
        eval_strategy="epoch",     # Evaluate after every epoch
        learning_rate=args.lr,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=args.nepoch,
        weight_decay=1e-4,
        lr_scheduler_type="cosine",
        logging_dir=os.path.join(args.training_dir,'logs'), # Directory to save logs
        logging_steps=10,
        save_strategy="epoch",           # Save the model after each epoch
        save_total_limit=2,              # Limit the number of saved checkpoints
        load_best_model_at_end=True,
        bf16 = True,
        eval_accumulation_steps=5,    # offloads memory for eval after every 5 steps
        gradient_checkpointing=True,  # Enables activation checkpointing
        report_to="wandb",                 # Report training metrics to Weights & Biases
        run_name="stories_finetuning_%s"%current_date
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        eval_dataset=tokenized_datasets,
        tokenizer=tokenizer,
        data_collator=data_collator,
        #compute_metrics=compute_metrics,  # Custom metrics for evaluation
    )

    if args.use_checkpoint is None:
        trainer.train()
    else:
        trainer.train(resume_from_checkpoint=checkpoint_dir)

    # Evaluate the model after training
    eval_results = trainer.evaluate()
    # Print and save the final evaluation results
    print(f"Final Evaluation Results: {eval_results}")
    if not os.path.exists(os.path.join(args.training_dir,current_date)):
        os.makedirs(os.path.join(args.training_dir,current_date))
    with open(os.path.join(args.training_dir,current_date,"eval_results.txt"), "w") as f:
        f.write(str(eval_results))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_dir',default = '/work/09192/jianing/ls6/Memory_generation/training')
    parser.add_argument('--story_transcript_dir',default = '/work/09192/jianing/ls6/Memory_generation/transcripts/moth_stories')
    parser.add_argument("--all_moth",action = 'store_true', default = 'use all moth stories (301 total including pie man). else just the 28 scanning stories + pie man')
    parser.add_argument("--moth_story_names_path",default = "/work/09192/jianing/ls6/Memory_generation/moth_story_names.pkl")
    parser.add_argument('--model',default = 'Llama3.2-3b-instruct')
    parser.add_argument('--use_checkpoint',type = str,default = None)
    parser.add_argument('--nepoch',type = int,default = 5)
    parser.add_argument('--lr',type = float, default = 1e-7)
    parser.add_argument('--batch_size',type = int, default = 4)
    args = parser.parse_args()
    main(args)

