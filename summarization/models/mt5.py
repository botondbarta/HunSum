import os

import pandas as pd
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, \
    Seq2SeqTrainer

from summarization.models.base_model import BaseModel


class MT5(BaseModel):
    def __init__(self, config_path):
        super().__init__(config_path)

        self.model_checkpoint = self.config.mt5.model_checkpoint

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_checkpoint)

    def process_data_to_model_inputs(self, batch):
        # Tokenize the input and target data
        inputs = self.tokenizer(batch['article'], padding='max_length', truncation=True)
        with self.tokenizer.as_target_tokenizer():
            outputs = self.tokenizer(batch['lead'], padding='max_length', truncation=True)

        inputs['labels'] = outputs['input_ids']
        return inputs

    def full_train(self):
        dataset = self.load_dataset(self.config.data_dir)
        tokenized_datasets = self.tokenize_datasets(dataset)

        model_name = self.model_checkpoint.split("/")[-1]
        args = Seq2SeqTrainingArguments(
            output_dir=f"{model_name}-test",
            evaluation_strategy="epoch",
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            weight_decay=self.config.weight_decay,
            save_total_limit=self.config.save_total_limit,
            num_train_epochs=self.config.epochs,
            predict_with_generate=True,
        )

        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        trainer = Seq2SeqTrainer(
            self.model,
            args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            #compute_metrics=compute_metrics
        )

        trainer.train()
