import os

import datasets
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import EncoderDecoderModel, BertTokenizer, IntervalStrategy, Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments

from summarization.models.base_model import BaseModel


class Bert2Bert(BaseModel):
    def __init__(self, config_path):
        super().__init__(config_path)

        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            self.config.bert2bert.tokenizer, self.config.bert2bert.tokenizer
        )
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert2bert.tokenizer)

        self.model.config.decoder_start_token_id = self.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.sep_token_id
        self.model.config.vocab_size = self.model.config.encoder.vocab_size

    def process_data_to_model_inputs(self, batch):
        # Tokenize the input and target data
        inputs = self.tokenizer(batch['article'], padding='max_length',
                                truncation=True, max_length=512)
        outputs = self.tokenizer(batch['lead'], padding='max_length',
                                 truncation=True, max_length=512)

        batch['input_ids'] = inputs.input_ids
        batch['attention_mask'] = inputs.attention_mask
        # batch["decoder_input_ids"] = outputs.input_ids
        # batch['decoder_attention_mask'] = outputs.attention_mask
        batch['labels'] = outputs.input_ids.copy()

        batch['labels'] = [[-100 if token == self.tokenizer.pad_token_id else token for token in labels]
                           for labels in batch['labels']]

        return batch

    def full_train(self):
        raw_datasets = DatasetDict()
        if self.config.do_train:
            raw_datasets['train'] = self.load_dataset(self.config.train_dir)
            raw_datasets['valid'] = self.load_dataset(self.config.valid_dir)

        if self.config.do_predict:
            raw_datasets['test'] = self.load_dataset(self.config.test_dir)

        tokenized_datasets = self.tokenize_datasets(raw_datasets)

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config.output_dir,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            evaluation_strategy=IntervalStrategy.STEPS,
            weight_decay=self.config.weight_decay,
            save_total_limit=self.config.save_total_limit,
            eval_steps=self.config.valid_steps,
            save_steps=self.config.valid_steps,
            predict_with_generate=True,
            warmup_steps=self.config.warmup_steps,
            fp16=True,
            load_best_model_at_end=True,
            # eval_accumulation_steps=30,
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
        )

        trainer.train()
        trainer.save_model(f'{self.config.output_dir}/best_model')

# tokenized_datasets.set_format(
#     type="torch", columns=["input_ids", "attention_mask", "decoder_attention_mask", "labels"],
# )
