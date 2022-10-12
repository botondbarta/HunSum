import os.path

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, \
    Seq2SeqTrainer, pipeline
from transformers import IntervalStrategy

from summarization.models.base_model import BaseModel


class MT5(BaseModel):
    def __init__(self, config_path):
        super().__init__(config_path)

        self.model_checkpoint = self.config.mt5.model_checkpoint

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_checkpoint)

    def process_data_to_model_inputs(self, batch):
        # Tokenize the input and target data
        inputs = self.tokenizer(batch['article'], padding='max_length', max_length=self.config.mt5.max_input_length,
                                truncation=True)
        with self.tokenizer.as_target_tokenizer():
            outputs = self.tokenizer(batch['lead'], padding='max_length', max_length=self.config.mt5.max_output_length,
                                     truncation=True)

        inputs['labels'] = outputs['input_ids']
        return inputs

    def full_train(self):
        dataset = self.load_dataset(self.config.data_dir)
        tokenized_datasets = self.tokenize_datasets(dataset)

        args = Seq2SeqTrainingArguments(
            output_dir=self.config.output_dir,
            evaluation_strategy=IntervalStrategy.STEPS,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            weight_decay=self.config.weight_decay,
            save_total_limit=self.config.save_total_limit,
            num_train_epochs=self.config.num_train_epochs,
            save_steps=self.config.save_checkpoint_steps,
            eval_steps=self.config.save_checkpoint_steps,
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
            load_best_model_at_end=True,
        )

        trainer.train()
        trainer.save_model(os.path.join(self.config.output_dir, 'best_model'))

    def inference(self, model_dir, data_file):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

        articles_df = pd.read_json(data_file, lines=True)

        articles = articles_df.article.tolist()

        summarizer = pipeline("summarization", model=self.model, tokenizer=self.tokenizer, framework="pt")
        leads = summarizer(articles, min_length=5, max_length=self.config.mt5.max_output_length)

        for lead in leads:
            print(lead)







