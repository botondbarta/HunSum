import logging
import os.path

from datasets import DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, \
    Seq2SeqTrainer
from transformers import IntervalStrategy

from summarization.models.base_model import BaseModel

logger = logging.getLogger(__name__)


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
        #with self.tokenizer.as_target_tokenizer():
        outputs = self.tokenizer(text_target=batch['lead'], padding='max_length', max_length=self.config.mt5.max_output_length,
                                     truncation=True)

        outputs["input_ids"] = [
            [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in outputs["input_ids"]
        ]
        inputs['labels'] = outputs['input_ids']
        return inputs

    def full_train(self):
        if self.config.do_preprocess:
            raw_datasets = DatasetDict()
            if self.config.do_train:
                raw_datasets['train'] = self.load_dataset(self.config.train_dir)
                raw_datasets['validation'] = self.load_dataset(self.config.valid_dir)

            if self.config.do_predict:
                raw_datasets['test'] = self.load_dataset(self.config.test_dir)
            tokenized_datasets = self.tokenize_datasets(raw_datasets)
            tokenized_datasets.save_to_disk(self.config.preprocessed_dataset_path)
        else:
            tokenized_datasets = DatasetDict.load_from_disk(self.config.preprocessed_dataset_path)

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
            load_best_model_at_end=True,
            warmup_steps=self.config.warmup_steps,
        )

        label_pad_token_id = -100
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=label_pad_token_id,
        )

        trainer = Seq2SeqTrainer(
            self.model,
            args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        if self.config.do_train:
            checkpoint = None
            if self.config.resume_from_checkpoint is not None:
                checkpoint = self.config.resume_from_checkpoint
            trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model(os.path.join(self.config.output_dir, 'best_model'))

        if self.config.do_predict:
            test_output = trainer.predict(
                test_dataset=tokenized_datasets["test"],
                metric_key_prefix="test",
                max_length=self.config.mt5.max_output_length,
                num_beams=self.config.num_beams,
                length_penalty=self.config.length_penalty,
                no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
            )

            predictions = test_output.predictions
            predictions[predictions == -100] = self.tokenizer.pad_token_id
            test_preds = self.tokenizer.batch_decode(
                predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            test_preds = list(map(str.strip, test_preds))
            with open(os.path.join(self.config.output_dir, "test_generations.txt"), 'w+') as f:
                for ln in test_preds:
                    f.write(ln + "\n\n")

