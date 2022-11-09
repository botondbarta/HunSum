import glob
import os
from abc import abstractmethod, ABC

import datasets
import pandas as pd
from datasets import DatasetDict, Dataset
from transformers import IntervalStrategy, Seq2SeqTrainingArguments, Seq2SeqTrainer, pipeline

from summarization.utils.config_reader import get_config_from_yaml


class BaseModel(ABC):
    def __init__(self, config_path):
        self.config = get_config_from_yaml(config_path)
        self.rouge = datasets.load_metric("rouge")

    @abstractmethod
    def process_data_to_model_inputs(self, batch):
        raise NotImplementedError

    def load_dataset(self, data_dir, shuffle=True):
        files = [data_dir] if os.path.isfile(data_dir) else glob.glob(f'{data_dir}/*.jsonl.gz')
        site_dfs = []
        for file in files:
            site_df = pd.read_json(file, lines=True)
            site_df = site_df[['lead', 'article']]
            site_df = self.drop_na_and_duplicates(site_df)
            site_df = site_df.astype('str')
            site_dfs.append(site_df)
        df = pd.concat(site_dfs)
        if shuffle:
            df = df.sample(frac=1, random_state=123)
        return Dataset.from_pandas(df)

    @staticmethod
    def drop_na_and_duplicates(df):
        df = df.dropna()
        df = df.drop_duplicates(subset='article')
        return df

    def tokenize_datasets(self, raw_datasets):
        return raw_datasets.map(self.process_data_to_model_inputs, batched=True, remove_columns=['article', 'lead'])

    @abstractmethod
    def get_seq2seq_trainer(self, training_args, tokenized_datasets) -> Seq2SeqTrainer:
        raise NotImplementedError

    def full_train(self):
        if self.config.do_preprocess:
            raw_datasets = DatasetDict()
            raw_datasets['train'] = self.load_dataset(self.config.train_dir)
            raw_datasets['validation'] = self.load_dataset(self.config.valid_dir)
            raw_datasets['test'] = self.load_dataset(self.config.test_dir, shuffle=False)
            tokenized_datasets = self.tokenize_datasets(raw_datasets)
            if self.config.save_tokenized_data:
                tokenized_datasets.save_to_disk(self.config.preprocessed_dataset_path)
        else:
            tokenized_datasets = DatasetDict.load_from_disk(self.config.preprocessed_dataset_path)

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
            generation_max_length=self.config.max_predict_length,
            generation_num_beams=self.config.num_beams,
            warmup_steps=self.config.warmup_steps,
            load_best_model_at_end=True,
            fp16=self.config.fp16,
            # eval_accumulation_steps=30,
        )

        trainer = self.get_seq2seq_trainer(training_args, tokenized_datasets)
        trainer.compute_metrics = self.compute_metrics

        # Training
        checkpoint = self.config.resume_from_checkpoint if self.config.resume_from_checkpoint else None
        train_output = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_output.metrics
        trainer.save_model(os.path.join(self.config.output_dir, 'best_model'))
        trainer.save_metrics("train", metrics)

        # Evaluation
        metrics = trainer.evaluate(
            metric_key_prefix="eval",
            max_length=self.config.max_predict_length,
            num_beams=self.config.num_beams,
            length_penalty=self.config.length_penalty,
            no_repeat_ngram_size=self.config.no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=self.config.encoder_no_repeat_ngram_size,
            early_stopping=self.config.generate_early_stopping,
        )

        trainer.save_metrics("eval", metrics)

        # Prediction
        test_output = trainer.predict(
            test_dataset=tokenized_datasets["test"],
            metric_key_prefix="test",
            max_length=self.config.max_predict_length,
            num_beams=self.config.num_beams,
            length_penalty=self.config.length_penalty,
            no_repeat_ngram_size=self.config.no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=self.config.encoder_no_repeat_ngram_size,
            early_stopping=self.config.generate_early_stopping,
        )

        metrics = test_output.metrics
        trainer.save_metrics("predict", metrics)

        predictions = test_output.predictions
        predictions[predictions == -100] = self.tokenizer.pad_token_id
        test_preds = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        test_preds = list(map(str.strip, test_preds))

        if self.config.prediction_file is not None:
            output_file = os.path.join(self.config.output_dir, self.config.prediction_file)
        else:
            output_file = os.path.join(self.config.output_dir, "test_generations.txt")
        with open(output_file, 'w+') as f:
            for ln in test_preds:
                f.write(ln + "\n\n")

    def predict_pipeline(self, text):
        nlp = pipeline(model=self.model, task='summarization', tokenizer=self.tokenizer)
        return nlp(text,
                   max_length=self.config.max_predict_length,
                   num_beams=self.config.num_beams,
                   length_penalty=self.config.length_penalty,
                   no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                   encoder_no_repeat_ngram_size=self.config.encoder_no_repeat_ngram_size,
                   early_stopping=self.config.generate_early_stopping,
                   )

    def compute_metrics(self, pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = self.tokenizer.pad_token_id
        label_str = self.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        rouge_output = self.rouge.compute(
            predictions=pred_str, references=label_str, rouge_types=["rouge2"]
        )["rouge2"].mid

        return {
            "rouge2_precision": round(rouge_output.precision, 4),
            "rouge2_recall": round(rouge_output.recall, 4),
            "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
        }
