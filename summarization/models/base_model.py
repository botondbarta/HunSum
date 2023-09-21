import glob
import os
from abc import abstractmethod, ABC

import datasets
import pandas as pd
from datasets import DatasetDict, Dataset
from transformers import IntervalStrategy, Seq2SeqTrainingArguments, Seq2SeqTrainer, pipeline, EarlyStoppingCallback

from summarization.utils.config_reader import get_config_from_yaml


class BaseModel(ABC):
    def __init__(self, config_path):
        self.config = get_config_from_yaml(config_path)
        if self.config.compute_training_metrics:
            self.rouge = datasets.load_metric("rouge")
            self.bert_score = datasets.load_metric("bertscore")

    @abstractmethod
    def process_data_to_model_inputs(self, batch):
        raise NotImplementedError

    def load_dataset(self, data_dir, shuffle=True):
        files = [data_dir] if os.path.isfile(data_dir) else sorted(glob.glob(f'{data_dir}/*.jsonl.gz'))
        site_dfs = []
        for file in files:
            site_df = pd.read_json(file, lines=True)
            site_df = site_df[['lead', 'article', 'uuid']]
            site_df = site_df.dropna()
            site_df = site_df.astype('str')
            site_dfs.append(site_df)
        df = pd.concat(site_dfs)
        if shuffle:
            df = df.sample(frac=1, random_state=123)
        return Dataset.from_pandas(df)

    def tokenize_datasets(self, raw_datasets):
        return raw_datasets.map(self.process_data_to_model_inputs, batched=True, remove_columns=['article', 'lead'])

    @abstractmethod
    def get_seq2seq_trainer(self, tokenized_datasets, load_train_data=True) -> Seq2SeqTrainer:
        raise NotImplementedError

    def _get_seq2seq_training_args(self):
        return Seq2SeqTrainingArguments(
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
            metric_for_best_model=self.config.metric_for_best_model if self.config.compute_training_metrics else 'loss',
        )

    def _load_tokenized_dataset(self, load_train: bool = True, load_test: bool = True):
        if self.config.do_preprocess:
            raw_datasets = DatasetDict()
            if load_train:
                raw_datasets['train'] = self.load_dataset(self.config.train_dir)
                raw_datasets['validation'] = self.load_dataset(self.config.valid_dir)
            if load_test:
                raw_datasets['test'] = self.load_dataset(self.config.test_dir, shuffle=False)
            tokenized_datasets = self.tokenize_datasets(raw_datasets)
            if self.config.save_tokenized_data:
                tokenized_datasets.save_to_disk(self.config.preprocessed_dataset_path)
        else:
            tokenized_datasets = DatasetDict.load_from_disk(self.config.preprocessed_dataset_path)
        return tokenized_datasets

    def train(self):
        tokenized_datasets = self._load_tokenized_dataset(load_train=True, load_test=True)

        trainer = self.get_seq2seq_trainer(tokenized_datasets, load_train_data=True)
        if self.config.compute_training_metrics:
            trainer.compute_metrics = self.compute_metrics
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=self.config.patience))

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

        self._generate_and_save(trainer, tokenized_datasets)

    def generate(self):
        raw_datasets = DatasetDict()
        raw_datasets['test'] = self.load_dataset(self.config.generate_dir, shuffle=False)
        tokenized_datasets = self.tokenize_datasets(raw_datasets)

        trainer = self.get_seq2seq_trainer(tokenized_datasets, load_train_data=False)
        trainer.compute_metrics = self.compute_metrics
        self._generate_and_save(trainer, tokenized_datasets)

    def _generate_and_save(self, trainer: Seq2SeqTrainer, tokenized_datasets):
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

        test_df = tokenized_datasets['test'].to_pandas()
        test_df['generated_lead'] = test_preds
        test_df = test_df[['generated_lead', 'uuid']]

        output_file = os.path.join(self.config.output_dir, self.config.prediction_file)

        with open(output_file, 'w', encoding='utf-8') as file:
            test_df.to_json(file, force_ascii=False, lines=True, orient='records')

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
            predictions=pred_str, references=label_str, rouge_types=["rouge1", "rouge2", "rougeL"]
        )

        bert_scores = self.bert_score.compute(predictions=pred_str, references=label_str,
                                              model_type="SZTAKI-HLT/hubert-base-cc", num_layers=8)

        rouge1 = rouge_output["rouge1"].mid
        rouge2 = rouge_output["rouge2"].mid
        rougeL = rouge_output["rougeL"].mid

        avg = lambda x: sum(x) / len(x)

        return {
            "rouge1_precision": round(rouge1.precision, 4),
            "rouge1_recall": round(rouge1.recall, 4),
            "rouge1_fmeasure": round(rouge1.fmeasure, 4),
            "rouge2_precision": round(rouge2.precision, 4),
            "rouge2_recall": round(rouge2.recall, 4),
            "rouge2_fmeasure": round(rouge2.fmeasure, 4),
            "rougeL_precision": round(rougeL.precision, 4),
            "rougeL_recall": round(rougeL.recall, 4),
            "rougeL_fmeasure": round(rougeL.fmeasure, 4),
            "bert_score_precision": round(avg(bert_scores['precision']), 4),
            "bert_score_recall": round(avg(bert_scores['recall']), 4),
            "bert_score_f1": round(avg(bert_scores['f1']), 4),
        }
