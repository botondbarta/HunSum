import logging
import os.path
from pathlib import Path

import evaluate
import nltk
import numpy as np
import pandas as pd
from rouge_score import rouge
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, \
    Seq2SeqTrainer, pipeline
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
        dataset = self.load_dataset(self.config.data_dir)
        tokenized_datasets = self.tokenize_datasets(dataset)

        if self.model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

        if (
                hasattr(self.model.config, "max_position_embeddings")
                and self.model.config.max_position_embeddings < self.config.mt5.max_input_length
        ):
            if True:
                logger.warning(
                    "Increasing the model's number of position embedding vectors from"
                    f" {self.model.config.max_position_embeddings} to {self.config.mt5.max_input_length}."
                )
                self.model.resize_position_embeddings(self.config.mt5.max_input_length)


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

        label_pad_token_id = -100
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=label_pad_token_id,
        )

        #data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        # Metric
        metric = evaluate.load("rouge")

        def postprocess_text(preds, labels):
            preds = [pred.strip() for pred in preds]
            labels = [label.strip() for label in labels]

            # rougeLSum expects newline after each sentence
            preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
            labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

            return preds, labels

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
            decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            # Some simple post-processing
            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

            result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
            result = {k: round(v * 100, 4) for k, v in result.items()}
            prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
            result["gen_len"] = np.mean(prediction_lens)
            return result

        trainer = Seq2SeqTrainer(
            self.model,
            args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            load_best_model_at_end=True,
            compute_metrics=compute_metrics
        )

        trainer.train()
        trainer.save_model(os.path.join(self.config.output_dir, 'best_model'))

        test_output = trainer.predict(
            test_dataset=tokenized_datasets["test"],
            metric_key_prefix="test",
            max_length=self.config.mt5.max_output_length,
            num_beams=1,
            #length_penalty=data_args.length_penalty,
            #no_repeat_ngram_size=data_args.no_repeat_ngram_size,
        )

        predictions = test_output.predictions
        predictions[predictions == -100] = self.tokenizer.pad_token_id
        test_preds = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        test_preds = list(map(str.strip, test_preds))
        with open(os.path.join(self.config.output_dir, "test_generations.txt"), 'w+') as f:
            for ln in test_preds:
                f.write(ln + "\n")

    def inference(self, model_dir, data_file):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir).to("cuda")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_checkpoint)

        dataset = self.load_dataset(data_file)
        tokenized_datasets = self.tokenize_datasets(dataset)

        def generate_summary(batch):
            inputs = self.tokenizer(batch['article'], padding='max_length', max_length=self.config.mt5.max_input_length,
                                    truncation=True, return_tensors="pt")
            input_ids = inputs.input_ids.to("cuda")
            # attention_mask = inputs.attention_mask.to("cuda")

            outputs = self.model.generate(input_ids)#, attention_mask=attention_mask)
            with self.tokenizer.as_target_tokenizer():
                output_str = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            #output_str = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            batch["pred_summary"] = output_str
            return batch

        batch_size = 16  # change to 64 for full evaluation

        results = tokenized_datasets.map(generate_summary, batched=True, batch_size=batch_size, remove_columns=["article"])

        print(results['pred_summary'])

        r = rouge.compute(predictions=results["pred_summary"], references=results["highlights"], rouge_types=["rouge2"])[
            "rouge2"].mid

        print(r)







