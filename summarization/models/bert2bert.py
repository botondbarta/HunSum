from transformers import EncoderDecoderModel, BertTokenizer, IntervalStrategy, Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments

from summarization.models.base_model import BaseModel


class Bert2Bert(BaseModel):
    def __init__(self, config_path):
        super().__init__(config_path)

        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            "SZTAKI-HLT/hubert-base-cc", "SZTAKI-HLT/hubert-base-cc"
        )
        self.tokenizer = BertTokenizer.from_pretrained("SZTAKI-HLT/hubert-base-cc")

        # model.config.decoder_start_token_id = tokenizer.bos_token_id
        self.model.config.decoder_start_token_id = self.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size

    def process_data_to_model_inputs(self, batch):
        # Tokenize the input and target data
        inputs = self.tokenizer(batch['article'], padding='max_length',
                                truncation=True, max_length=512)
        outputs = self.tokenizer(batch['lead'], padding='max_length',
                                 truncation=True, max_length=512)

        batch['input_ids'] = inputs.input_ids
        batch['attention_mask'] = inputs.attention_mask
        # batch["decoder_input_ids"] = outputs.input_ids
        batch['decoder_attention_mask'] = outputs.attention_mask
        batch['labels'] = outputs.input_ids.copy()

        batch['labels'] = [[-100 if token == self.tokenizer.pad_token_id else token for token in labels] for labels in
                           batch['labels']]

        return batch

    def full_train(self):
        dataset = self.load_dataset(self.config.data_dir)
        tokenized_datasets = self.tokenize_datasets(dataset)

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            predict_with_generate=True,
            evaluation_strategy=IntervalStrategy.STEPS,
            eval_steps=self.config.valid_steps,
            save_steps=self.config.save_checkpoint_steps,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            save_total_limit=self.config.save_total_limit,
            warmup_steps=self.config.warmup_steps,
            fp16=True,
            # eval_accumulation_steps=30,
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
        )

        trainer.train()

# tokenized_datasets.set_format(
#     type="torch", columns=["input_ids", "attention_mask", "decoder_attention_mask", "labels"],
# )
