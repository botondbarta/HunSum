import logging

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer

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
        # with self.tokenizer.as_target_tokenizer():
        outputs = self.tokenizer(text_target=batch['lead'], padding='max_length',
                                 max_length=self.config.mt5.max_output_length,
                                 truncation=True)

        outputs["input_ids"] = [
            [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in outputs["input_ids"]
        ]
        inputs['labels'] = outputs['input_ids']
        return inputs

    def get_seq2seq_trainer(self, training_args, tokenized_datasets) -> Seq2SeqTrainer:
        label_pad_token_id = -100
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=label_pad_token_id,
        )

        return Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
