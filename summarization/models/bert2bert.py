from transformers import EncoderDecoderModel, BertTokenizer, Seq2SeqTrainer

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

    def get_seq2seq_trainer(self, training_args, tokenized_datasets) -> Seq2SeqTrainer:
        return Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
        )

# tokenized_datasets.set_format(
#     type="torch", columns=["input_ids", "attention_mask", "decoder_attention_mask", "labels"],
# )
