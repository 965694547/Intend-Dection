import paddle
import paddlenlp as ppnlp

from prefix_encoder import PrefixEncoder

from paddlenlp.transformers import ErnieModel,ErniePretrainedModel

class PromptTuningV2(ErniePretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.ernie = ErnieModel(config)
        self.dropout = paddle.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = paddle.nn.Linear(config.hidden_size, config.num_labels)

        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = paddle.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config)

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.ernie.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
    ):
        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = paddle.ones(batch_size, self.pre_seq_len).to(self.ernie.device)
        attention_mask = paddle.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.ernie(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            past_key_values=past_key_values,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        total_loss = 0
        if labels is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = paddle.nn.MSELoss()
                intent_loss = intent_loss_fct(logits.squeeze(), labels.squeeze())
            else:
                intent_loss_fct = paddle.nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            total_loss += intent_loss

        output = (logits,) + outputs[2:]
        return ((total_loss,) + output)