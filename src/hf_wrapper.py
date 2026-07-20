from torch import nn


class HFWrapper(nn.Module):
    def __init__(self, model, pad_token_id=None):
        super().__init__()
        self.model = model
        self.pad_token_id = pad_token_id

    def forward(self, input_ids, attention_mask=None, labels=None,
                past_key_values=None, use_cache=False):
        if attention_mask is None and self.pad_token_id is not None and past_key_values is None:
            attention_mask = (input_ids != self.pad_token_id).long()
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            labels=labels, past_key_values=past_key_values, use_cache=use_cache,
        )
        if labels is not None or use_cache:
            return outputs          # training or cache-aware generation
        return outputs.logits       # classification raw tensor