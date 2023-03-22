import torch.nn as nn
from transformers import AutoModelForTokenClassification, AutoTokenizer


class BertModel(nn.Module):
    def __init__(self, num_labels):
        super(BertModel, self).__init__()
        num_labels = num_labels
        self.model_name = "bert-base-uncased"
        self.bert = AutoModelForTokenClassification.from_pretrained(
            self.model_name, num_labels=num_labels, output_attentions=False, output_hidden_states=False
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def forward(self, input_id, mask, label):
        output = self.bert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)
        return output
