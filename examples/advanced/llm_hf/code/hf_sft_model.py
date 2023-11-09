import torch
from transformers import AutoModelForCausalLM


class CausalLMModel(torch.nn.Module):
    def __init__(self, model_path):
        super(CausalLMModel, self).__init__()
        self.model_path = model_path
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
        )

    def forward(self, input_id):
        output = self.model(input_ids=input_id, return_dict=False)
        return output
