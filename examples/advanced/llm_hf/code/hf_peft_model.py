import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

class CausalLMPEFTModel(torch.nn.Module):
    def __init__(self, model_path):
        super(CausalLMModel, self).__init__()
        self.model_path = model_path

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
        )

    def forward(self, input_id):
        # PEFT configs
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, peft_config)
        output = self.model(input_ids=input_id, return_dict=False)
        return output
