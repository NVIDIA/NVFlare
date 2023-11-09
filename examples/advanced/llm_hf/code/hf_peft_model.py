import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class CausalLMPEFTModel(torch.nn.Module):
    def __init__(self, model_path):
        super(CausalLMPEFTModel, self).__init__()
        self.model_path = model_path
        # bitsandbytes
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        # PEFT configs
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )
        full_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=bnb_config,
        )
        full_model = prepare_model_for_kbit_training(full_model)
        self.model = get_peft_model(full_model, peft_config)

    def forward(self, input_id):
        output = self.model(input_ids=input_id, return_dict=False)
        return output
