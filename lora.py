import os 
import torch 
import torch.nn as nn
import bitsandbytes as bnb 
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import LoraConfig,get_peft_model


# check if cuda is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'


model = AutoModelForCausalLM.from_pretrained(
    'bigscience/bloom-3b',
    torch_dtype=torch.float16,
    device_map='auto'
)

tokenizer = AutoTokenizer.from_pretrained("bigscience/tokenizer")

#print the model 
print(model)


for param in model.parameters():
    param.requires_grad=False 

    if param.ndim == 1: # cast the small parameters to fp32 for stability 
        param.data = param.data.to(torch.float32)
    

model.gradient_checkpointing_enable()
model.enable_input_require_grads()


class CastOutputToFloat(nn.Sequential):
    def forward(self,x): return super().forward(x).to(torch.float32)

model.lm_head = CastOutputToFloat(model.lm_head)



# print trainable parameters 

def print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0

    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(f'Trainable Params:{trainable_params}, All Params:{all_params}, trainable % {100 * (trainable_params/all_params)}')



# obtain Lora config 
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=['query_key_value'],
    lora_dropout=0.05,
    bias="none",
    task_type="CASUAL_LM"
)

model = get_peft_model(model=model,peft_config=config)
print_trainable_parameters(model=model)


