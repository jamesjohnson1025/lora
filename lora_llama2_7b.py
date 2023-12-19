# T o load the dataset 
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser,TrainingArguments, pipeline
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import torch


# Setting up the model and tokenizer 
data_name = "mlabonne/guanaco-llama2-1k"
training_data = load_dataset(data_name,split='train')

# Model and tokenizer names 
base_model_name = "NousResearch/Llama-2-7b-chat-hf"


#Tokenizer

llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = 'right'

# Quantization Config 

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)

# Model 
base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config = quant_config,
        device_map='auto'
)
base_model.config.use_cache=False
base_model.config.pretraining_tp=1 # tensor parallelism rank

''''
Double quantization is a technique where weights are quantized twice with different quantization parameters, 
potentially improving the accuracy of the quantized model. However, it may also increase computational complexity.
'''

'''
LoRA-Specific Parameters

Dropout Rate (lora_dropout): This is the probability that each neuron’s output is set to zero during training, used to prevent overfitting.
Rank (r): Rank is essentially a measure of how the original weight matrices are broken down into simpler, smaller matrices. This reduces 
        computational requirements and memory consumption. Lower ranks make the model faster but might sacrifice performance. The original LoRA paper 
        suggests starting with a rank of 8, but for QLoRA, a rank of 64 is required.

lora_alpha: This parameter controls the scaling of the low-rank approximation. It’s like a balancing act between the original model and the low-rank approximation. 
        Higher values might make the approximation more influential in the fine-tuning process, affecting both performance and computational cost.

'''

# Lora Config 
peft_config = LoraConfig(lora_alpha=16,
                         lora_dropout=0.1,
                         r=8,
                         bias='none',
                         task_type='CAUSAL_LM')


# Training args 
train_params = TrainingArguments(
    output_dir="./",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant"
)

# Trainer 
fine_tuning = SFTTrainer(
    model=base_model,
    train_dataset=training_data,
    peft_config=peft_config,
    dataset_text_field="text",
    tokenizer=llama_tokenizer,
    args=train_params
)

# call the train function 
fine_tuning.train()

# save  the model 
fine_tuning.save_model("llama_7b_james")