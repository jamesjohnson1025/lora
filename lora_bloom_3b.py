# Lora exercise


from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig,get_peft_model,PeftModel,PeftConfig
from datasets import load_dataset
import bitsandbytes as bnb 
import transformers
import torch.nn as nn
import torch 

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

# The key takeaway here is that CastOutputToFloat is primarily designed to ensure the output of model.lm_head is in float32 format. 
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
    task_type="CAUSAL_LM"
)

# Find out the difference between pretrained model and the current model 
model = get_peft_model(model=model,peft_config=config)
print_trainable_parameters(model=model)


qa_dataset = load_dataset('squad_v2')


def create_prompt(context, question, answer):
    result = ""
    if len(answer['text']) < 1:
        result =  "I don't the answer"
    else:
        result = answer['text'][0]
    prompt_template = f"### CONTEXT\n{context}\n\n### QUESTION\n{question}\n\n### AMSWER\n{result}</s>"
    return prompt_template

mapped_dataset = qa_dataset.map(lambda samples: tokenizer(create_prompt(samples['context'],samples['question'],samples['answers'])))

# Understand the parameters once again
trainer = transformers.Trainer(
                model=model,
                train_dataset=mapped_dataset['train'],
                args=transformers.TrainingArguments(
                    per_device_eval_batch_size=4,
                    gradient_accumulation_steps=4,
                    warmup_steps=100,
                    max_steps=100,
                    num_train_epochs=3,
                    learning_rate=1e-3,
                    fp16=True,
                    logging_steps=1,
                    output_dir='outputs'
                ),
                data_collator=transformers.DataCollatorForLanguageModeling(tokenizer,mlm=False)
            )

# WHat is the use of cache here.
model.config.use_cache = False
trainer.train()


# Upload to hugging_face
model_name = "bloom7b__finetune_sample"
HUGGING_FACE_USER_NAME = "james92"

model.push_to_hub(f"{HUGGING_FACE_USER_NAME}/{model_name}", use_auth_token=True)

print("Model is saved in hggingface")