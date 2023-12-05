from peft import LoraConfig,PeftModel,PeftConfig
from transformers import AutoTokenizer,AutoModelForCausalLM

model_name = "bloom7b__finetune_sample"
HUGGING_FACE_USER_NAME = "james92"

# Do the inference 
peft_model_id = f'{HUGGING_FACE_USER_NAME}/{model_name}'
config = PeftConfig.from_pretrained(peft_model_id)
inference_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,return_dict=True,load_in_8bit=False,device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Do the inference 
qa_model = PeftModel.from_pretrained(inference_model,peft_model_id)

# Print the model.
print(qa_model)