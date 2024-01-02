from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM,BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline
from peft import PeftModel
import os, torch, logging
from transformers import GenerationConfig

pt_model = AutoModelForCausalLM.from_pretrained(
    "Jacaranda/UlizaLlama",
    device_map="auto",
    return_dict=True,
    torch_dtype='auto',
    use_cache=True,
    offload_folder="pt_model"
)
tokenizer = AutoTokenizer.from_pretrained("Jacaranda/UlizaLlama")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

peft_path = "lora_model"
peft_model= PeftModel.from_pretrained(
    pt_model,
    peft_path,
    torch_dtype=torch.float16
)

text_question = "The glass water motor will not turn, and the wiper will not move."

prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

  ### Instruction:
  {}

  ### Response:""".format(text_question)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')

generation_output = peft_model.generate(
      input_ids=input_ids,
      generation_config=GenerationConfig(do_sample=True,temp=0.2,top_p= 0.75,num_beams=1),
      max_new_tokens=256,

  )

def extract_response_text(input_string):
    start_marker = '### Response:'
    end_marker = '###'

    start_index = input_string.find(start_marker)
    if start_index == -1:
        return None

    start_index += len(start_marker)

    end_index = input_string.find(end_marker, start_index)
    if end_index == -1:
        return input_string[start_index:]

    return input_string[start_index:end_index].strip()


print("Question: "+text_question+'\n')
print("Generated Response: " +extract_response_text(tokenizer.decode(generation_output[0])))