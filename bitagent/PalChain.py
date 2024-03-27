from transformers import AutoTokenizer, AutoModelForCausalLM ,LlamaForCausalLM
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from transformers import pipeline

import re

#tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True)
#model = LlamaForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
# prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
T5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
T5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", device_map="auto", torch_dtype=torch.float16)
print("1")
# def extract_last_numeric_value(text):
#     # Define a regular expression pattern to match numeric values (integers and floats)
#     pattern = r'\b\d+\b'  # This pattern matches integers

#     # Use re.findall to find all matches of the pattern in the text
#     matches = re.findall(pattern, text)

#     if matches:
#         # Convert the matched value to an integer and store it
#         last_match = str(matches[-1])
#         return last_match

#     return None
print("2")
def Logical_Initiatoin(input_text):
      # "Making Programming Logic"
 #     tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True)
  #    model = LlamaForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
    # # text = 

   #   messages=[
    #      { 'role': 'user', 'content': f"write a function logically after understandinig this '{input_text}' "}
     # ]

      #inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

      # tokenizer.eos_token_id is the id of <|EOT|> token

      #outputs = model.generate(inputs, max_new_tokens=512, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)

      # print(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))

      #Solutions = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

      #print(Solutions)
      print("generating_Anwer....")
    #   pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta", torch_dtype=torch.bfloat16, device_map="auto")

    #   messages = [
    #   {
    #     "role": "system",
    #     "content": """Welcome! You're interacting with a mathematical chatbot that provides numeric solutions. Please adhere to the following guidelines:
    # - Only provide numerical answers.
    # - Ensure the answer consists of less than 10 tokens.
    # - Avoid adding any context or steps.
    # - For example, if asked "What is 3 + 2?", respond with "5" only.
    #   """,
    #   },
    #   {"role": "user", "content": f"Use '{Solutions}' to solve the following problem: '{text}'. Provide the numerical answer."},
    #   ]     
    
      # outputs = pipe(prompt, max_new_tokens=200, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
      # print(outputs[0]["generated_text"])     
      # split_text = text.split('</s>')
      # text = split_text[-1].strip()
      # print(f"Text_Split : {text} ") 
      # result = text  # Unused variable?
      # print(f"result: {result}")
      # last = extract_last_numeric_value(outputs[0]["generated_text"])
      # text = f"Apply the provided solution '{Solutions}' to address the question '{input_text}', ensuring that the output remains strictly numerical without any accompanying context."
     
     # text = f"""use this  {Solutions} to answer this {input_text} question.Please adhere to the following guidelines:
      #       - Only provide numerical answers.
       #      - Ensure the answer consists of no context.
        #     - Avoid adding any context or steps.
         #    - For example, if asked "What is 3 + 2?", respond with "5" only."""
    
      # text = f"Utilize the Solutions '{solution}' to respond to the question '{input_text}'. Ensure that the output is purely numerical answer and devoid of any contextual information."
      # text = f"use this  {Solutions} to answer this {input_text} question(Note that output should be numerical and context should not be provided in it)"
      text  = f"Answer this {input_text} and only pass the numerical value and no context should be given"
      input_ids = T5_tokenizer(text, return_tensors="pt").input_ids.to("cuda")
      outputs = T5_model.generate(input_ids)  
      # outputs = T5_model.generate(input_ids)
      last = T5_tokenizer.decode(outputs[0])
      text_2 = f"Only collect the numerical value from this {last}"
      input_ids = T5_tokenizer(text_2, return_tensors="pt").input_ids.to("cuda")
      outputs = T5_model.generate(input_ids)  
      # outputs = T5_model.generate(input_ids)
      last = T5_tokenizer.decode(outputs[0])

      print(last)  
      

      return last
# print("4")
if __name__ == "__main__":
     input_text = "If my age is half of my dad's age and he is going to be 60 next year"
     print("5")
     print(f"Answer : {Logical_Initiatoin(input_text)}")
