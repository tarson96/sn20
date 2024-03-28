import random
from typing import List
import transformers
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from sentence_transformers.cross_encoder import CrossEncoder
import re

model_name_or_path = "TheBloke/Mistral-7B-OpenOrca-AWQ"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)
model = AutoAWQForCausalLM.from_quantized(model_name_or_path, fuse_layers=True, trust_remote_code=False, safetensors=True)

cross_encoder = CrossEncoder("cross-encoder/stsb-distilroberta-base")

def validator_llm(input_text):
        text = f'''
<|im_start|>user
{input_text}<|im_end|>
<|im_start|>assistant
'''
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
        outputs = model.generate(input_ids, max_new_tokens=160, temperature=0.7, top_k=40, top_p=0.95, do_sample=True, repetition_penalty=1.1)
        result = tokenizer.decode(outputs[0])
        result = result.split("<|im_start|> assistant\n")[-1].replace("<|im_end|>","").strip()
        return result

def extract_last_numeric_value(text):
    # Define a regular expression pattern to match numeric values (integers and floats)
    pattern = r'\b\d+\b'  # This pattern matches integers

    # Use re.findall to find all matches of the pattern in the text
    matches = re.findall(pattern, text)

    if matches:
        # Convert the matched value to an integer and store it
        last_match = str(matches[-1])
        return last_match

    return None

def generate_answer(prompt: str) -> str:
    # Implement logic to generate answer based on prompt
    # Example logic:
    if "pet names" in prompt:
       question_string_lower = prompt.lower()
       if "female" in question_string_lower:
            gender = "female"
            x =  validator_llm(f"Answer this : {prompt}(Give numerical value only)")
            x = extract_last_numeric_value(x)
            print(x)
            answer = x
            # answer = validator_llm(f"Count {gender} names from this {x} (pass only numerical)")
       elif "male" in question_string_lower:
            gender = "male"

            x =  validator_llm(f"Answer this : {prompt}(Give numerical value only)")
            x = extract_last_numeric_value(x)
            print(x)
            answer = x
            # answer = validator_llm(f"Count {gender} names from this {x} (pass only numerical)")
    elif "HTML table" in prompt:
        formula = """Initialize a variable count to 0.
        Iterate through each cell in the grid.
        If the current cell is a 1 and has not been visited yet:
        a. Increment count by 1.
        b. Use Depth-First Search (DFS) or Breadth-First Search (BFS) to traverse all connected cells of the current island, marking them as visited.
        Continue iterating until all cells have been visited.
        Return the final value of count, which represents the total number of islands in the grid."""
        x =  validator_llm(f"Answer this : {prompt}(Give numerical value only)")
        x = extract_last_numeric_value(x)
        print(x)
        answer = x
    elif " synonyms" in prompt:
        x =  validator_llm(f"Answer this : {prompt}(Give numerical value only)")
        x = extract_last_numeric_value(x)
        print(x)
        answer = x
    elif "tricks" in prompt:
        x =  validator_llm(f"Answer this : {prompt}(Give numerical value only)")
        x = extract_last_numeric_value(x)
        print(x)
        answer = x
    elif "How many" in prompt:
        x =  validator_llm(f"Answer this : {prompt}(Give numerical value only)")
        x = extract_last_numeric_value(x)
        print(x)
        answer = x
    elif "count of islands" in prompt:
        x =  validator_llm(f"Answer this : {prompt}(Give numerical value only)")
        x = extract_last_numeric_value(x)
        print(x)
        answer = x
    else:
        x =  validator_llm(f"Answer this : {prompt}(Give numerical value only)")
        x = extract_last_numeric_value(x)
        print(x)
        answer = x

    return str(answer)
