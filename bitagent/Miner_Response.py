import torch
from transformers import pipeline
# from fm_mf import count_unique_pet_names
from fuzzywuzzy import fuzz
# from Model_Test_Miner import Json_To_Vector, Response
import spacy
import re
from PalChain import Logical_Initiatoin
from bitagent.Orca import generate_answer
summarizer = 0

clf = 0

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


def read_word_list(file_path):
    with open(file_path, "r") as file:
        return [line.strip().lower() for line in file]

def count_unique__names(input_str):
    nlp = spacy.load("en_core_web_sm")

    # Load male and female words
    male_words = read_word_list("male.txt")
    female_words = read_word_list("female.txt")

    # Determine gender based on input string
    if "male" in input_str.lower():
        pet_words = male_words
    elif "female" in input_str.lower():
        pet_words = female_words
    else:
        return "Invalid input format."

    # Extract names from input string
    names = [name.strip() for name in input_str.split(":")[1].split(",")]

    unique_pet_names = set(names)
    pet_name_count = 0

    for name in unique_pet_names:
        doc = nlp(name)
        for token in doc:
            if token.text.lower() in pet_words:
                pet_name_count += 1
                break

    return pet_name_count
    
# pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta", torch_dtype=torch.bfloat16, device_map="auto")

# Threshold for considering strings similar
threshold = 60

def similarity(a, b):
    return fuzz.ratio(a, b)

def Summary_process(text):
    processed = summarizer(text)
    return processed[0]['summary_text']

def Logic_process(text):
    # if similarity(text, "Given the following list of pet names, determine the number of unique male pet names:Charlie ,Max ,Buddy Bella ,Charlie,Cooper ,Whiskers.") >= threshold:
    #     print("male_female count")
    #     processed = count_unique__names(text)
    #     return processed
    # else:
    #     Json_To_Vector("Logic.json")
    #     semantic_search = Response(text)
        # messages = [
        #     {
        #         "role": "system",
        #         "content": "I am a genius bot who is tasked with analyzing and responding accurately to queries related to mathematics providing numerical values based on the question asked. and use relevant data:{semantic_search}",
        #     },
        #     {
        #         "role": "system",
        #         "content": "My context length will be less than 6 numeric (final numeric answer) tokens.",
        #     },
        #     {"role": "user", "content": f"Given the input '{text}', understand the {semantic_search} and using your on logic  generate an output  numerical response independently."},
        #     {"role": "user", "content": f"If a mathematical question of any type is asked, just give the numerical answer to that without mentioning its steps."},
        # ]
        # input_ids = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # outputs = pipe(input_ids, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        # text = outputs[0]["generated_text"]
        # text = text.split(' ')  # Fixing the split
        # result = text[-1]  # Unused variable?
        
        
        
        # last = extract_last_numeric_value(result)
        # print(last_numeric_value)  
        # processed = result
        result = generate_answer(text)
        return result

def Miner_Model(text):
    label_names = {0: 'Summary', 1: 'Logic', 2: 'Greets'}
    # output = clf(text)
    # predictions = output[0]
    # summary_score = 0
    # logic_score = 0
    # for prediction in predictions:
    #     label_idx = int(prediction['label'].split('_')[-1])
    #     label_name = label_names[label_idx]
    #     score = prediction['score']

    #     if label_name == 'Summary':
    #         summary_score = score
    #     elif label_name == 'Logic':
    #         logic_score = score

    result = Logic_process(text)
    # if summary_score > logic_score:
    #     result = Summary_process(text)
    # elif logic_score > summary_score:

    return result

# Uncomment and use this main block if needed
# if __name__ == "__main__":
#     print("Start...........")
#     text = ['Anna: where are you Omenah: at home Anna: I will be there in a minute ',
#             'How many different male pet names are listed here: Duke, Max, Buddy, Bella, Duke, Cooper, Whiskers, Bella?',
#             'Simply provide the numerical value representing the count of unique female pet names within the given list: Lorraine, Howard, Caroline, Austin.',
#             'Olivia: Who are you voting for in this election? Oliver: Liberals as always. Olivia: Me too!! Oliver: Great',
#             'Given the list of pet names below, how many unique male pet names are there: Daisy, Max, Buddy, Luna, Daisy, Cooper, Whiskers?']
#
#     for i in range(len(text)):
#         print(f"Query : {text[i]}")
#         print(f"Response : {Miner_process(text[i])}")
#         print("Finished..........")
