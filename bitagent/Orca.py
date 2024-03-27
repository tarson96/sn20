import random
from typing import List
import transformers
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from sentence_transformers.cross_encoder import CrossEncoder
import re
model_name_or_path = "TheBloke/Mistral-7B-OpenOrca-AWQ"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)
model = AutoAWQForCausalLM.from_quantized(model_name_or_path, fuse_layers=True,
                                              trust_remote_code=False, safetensors=True)

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

# Example usage:

if __name__ == "__main__":
   
prompt1 = """How many (provide numerical value only) words in this list are synonyms of desire: aspiration, eagerness, banana, craving, thirst, orchestra, yearning, pizza, enthusiasm, longing, hunger, parrot, zebra, penguin, wish, guitar, rainbow."
How many (provide numerical value only) animals are mentioned in the following list: lion, tiger, elephant, giraffe, crocodile, monkey, bear, rhinoceros, hippopotamus, koala, penguin, cheetah, panda, gorilla, kangaroo, wolf, deer, fox, sloth, ostrich, dolphin, seal, octopus, eagle, lizard, shark, whale, turtle, parrot, flamingo, camel, bat, squirrel, raccoon, otter, platypus, chameleon, hedgehog, armadillo, meerkat,
 flamingo, mongoose, buffalo, anteater, ibex, mongoose, komodo dragon, vulture, walrus, pelican, beaver."""

prompt2 = """Given the descriptions of tricks provided below:
1 - 'Bark' - The dog vocalizes on command, producing a bark. This trick is often used to demonstrate the dog's ability to respond vocally to cues from the handler.
2 - 'Sit' - The dog lowers its body to sit on its hindquarters while keeping the front paws straight. This basic obedience command is one of the first tricks dogs learn and is used to instill discipline and control.
3 - 'Rollover' - The dog starts from a lying position, rolls its body over to one side until it makes a complete 360-degree turn, and ends up back in the lying position. This trick showcases the dog's agility and willingness to follow more complex commands.
4 - 'Jump' - The dog leaps off the ground, usually over an obstacle or simply as a form of energetic expression. This trick is a good way to exercise the dog and improve its coordination and fitness.
5 - 'Bow' - The dog lowers its front body while keeping its rear end up, mimicking a bowing position. This trick is often used as a polite greeting or a sign of readiness to play.
6 - 'Shake' - The dog extends one of its front paws to the handler upon command, mimicking a handshake. This trick is popular as a way to show off the dog's friendly demeanor and ability to engage in social behaviors.
7 - 'Lie Down' - The dog moves from a standing or sitting position to lying flat on its belly with the legs extended. This command is fundamental in obedience training, helping in calming the dog or preparing it for more advanced tricks.

Please devise an innovative and intricate command to instruct the pet to perform the trick: 'Bark'.

Crafted Pet Command: "Convey your message through a series of rhythmic claps followed by a single prolonged whistle, akin to the sound of a distant train. Maintain a steady gaze, then with a swift gesture, extend your arm forward and upward, tracing a half-circle in the air, reminiscent of the moon's ascent. Accompany this movement with a gentle vocal cue, softly uttering 'Seranade'."

Which Trick ID (provide numerical number only) are you requesting? Trick ID: """

prompt3 = """Please provide only the numerical value indicating the count of islands (sequences of 1's) within the given 2D grid of 0s and 1s from below:

[[1, 0, 0, 0, 1, 1, 0, 1, 1, 0],
[1, 1, 0, 1, 0, 0, 1, 0, 1, 0],
[0, 0, 0, 0, 0, 1, 0, 1, 1, 0],
[0, 1, 0, 1, 0, 0, 0, 0, 1, 1],
[0, 0, 0, 1, 1, 0, 0, 1, 1, 1],
[1, 1, 0, 0, 0, 1, 1, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
[0, 0, 0, 1, 1, 0, 1, 1, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
[1, 1, 1, 1, 0, 1, 0, 1, 0, 0]]

"""

prompt4 = """Summate the values within the column labeled 'Doctor' in the provided HTML table. Present only the resulting numerical value without displaying the calculation process:
      Athlete	     Teacher	   Doctor	Artist	     Engineer
         24            19            82           47            37
         35            41            53           29            68
         52            44            74           38            28
         67            23            31           59            49
         73            36            17           41            59
         39            29            63           88            51
         82            52            27           71            38
         65            17            78           33            42
         49            62            19           54            57
         33            47            81           36            29
         27            88            35           21            63
         72            56            44           78            31
         38            45            67           29            57
         44            66            51           92            29
         86            22            39           75            53
         31            83            29           68            47
         55            48            67           33            72
         63            57            21           78            39
         49            34            69           58            25
         52            43            54           31            66       """
prompt5 = "Please supply the numerical value indicating the count of unique female pet names within the provided list: Benjamin,Michelle,Darlene,Andrew,Jonathan."

Prompt_list = [prompt1,prompt2,prompt3,prompt4,prompt5]
print("..................................................")
for i in range(len(Prompt_list)):

    print(f"Question {i+1} : {Prompt_list[i]}")
    # answer =
    print(" ")
    print("Generated Answer:", generate_answer(Prompt_list[i]))
    print("  .........>")
