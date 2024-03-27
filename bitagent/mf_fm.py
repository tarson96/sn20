import spacy

def read_word_list(file_path):
    with open(file_path, "r") as file:
        return [line.strip().lower() for line in file]

def count_unique_names(input_str):
    nlp = spacy.load("en_core_web_sm")

    # Load male and female words
    male_words = read_word_list("/content/male.txt")
    female_words = read_word_list("/content/female.txt")

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

# # Example input strings
# input_str_male = "Given the following list of pet names, determine the number of unique male pet names:Charlie ,Max ,Buddy Bella ,Charlie,Cooper ,Whiskers."
# input_str_female = "Simply provide the numerical value representing the count of unique female pet names within the given list: Lorraine,Howard,Caroline,Austin,Roxy."

# # Count unique pet names
# male_pet_name_count = count_unique_names(input_str_male)
# print(f"The count of unique male pet names within the given list is: {male_pet_name_count}")

# female_pet_name_count = count_unique_names(input_str_female)
print(f"The count of unique female pet names within the given list is: {female_pet_name_count}")
