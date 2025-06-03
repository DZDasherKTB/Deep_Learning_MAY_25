from predictor import predict_next_word
import random

def generate_sentence(prompt, max_words=20):
    sentence = prompt
    for _ in range(max_words):
        predictions = predict_next_word(sentence, k=1)
        next_word = random.choice(predictions)
        sentence += " " + next_word
    return sentence

# --- Start ---
print("📝 Random Sentence Generator\n")
start = input("Enter a starting prompt: ")

print("\n🧠 Generated Sentences:\n")
for i in range(1):
    result = generate_sentence(start, max_words=50)
    print(f"{i+1}. {result}\n")