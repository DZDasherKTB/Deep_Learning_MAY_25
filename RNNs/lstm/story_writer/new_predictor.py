from predictor import generate_story


print("📝 Random Sentence Generator\n")
start = input("Enter a starting prompt: ")

print("\n🧠 Generated Sentences:\n")
for i in range(1):
    result = generate_story(start, max_words=60, temperature=0.8, top_k=10, device='cuda')
    print(f"{i+1}. {result}\n")