from predictor import predict_next_word

print("welcome to the world of AI!")   
prompt = input("Enter a prompt: ")
predictions = predict_next_word(prompt)

print("Predicted next words:", predictions)

while True:
    prompt = prompt+" "+input("Enter a prompt: ")
    if prompt.lower() == "exit":
        break
    predictions = predict_next_word(prompt)
    print(f"Prompt: '{prompt}'")
    print("Predicted next words:", predictions)