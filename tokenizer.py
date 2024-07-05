from transformers import AutoTokenizer


checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


raw_input = [
    "I've been waiting for hugging face my whole life.",
    "I hate this so much!"
]
inputs = tokenizer(raw_input, padding=True, truncation=True, return_tensors="pt")

print(inputs)