# from transformers import pipeline, AutoTokenizer, GPT2LMHeadModel
# tokenizer = AutoTokenizer.from_pretrained('bolbolzaban/gpt2-persian')
# model = GPT2LMHeadModel.from_pretrained('bolbolzaban/gpt2-persian')
# generator = pipeline('text-generation', model, tokenizer=tokenizer, config={'max_length':256})
# sample = generator('در یک اتفاق شگفت انگیز، پژوهشگران')



from transformers import GPT2LMHeadModel, AutoTokenizer, GPT2Model
# from ChatData import ChatData


tokenizer = AutoTokenizer.from_pretrained("bolbolzaban/gpt2-persian")
# tokenizer.add_special_tokens({"pad_token": "<pad>", 
#                               "bos_token": "<startofstring>",
#                               "eos_token": "<endofstring>"
#                               })
# tokenizer.add_tokens(["<bot>:"])

model = GPT2LMHeadModel.from_pretrained("bolbolzaban/gpt2-persian")

# generate_text = model.generate(**tokenizer("داودی",\
#     return_tensors = "pt"))[0]

encode = tokenizer.encode("داودی")
# decode = tokenizer.decode(generate_text)
second_encode = tokenizer.decode(encode)

print(encode)
print(second_encode)
# with open("output.txt", 'wt')as f:
    
#     print(generate_text)
#     f.write(str(generate_text) + "\n")
#     print(decode)
#     f.write(str(decode) + "\n")
#     # print(encode)
#     # f.write( str(encode)+ "\n")
#     print(second_encode)
#     f.write( str(second_encode)+ "\n")
