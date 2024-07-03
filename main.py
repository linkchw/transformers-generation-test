from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Model
# from ChatData import ChatData


tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
# tokenizer.add_special_tokens({"pad_token": "<pad>", 
#                               "bos_token": "<startofstring>",
#                               "eos_token": "<endofstring>"
#                               })
# tokenizer.add_tokens(["<bot>:"])

model = GPT2LMHeadModel.from_pretrained("gpt2-xl")

generate_text = model.generate(**tokenizer("i was good at backetball but ",\
    return_tensors = "pt"))[0]

decode = tokenizer.decode(generate_text)

print(decode)
model.save_pretrained("gpt-xl")


# chatData = ChatData("~/personal/data_manufacture/csv files/final_clean_names.csv",\
#                     tokenizer)


