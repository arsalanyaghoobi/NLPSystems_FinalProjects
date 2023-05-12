import transformers
transformers.logging.set_verbosity_error()
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

def conv_function():
    continu = True
    con_index = 0
    while continu:
        new_user_input_ids = tokenizer.encode(input(">> User:  ") + tokenizer.eos_token  , return_tensors='pt')
        if con_index == 0:
            bot_input_ids = new_user_input_ids
        else:
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        print(">> BOT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
        con_index += 1
        choice = input("for quit type q   ")
        if choice == 'q':
            continu =False



if __name__ == '__main__':
    conv_function()