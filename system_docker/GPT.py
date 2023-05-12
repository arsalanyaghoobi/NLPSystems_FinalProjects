import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
from transformers import GPT2Tokenizer, GPT2LMHeadModel
path = r"GPT"
tokenizer = GPT2Tokenizer.from_pretrained(path, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(path)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.eos_token_id
max_len = 50

def conversation_generation(input_text, label):
    # label = ""
    new_user_input_ids = tokenizer.encode_plus('[BOS]' + label+' ' + '[EOS]' +'[BOS]' + '' + input_text + '' +'[EOS]'+ '[BOS]', return_tensors='pt', truncation=True, max_length=max_len, padding="max_length")
    bot_input_ids = new_user_input_ids['input_ids']
    attention_mask = new_user_input_ids['attention_mask']
    output = model.generate(bot_input_ids, attention_mask=attention_mask, max_length= 100, do_sample=True, temperature=1.0, top_k=2)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_text = generated_text.lstrip(label)
    generated_text = generated_text.lstrip(input_text)
    return generated_text

# if __name__ == '__main__':
#     label_map = {0: 'Joyful', 1: 'Scared', 2: 'Sad', 3: 'Neutral', 4: 'Excited', 5: 'Mad'}
#     emotion = input('emotion: ')
#     text = input('text: ')
#     conversation_generation(text, emotion)
