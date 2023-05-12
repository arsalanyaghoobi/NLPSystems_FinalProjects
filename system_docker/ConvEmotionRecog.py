import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

dir = 'ConvEmotionRecog'
tokenizer = RobertaTokenizer.from_pretrained(dir)
model = RobertaForSequenceClassification.from_pretrained(dir, num_labels=6)
label_map = {0:'Joyful', 1:'Scared', 2:'Sad', 3:'Neutral', 4:'Excited', 5:'Mad'}

def EmotionRecog(data1, data2):
    data = data1+" "+data2
    encodes = tokenizer.encode(data, return_tensors='pt')
    out = model(encodes)
    preds = torch.argmax(out.logits, dim=1)
    label = label_map[preds.item()]
    # print(f">> label: {label}")
    return label

# if __name__ == '__main__':
#     text_1 = input("1: ")
#     text_2 = input("2. ")
#     EmotionRecog(text_1, text_2)
