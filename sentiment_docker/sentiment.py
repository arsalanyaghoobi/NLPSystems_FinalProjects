import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

dir = 'Sentiment'
tokenizer = RobertaTokenizer.from_pretrained(dir)
model = RobertaForSequenceClassification.from_pretrained(dir, num_labels=6)
label_map = {0:'Joyful', 1:'Scared', 2:'Sad', 3:'Neutral', 4:'Excited', 5:'Mad'}

def sentiment(data):
    encodes = tokenizer.encode(data, return_tensors='pt')
    out = model(encodes)
    preds = torch.argmax(out.logits, dim=1)
    label = label_map[preds.item()]
    print(f">> label: {label}")
    return label

if __name__ == '__main__':
    continu = True
    while continu:
        text = input(">> Type: ")
        sentiment(text)
        result = input(">> For Quit Press q ")
        if result=='q':
            continu=False


# happy birthday
# are you serious? I can not believe it
# accept my condolence on your father death
# what the hell are you talking about
# it is frightenning; I can not handle it anymore