from Sentiment import sentiment
from ConvEmotionRecog import EmotionRecog
from GPT import conversation_generation
label_map = {0:'Joyful', 1:'Scared', 2:'Sad', 3:'Neutral', 4:'Excited', 5:'Mad'}
def conv_sys():
    choice = True
    conversation_indx = 0
    bot_utterance = ""
    while choice:
        if conversation_indx == 0:
            utterance_1 = input(">>User: ")
            utterance_1_emotion = sentiment(utterance_1)
            print("User Emotion: ", utterance_1_emotion)
            bot_utterance = conversation_generation(utterance_1, utterance_1_emotion)
            print(f">>Bot: ", bot_utterance)
            conversation_indx += 1
        else:
            utterance_1 = input(f">>User: ")
            utterance_1_emotion = EmotionRecog(bot_utterance, utterance_1)
            print("user emotion: ", utterance_1_emotion)
            bot_utterance = conversation_generation(utterance_1, utterance_1_emotion)
            print(">>Bot: ", bot_utterance)
            result = input("For Quit Press q: ")
            if result == 'q':
                choice = False






if __name__ == '__main__':
    conv_sys()