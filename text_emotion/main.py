from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline

tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
model = TFRobertaForSequenceClassification.from_pretrained("arpanghoshal/EmoRoBERTa")

emotion = pipeline('sentiment-analysis', 
                    model='arpanghoshal/EmoRoBERTa')

emotion_labels = emotion("Thanks for using it.")

emotion = emotion_labels[0]['label']

emotion_score = emotion_labels[0]['score']



happy = ['admiration', 'optimism', 'pride', 'realization', 
         'relief', 'amusement', 'approval', 'caring', 
         'desire', 'gratitude', 'joy', 'desire', 'excitement', 
         'curiosity', 'love', 'surprise']

angry = ['anger', 'annoyance']
sad = ['confusion', 'remorse', 'sadness', 'disappointment', 'nervousness', 'disapproval', 'disgust', 'embarrassment', 'grief', 'fear',]

emo = 'neutral'

if emotion in happy:
    emo = 'happy'

elif emotion in sad:
    emo = 'sad'
    
elif emotion in angry:
    emo = 'angry'

    
print(emo)
print(emotion_score)

