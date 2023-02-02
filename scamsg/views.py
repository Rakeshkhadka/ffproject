from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.
import pickle
# from .utils import LogisticRegression


def load_models():
    # model_file = 'models/model.pkl'
    # vectorizer_file = 'models/vectorizer.pkl'
    # model_file = 'models/spam-sms-mnb-model.pkl'
    # vectorizer_file = 'models/cv-transform.pkl'
    model_file = 'models/gnb/gnb_scammodel.pkl'
    vectorizer_file = 'models/gnb/gnbscamvectorizer.pkl'
    model = pickle.load(open(model_file, 'rb'))
    vectorizer = pickle.load(open(vectorizer_file, 'rb'))
    return model, vectorizer

model, vectorizer = load_models()




def detect_scam_using_gnb(request):
    if request.method == 'POST':
        message = request.POST['message']
        message = vectorizer.transform([message])
        prediction = model.predict(message.toarray())
        print(prediction)
        return HttpResponse(f'The message is {"not a scam." if prediction[0] == 0 else "a scam."}')

    return render(request, 'detect_scam.html')


from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

# Load the tokenizer
with open('models/lstm/tokenizer_l.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the model
model = load_model('models/lstm/lstmmodel_l.h5')

def detect_scam_lstm(request):
    if request.method == 'POST':
        # Get the new text data from the request
        message = request.POST['message']

        # Preprocess the new text data
        message_sequence = tokenizer.texts_to_sequences([message])
        message_padded = pad_sequences(message_sequence, maxlen=189)

        # Make predictions
        predictions = model.predict(message_padded)
        print(predictions)
        # Convert predictions to binary labels
        predictions = [0 if prediction[0] < 0.5 else 1 for prediction in predictions]

        print(predictions)
        # Render the predictions in a template
        return render(request, 'detect_scam1.html', {'predictions': predictions})

    return render(request, 'detect_scam1.html')