from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

import pickle
# from .utils import LogisticRegression


# def load_model():
#     # model_file = 'models/model.pkl'
#     # vectorizer_file = 'models/vectorizer.pkl'
#     # model_file = 'models/spam-sms-mnb-model.pkl'
#     # vectorizer_file = 'models/cv-transform.pkl'
#     model_file = 'models/rfc_model.pkl'
#     vectorizer_file = 'models/cv1_vectorizer.pkl'
#     model = pickle.load(open(model_file, 'rb'))
#     vectorizer = pickle.load(open(vectorizer_file, 'rb'))
#     return model, vectorizer

# model, vectorizer = load_model()

with open('models/sms_spam_tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the model
model = load_model('models/lstmmodel.h5')



def detect_scam(request):
    if request.method == 'POST':
        message = request.POST['message']
        # message = vectorizer.transform([message])
        # prediction = model.predict(message)
         # Preprocess the new text data
        new_text_sequence = tokenizer.texts_to_sequences([message])
        new_text_padded = pad_sequences(new_text_sequence, maxlen=189)

        # Make predictions
        predictions = model.predict(new_text_padded)
        predictions = [0 if prediction[0] < 0.5 else 1 for prediction in predictions]
        print(predictions)
        return HttpResponse(f'The message is {"not a scam." if predictions == 0 else "a scam."}')

    return render(request, 'detect_scam.html')



    def load_model():
    # model_file = 'models/model.pkl'
    # vectorizer_file = 'models/vectorizer.pkl'
    # model_file = 'models/spam-sms-mnb-model.pkl'
    # vectorizer_file = 'models/cv-transform.pkl'
    model_file = 'models/rfc_model.pkl'
    vectorizer_file = 'models/cv1_vectorizer.pkl'
    model = pickle.load(open(model_file, 'rb'))
    vectorizer = pickle.load(open(vectorizer_file, 'rb'))
    return model, vectorizer

model, vectorizer = load_model()