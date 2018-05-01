import pickle
import spacy
import re
from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load('en', disable=['parser'])


CLASSIFIER_ROOT = 'classifiers/'
TRANSFORMERS = ['transform_bag_of_words_0.sav',
                'transform_bag_of_words_1.sav']
MODELS = ['nb.sav']


STOP_WORDS.add("'s")
for word in STOP_WORDS:
    lexeme = nlp.vocab[word]
    lexeme.is_stop = True

def load_model(model_name):
    with open('{0}{1}'.format(CLASSIFIER_ROOT, model_name), 'rb') as f:
        model = pickle.load(f)
    return model

CLF_NB = load_model(MODELS[0])
TRANSFORMERS_MODELS = [load_model(TRANSFORMERS[0]), load_model(TRANSFORMERS[1])]

def clean_html(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext.lower()


def lemmatization(text):
    lemma = []
    tokenized_sent = nlp(clean_html(text), disable=['parser'])
    for token in tokenized_sent:
        if not token.is_stop and not token.is_punct:
            lemma.append(token.lemma_)
    return " ".join(lemma)


def predict_nb(text):
    modified_text = TRANSFORMERS_MODELS[1].transform(
        TRANSFORMERS_MODELS[0].transform([lemmatization(text)]))
    return CLF_NB.predict(modified_text)[0]
