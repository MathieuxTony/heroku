import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

stop_words = set(stopwords.words('english'))

dico = {'c#': 'csharp',
        'c++': 'cplusplus',
        '.net': 'dotnet',
        'n-d': 'nd'}

def pretraitement_titres(text, dico):
    text = text.lower()
    for key in dico.keys():
        text = text.replace(key, dico[key])
    text = re.sub(r'[^\w\s]', ' ', text).replace(' js ', ' javascript ')
    return text

def tokenisation(text):
    text_token = word_tokenize(text)
    return text_token

def get_nouns(tokens):
    tag_tokens = nltk.pos_tag(tokens)
    return  [w[0] for w in tag_tokens if w[1] == 'NN']

def remove_stop_words(tokens, stop_words):
    filtered_tokens = [w for w in tokens if not w in stop_words]
    return filtered_tokens

def lemmatisation(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in tokens]

def preparation_data(data):
    y = pretraitement_titres(data, dico)
    y = tokenisation(y)
    y = get_nouns(y)
    y = remove_stop_words(y, stop_words)
    y = lemmatisation(y)
    return y