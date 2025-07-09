

import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from .models import Stopword, Slangword, PreprocessingModel

import re
from nltk.corpus import stopwords as nltk_stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from textblob import TextBlob
from .models import Stopword, Slangword
from deep_translator import GoogleTranslator
import nltk
class Preprocessing:
    def __init__(self):
       
        db_stopwords = set(s.stopwords.strip().lower() for s in Stopword.objects.all())
        
        try:
            nltk.data.find('corpora/stopwords')
        except nltk.downloader.DownloadError:
            nltk.download('stopwords')
        try:
            nltk.data.find('tokenizers/punkt')
        except nltk.downloader.DownloadError:
            nltk.download('punkt')

        nltk_id_stopwords = set(nltk_stopwords.words('indonesian'))
        self.stopwords = db_stopwords.union(nltk_id_stopwords)
       

        self.slang_dict = {s.katatidakbaku.lower(): s.katabaku.lower() for s in Slangword.objects.all()}

        
        self.stemmer = StemmerFactory().create_stemmer()

    def _remove_links(self, text):
 
        return re.sub(r'https?://\S+|www\.\S+', '', text)

    def _remove_emoticons(self, text):
        """Removes common emoticons/emojis from the text."""
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # Emoticons
            "\U0001F300-\U0001F5FF"  # Symbols & pictographs
            "\U0001F680-\U0001F6FF"  # Transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # Flags
            "\U00002700-\U000027BF"  # Dingbats
            "\U0001F900-\U0001F9FF"  # Supplemental symbols and pictographs
            "\U0001FA70-\U0001FAFF"  # Extended symbols
            "\U00002600-\U000026FF"  # Misc symbols
            "\U000024C2-\U0001F251"  # Enclosed characters
            "]+", flags=re.UNICODE
        )
        return emoji_pattern.sub(r'', text)

    def _remove_numbers(self, text):
        return re.sub(r'\d+', '', text)

    def _case_folding(self, text):
        text = self._remove_links(text)
        text = self._remove_numbers(text)
        text = self._remove_emoticons(text)
        return text.lower().strip() 

    def _remove_symbols(self, text):
        return re.sub(r'[^a-zA-Z\s]', '', text)

    def _replace_slang(self, tokens):
        return [self.slang_dict.get(word, word) for word in tokens]

    def _remove_stopwords(self, tokens):
      
        return [word for word in tokens if word.strip().lower() not in self.stopwords]


    def _stemming(self, tokens):
        return [self.stemmer.stem(word) for word in tokens]

    def _tokenize(self, text):
        return nltk.word_tokenize(text)

    def clean_text(self, text):
        text = self._case_folding(text)
        text = self._remove_symbols(text) 
        tokens = self._tokenize(text)
        tokens = self._replace_slang(tokens)
        tokens = self._remove_stopwords(tokens)
        tokens = self._stemming(tokens)
        return ' '.join(tokens)

    def process_corpus(self, texts):
        if not isinstance(texts, list):
            raise TypeError("Input 'texts' harus berupa list.")
        return [self.clean_text(text) for text in texts]
class Klasifikasi:
    def __init__(self, model_path='logistic_model.pkl', vectorizer_path='tfidf_vectorizer.pkl'):
        self.model = None
        self.vectorizer = None
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self._load_or_initialize_models() 

    def _load_or_initialize_models(self):
        try:
            self.model = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)
            
        except FileNotFoundError:
           
            self.vectorizer = TfidfVectorizer()
            self.model = LogisticRegression(max_iter=500, solver='liblinear') 

    def train_model(self, texts, labels):

        X = self.vectorizer.fit_transform(texts)
        
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42, stratify=labels) 

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred, labels=self.model.classes_)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        acc = accuracy_score(y_test, y_pred)

        joblib.dump(self.model, self.model_path)
        joblib.dump(self.vectorizer, self.vectorizer_path)
       

        return cm, report, acc

    def predict(self, input_text):
       
        if self.model is None or self.vectorizer is None:
            raise RuntimeError("Model Belum terlatih atau belum dipanggil. silakan latih model atau panggil model terlebih dahulu.")

        pre = Preprocessing()
        cleaned = pre.clean_text(input_text)
        
        vec = self.vectorizer.transform([cleaned])
        return self.model.predict(vec)[0]

    def transform_tfidf(self, texts):
      
        if self.vectorizer is None:
            raise RuntimeError("Model Vectorizer Belum terlatih atau belum dipanggil. silakan latih model atau panggil model terlebih dahulu.")

        cleaned_texts = PreprocessingModel.objects.all().values_list('text_bersih', flat=True)
        X = self.vectorizer.transform(cleaned_texts) 
        feature_names = self.vectorizer.get_feature_names_out()
        return X, feature_names, cleaned_texts



class Labeling:
    def __init__(self):
     
        try:
            nltk.data.find('tokenizers/punkt')
        except nltk.downloader.DownloadError:
            nltk.download('punkt')

        self.translator = GoogleTranslator(source='auto', target='en')
        self.processed_count = 0 

    def _translate_text(self, text):
        try:
            translated_text = self.translator.translate(text)
            if translated_text is None or not translated_text.strip():
                return "" 
            return translated_text
        except Exception as e:
            return "" 

    def label_by_textblob_fast(self, text):
        if not text or not text.strip():
            return 
            
        en_text = self._translate_text(text)
        if not en_text: 
            return 
        
        polarity = TextBlob(en_text).sentiment.polarity
        self.processed_count += 1

        # if polarity > 0.1:
        #     return 'positif'
        # elif polarity < -0.1:
        #     return 'negatif'
        # elif polarity == 0:
        #     return 'netral'
        # else:
        #     return 'netral'
        
        if polarity > 0.1:
            return 'positif'
        elif polarity < -0.1:
            return 'negatif'
        else:
            return 'netral'
        
        
