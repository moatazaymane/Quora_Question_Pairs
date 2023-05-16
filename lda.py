import pandas as pd
import gensim
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if token not in stop_words]
    return tokens


def lda_topic_modeling(dataframe, text_column, num_topics=10):
    # Preprocess the text
    dataframe['preprocessed_text'] = dataframe[text_column].apply(preprocess_text)

    # Create dictionary and corpus from preprocessed text
    dictionary = Dictionary(dataframe['preprocessed_text'])
    corpus = [dictionary.doc2bow(doc) for doc in dataframe['preprocessed_text']]

    # Build LDA model
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=10,
        alpha='auto',
        eta='auto'
    )

    # Compute coherence score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=dataframe['preprocessed_text'], dictionary=dictionary,
                                         coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()

    # Print topics and coherence score
    print(f"Coherence Score: {coherence_lda:.3f}")
    for idx, topic in lda_model.print_topics(-1):
        print(f"Topic: {idx} \nWords: {topic}\n")

    # Create word cloud for each topic
    for idx in range(num_topics):
        plt.figure()
        plt.imshow(WordCloud().fit_words(dict(lda_model.show_topic(idx, topn=20))))
        plt.axis("off")
        plt.title(f"Topic #{idx}")
        plt.show()