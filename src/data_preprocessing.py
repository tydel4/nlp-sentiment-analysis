def clean_text(text):
    import re
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer

    # Initialize the stemmer
    stemmer = PorterStemmer()
    # Load English stop words
    stop_words = set(stopwords.words('english'))

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters and numbers
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    words = text.split()
    # Remove stop words and stem the words
    cleaned_text = ' '.join(stemmer.stem(word) for word in words if word not in stop_words)

    return cleaned_text

def remove_stopwords(text):
    from nltk.corpus import stopwords

    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_text = ' '.join(word for word in words if word not in stop_words)

    return filtered_text