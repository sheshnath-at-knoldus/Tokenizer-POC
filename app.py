import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')


def read_file(file_path):
    file = open(file_path, 'r')
    read_content = file.read()
    file.close()
    return read_content


def write_into_file(file_path, content_to_write):
    file = open(file_path, 'w')
    file.write(content_to_write)
    file.close()


def tokenization_by_word(sentence):
    tokens = nltk.word_tokenize(sentence)
    return tokens


def tokenization_by_sentence(sentence):
    tokens = nltk.sent_tokenize(sentence)
    return tokens


def freq_dist_by_tokenized_word(words):
    freq = nltk.FreqDist(words)
    return freq


def tokenize_word_without_stop_words(tokenize_words):
    stop_words = set(stopwords.words("english"))
    words_without_stop_word = []
    for word in tokenize_words:
        if word not in stop_words:
            words_without_stop_word.append(word)
    return words_without_stop_word


def sentiment_analyzer_from_sentence(sentence):
    sent_analyzer = SentimentIntensityAnalyzer()
    return sent_analyzer.polarity_scores(sentence)


result_tokenize_by_word = tokenization_by_word(read_file("file.txt"))
result_tokenize_by_sentence = tokenization_by_sentence(read_file("file.txt"))
result_freq_dis = freq_dist_by_tokenized_word(result_tokenize_by_word)
result_of_sentiment_analyzer = sentiment_analyzer_from_sentence(read_file("file.txt"))
result_of_words_without_stopword = tokenize_word_without_stop_words(result_tokenize_by_word)

print("Sentiment analyzer", result_of_sentiment_analyzer)
print("\nTokenized words without stop word", result_of_words_without_stopword)
print("\nfreq of tokenize word which is most common ", result_freq_dis.most_common(40))
print("\ntokenize by word ", result_tokenize_by_word)
print("\n tokenize by sentence  ", result_tokenize_by_sentence)

write_into_file("output_file.txt", str(result_tokenize_by_word))
