import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import pymorphy2

nltk.download('stopwords')

# Открытие и прочтение файла
with open('corpus_minaev.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Токенизация текста
words = word_tokenize(text)

# Лемматизация слов
morph = pymorphy2.MorphAnalyzer()
lemmatized_words = [morph.parse(word)[0].normal_form for word in words]

# Избавление от стоп-слов
stop_words = stopwords.words('russian')
filtered_words = [word for word in lemmatized_words if word.lower() not in stop_words and word.isalpha()]

# Определение части речи и выбор определенных частей речи
words_pos = [(morph.parse(word)[0].normal_form, morph.parse(word)[0].tag.POS) for word in filtered_words]
nouns = [word[0] for word in words_pos if word[1] == 'NOUN' or 'VERB' or 'ADJF' or 'ADJS' or 'COMP' or 'INFN']

# Частотность слов
freq_dist = FreqDist(nouns)

# Список 100 частотных слов
top_100_words = freq_dist.most_common(100)

print(top_100_words)
