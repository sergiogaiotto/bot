from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm

#dados para treino
with open("baseTreino.csv", encoding='utf-8') as f:
    sentences = f.read().split("\n")

label = []
interaction = []

for sentence in sentences[:-1]:
    sentence = sentence.split(";")
    label.append(sentence[0])
    interaction.append(sentence[1])

#stop words
with open("stopWords.csv", encoding="utf-8") as f:
    pt_stop_words = f.read().split("\n")

#Features
tfidf = TfidfVectorizer(sublinear_tf=True,
                        min_df=5, norm='l2',
                        encoding='latin-1',
                        ngram_range=(1, 2),
                        stop_words=pt_stop_words)
features = tfidf.fit_transform(interaction).toarray()
labels = label

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(interaction)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#SVM Model
svm_model = svm.SVC(kernel='linear', C=0.1).fit(X_train_tfidf, labels)

def classification(interaction):
    return svm_model.predict(count_vect.transform([interaction]))[0]
