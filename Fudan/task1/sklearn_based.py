import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import os

data_path = "data"
train = pd.read_csv(os.path.join(data_path, 'train.tsv'), sep='\t')
test = pd.read_csv(os.path.join(data_path, 'test.tsv'), sep='\t')

ngram_vectorizer = CountVectorizer(ngram_range=(1, 1))
model = LogisticRegression(max_iter=1000)
params = {
    'C': [0.5, 0.8, 1.0],
    'penalty': ['none', 'l2'],
}
skf = StratifiedKFold(n_splits=3)
gsv = GridSearchCV(model, params, cv=skf)
pipeline = Pipeline([
    ("ngram", ngram_vectorizer),
    ("model", gsv),
])
X = train["Phrase"]
y = train["Sentiment"]

# gsv.fit(X, y)
pipeline.fit(X, y)
print(gsv.best_score_)
print(gsv.best_params_)
# test['Sentiment'] = pipeline.predict(test['Phrase'])
# test[['Sentiment', 'PhraseId']].set_index("PhraseId").to_csv('sklearn_based_lr.csv')
