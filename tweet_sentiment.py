import pandas as pd
import numpy as np
import re
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold
from fasttext import train_supervised, load_model




columns = ['id', 'tdate', 'tmane', 'ttext', 'ttype', 'trep', 'trtw', 'tfav', 'tstcount', 'tfol', 'tfrien', 'listcount']
pos = pd.read_csv('positive.csv', sep=';', names=columns)
neg = pd.read_csv('negative.csv', sep=';', names=columns)


def cleaner(documents):
    """
    Replaces urls, hashtags and retweets with spaces.
    Replaces smiles with special tag.
    :param documents: list: of str
    :returns docs: list: of str
    """
    docs = list()

    for doc in documents:
        text = re.sub("(@\w+)|(#\w+)", " ", doc.lower())

        text = re.sub("\n", " ", text)
        text = re.sub("(\w+:\/\/\S+)", " ", text)
        text = re.sub("rt ", " ", text)
        text = re.sub(" rt ", " ", text)
        text = re.sub(":\(", " bad_flag ", text)
        text = re.sub("\(\(+", " bad_flag ", text)
        text = re.sub("99+", " bad_flag ", text)
        text = re.sub("0_0", " bad_flag ", text)
        text = re.sub("o_o", " bad_flag ", text)
        text = re.sub("о_о", " bad_flag ", text)
        text = re.sub(":-\(", " bad_flag ", text)
        text = re.sub("=\(", " bad_flag ", text)
        text = re.sub(" \(", " bad_flag ", text)
        text = re.sub(";\)", " good_flag ", text)
        text = re.sub(":d+", " good_flag ", text)
        text = re.sub("\=\)+", " good_flag ", text)
        text = re.sub("\)+", " good_flag ", text)
        text = re.sub(":\)", " good_flag ", text)

        text = re.sub(r'[^\w\s]', '', text)

        text = text.strip()
        docs.append(text)

    return docs


data_train = pd.concat([pos[['ttext', 'ttype']], neg[['ttext', 'ttype']]])
data_train.rename(columns={"ttext": "body"}, inplace=True)
data_train.rename(columns={"ttype": "label"}, inplace=True)

print("Classic models training:")

pipe = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', SGDClassifier()),
])

param_grid = [
    {
        'vect': [TfidfVectorizer()],
        'vect__max_df': (0.25, 0.5, 0.75),
        'vect__ngram_range': ((1, 1), (1, 2)),
    },
    {
        'vect': [CountVectorizer()],
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'vect__ngram_range': ((1, 1), (1, 2))
        #         'regr__alpha': np.logspace(-4, 1, 6),
    },

    {
        'clf': [SGDClassifier()],
        'clf__max_iter': (20, 25),
        'clf__alpha': (0.00001, 0.000001),
        'clf__penalty': ('l2', 'elasticnet'),
    },
    {
        'clf': [LogisticRegression()],
        'clf__C': np.logspace(-3, 3, 4),
        'clf__penalty': ('l2', 'l2'),
    },
    {'clf': [XGBClassifier(objective='binary:logistic', colsample_bytree=0.3, learning_rate=0.1,
                           max_depth=5, alpha=10, n_estimators=10)],
     'clf__min_child_weight': [0.5, 1],
     'clf__gamma': [5, 10],
     'clf__subsample': [0.7, 1.0],
     'clf__colsample_bytree': [0.5, 1.0],
     'clf__max_depth': [8, 12]
     }
]

f1 = make_scorer(f1_score, average='macro')

grid_search = GridSearchCV(pipe, param_grid, n_jobs=-1, verbose=1, scoring=f1)

grid_search.fit(data_train['body'], data_train['label'])

print("Best score: %0.6f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
print('Best classical model:', best_parameters)

print("FastText training:")

models = {}
params = dict()
params['epoch'] = [30, 40, 50]
params['lr'] = [1, 0.1, 0.01]
params['min_count'] = [5, 10, 15]
params['word_ngrams'] = [1, 2]
params['dim'] = [100, 200]
params['loss'] = ['softmax', 'hs']

model_results = pd.DataFrame(columns=['model', 'type', 'precision', 'recall', 'f1-score'])
types = ['pos', 'neg']


def fasttext_classification_metrics(target_class, predicted_class):
    result = metrics.classification_report(target_class, predicted_class, target_names=['pos', 'neg'], output_dict=True)
    return result


df = data_train

new_folds = []

skf = StratifiedKFold(n_splits=3, shuffle=True)
folds = skf.get_n_splits(df['body'], df['label'])
for train_index, test_index in skf.split(data_train['body'], data_train['label']):
    new_folds.append([df.iloc[train_index], df.iloc[test_index]])

# print(new_folds)

folds = new_folds

kfold_metrics = []

# paths to write files in fasttext format
train_path = 'data_train'
test_path = 'data_test'


def to_ft_format(documents, labels, file_path):
    """
    Converts and save a dataset to fasttext compliant training format.

    :param documents: list: of str
    :param labels: list: of str/int
    :file_path: str
    """

    # Add mandatory "__label__" prefix to the labels as required by fasttext
    labels = ["__label__" + str(label) for label in labels]

    # clean up documents
    documents = cleaner(documents)

    with open(file_path, 'w', encoding="utf-8") as f:
        for doc, label in zip(documents, labels):
            f.write(label + " " + doc + "\n")

    print("Output file with %d samples saved at location: %s" % (len(labels), file_path))
    
to_ft_format([x for x in data_train['body']], [x for x in data_train['label']], 'data_train.txt')

model_results = pd.DataFrame()
modelnum = 1

for dim in params['dim']:
    for param_loss in params['loss']:
        for epoch in params['epoch']:
            for lr in params['lr']:
                for min_count in params['min_count']:
                    for word_ngrams in params['word_ngrams']:
                        kfold_metrics = []
                        for fold in folds:
                            train_df = fold[0]
                            test_df = fold[1]
                            to_ft_format(train_df['body'], train_df['label'], train_path)
                            to_ft_format(test_df['body'], test_df['label'], test_path)
                            print(epoch, lr, min_count, word_ngrams)
                            model = train_supervised(input=train_path, loss=param_loss, minCount=min_count, dim=dim,
                                                     lr=lr, ws=5, epoch=epoch, wordNgrams=word_ngrams, thread=8)
                            modelnum += 1

                            test_X = cleaner(test_df['body'])
                            predictions = [model.predict(doc)[0][0].replace("__label__", "") for doc in test_X]

                            result = metrics.classification_report(test_df['label'].astype(str), predictions,
                                                                   target_names=('pos', 'neg'), output_dict=True)
                            kfold_metrics.append(result)
                        print(kfold_metrics)
                        cross_f1_score = np.mean([elem['macro avg']['f1-score'] for elem in kfold_metrics])
                        current_model = pd.DataFrame([["model_num:" + str(model) + 'epoch:' + str(epoch) + 'lr:' + str(
                            lr) + ' ,min_count:' + str(min_count) + ' ,word_ngrams:' + str(
                            word_ngrams) + ' ,loss:' + str(param_loss) + ' ,dim:' + str(dim), cross_f1_score]],
                                                     columns=['model', 'f1-score'])
                        model_results = pd.concat([model_results, current_model])

print('Best FastText model:', model_results.sort_values(by=['f1-score'], ascending=False)[:1])
