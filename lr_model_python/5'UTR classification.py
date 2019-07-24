import pandas as pd
import numpy as np
import copy
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import mglearn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
import pickle


np.random.seed(30)
random.seed(30)


def get_mer(seq, size):
    """ Generates k-mers for a given sequence and create a sentence.
    # Args:
    #  seq: Sequence to be converted
    #    s: Length of the k-mers
    # Returns
    #    Sentence of k-mers."""
    return ' '.join([seq[x:x+size].lower() for x in range(len(seq) - size + 1)])


def comp(seq):
    """Generates the complement of a given sequence.
    # Args:
    #  seq: Sequence to be converted
    # Returns
    #    Sentence of k-mers."""
    for base in seq:
        if base not in 'ATCGN':
            print("Njehie: check your sequence")
            return None
    # Comprehension
    cmp = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    return ''.join([cmp[base] for base in seq])


#
def dat(file, t_split=0.1, kmer=10):
    """Process data into train and test set using get_mer().
    # Args:
    #  file: CSV file with sequences first column and labels in second column to converted
    #    s: train and test splitting percentage.
    # Returns
    #    Returns processed train and test data."""
    print('Preparing..')
    # Load .csv having "Seq" and "Lab"
    foo = pd.read_csv(file)
    # deep copy
    foo2 = copy.deepcopy(foo)
    # Complement
    foo2.Seq = foo2.Seq.apply(lambda x: comp(x))
    # Bind Final
    foo = foo.append(foo2, ignore_index=True)
    # Encode
    foo.Seq = foo.Seq.apply(lambda x: get_mer(x, kmer))
    # Randomise for splitting
    foo = foo.reindex(np.random.permutation(foo.index))
    # Create data split
    train_size = int(len(foo) * (1 - t_split))
    xx_train = foo.Seq.values[:train_size]
    yy_train = np.array(foo.Lab.values[:train_size])
    xx_test = np.array(foo.Seq.values[train_size:])
    yy_test = np.array(foo.Lab.values[train_size:])
    print('Mean Seq length: train {}'.format(np.mean(list(map(len, xx_train)), dtype=int)))
    print('Mean Seq length: test  {}'.format(np.mean(list(map(len, xx_test)),  dtype=int)))
    print("Number of documents in train data: {}".format(len(xx_train)))
    print("Samples per class (train): {}".format(np.bincount(yy_train)))
    data = (xx_train, yy_train, xx_test, yy_test)
    return data


def rl(x, y, nf=5):
    """Logistic regression model with grid search
    # Args:
    #  X: training data
    #  y: label for the training data
    #  nf: cross-validation fold
    # Returns
    #    Best model using optimised grid search, switch kernel to match needs"""
    cs = [0.001, 0.01, 0.1, 1, 10]
    rs = [None, 7, 14]
    param_grid = {'C': cs, 'random_state': rs}
    grid = GridSearchCV(LogisticRegression(solver='lbfgs', max_iter=5000), param_grid, cv=nf)
    grid.fit(x, y)
    grid.best_score_
    grid.best_params_
    grid.best_estimator_.coef_
    grid.best_estimator_
    resd = (grid.best_score_, grid.best_params_, grid.best_estimator_.coef_, grid.best_estimator_)
    return resd


# Usage
input_file = 'train_ex.csv'
x_train, y_train, x_test, y_test = dat(input_file, 0.15, 10)

# Set word vector
vect = CountVectorizer(min_df=5, ngram_range=(2, 2))
X_train = vect.fit(x_train).transform(x_train)
X_test = vect.transform(x_test)
# Test the dimensions
print("Vocabulary size: {}".format(len(vect.vocabulary_)))
print("X_train:\n{}".format(repr(X_train)))
print("X_test: \n{}".format(repr(X_test)))
#
feature_names = vect.get_feature_names()
print("Number of features: {}".format(len(feature_names)))


# Build model and extract
best_score, best_params, best_estimator_coef, best_estimator = rl(X_train, y_train)
#
print("Best cross-validation score: {:.2f}".format(best_score))
print("Best parameters: ", best_params)
#
mglearn.tools.visualize_coefficients(best_estimator_coef, feature_names, n_top_features=15)
plt.show()
#
print("Best estimator: ", best_estimator)
#

Res = best_estimator.predict(X_test)
print("Score: {:.2f}".format(best_estimator.score(X_test, y_test)))
#
confusion = confusion_matrix(y_test, Res)
print("confusion matrix \n", confusion)

# ROC-AUC Image not used
probs = best_estimator.predict_proba(X_test)
preds = probs[:, 1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label='GridSearchCV (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

# Classification report
print(classification_report(y_test, Res))

# Save model
filename = 'modelF.sav'
pickle.dump(best_estimator, open(filename, 'wb'))

# Predicts
# Single Sequence prediction usage
s_predict = "ATGCGCCCTGCAGCCCTGCGCGGGGCCCTGCTGGGCTGCCTCTGCCTGGCGTTGCTTTGCCTGGGCGGTGCGGACAAGCGAGCCCGATAA"
s_pp = [get_mer(s_predict, 1)]
print("Prediction: {}". format(best_estimator.predict(vect.transform(s_pp))))

# Predict on .csv many also called transcriptome-wide predict
foo3 = pd.read_csv("test_exTest.csv")
# Encode
foo3.Seq = foo3.Seq.apply(lambda x: get_mer(x, 10))
#
ubi_test = vect.transform(np.array(foo3.Seq.values))
#
ubires = best_estimator.predict(ubi_test)
np.savetxt('ubire.csv', ubires, delimiter=',')
