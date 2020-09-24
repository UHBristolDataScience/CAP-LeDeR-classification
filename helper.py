from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, precision_score, recall_score, f1_score
from IPython.core.display import display, HTML

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re


def pd_print(df):

    display(HTML(df.to_html()))


def accuracy(clf_, x_, y_, name='Test', print_flag=True):
    y_score = clf_.predict(x_)

    n_right = 0
    for i in range(len(y_score)):
        if y_score[i] == list(y_)[i]:
            n_right += 1
    acc = (n_right / float(len(y_)) * 100)

    if print_flag:
        print("%s Accuracy: %.2f%%" % (name, acc))

    return acc


def lemmatize_text(text, stemmer, remove_nan=False):

    documents = []

    for sen in range(0, len(text)):

        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(text.iloc[sen]))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
        # document = re.sub(r'\s+[a-zA-Z]\s+', ' ', str(X[sen]))

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)

        # Converting to Lowercase
        document = document.lower()

        if remove_nan:
            document = re.sub('nan', '', document)

        # Lemmatization
        document = document.split()

        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)

        documents.append(document)

    return documents


def summarise_gridsearch_classifier(clf):

    print("Best score: %0.3f" % clf.best_score_)
    print("Best parameters set:")
    best_parameters = clf.best_estimator_.get_params()

    parameters = [p for p in best_parameters.keys()
                  if any(x in p for x in ['clf__',
                                          'vect__',
                                          'tfidf__'])]

    for param_name in sorted(parameters):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


def calibrate_random_forest(x_train, y_train):

    cal_parameters = {
        'clf__base_estimator__n_estimators': (100,),
        'clf__base_estimator__max_samples': (0.8,),
        'vect__ngram_range': ((1, 2),),
        'vect__max_df': (0.7,),
        'vect__min_df': (5,),
        'vect__max_features': (1500, 2000, 2500),
    }
    cal_pipeline = Pipeline([
        ('vect', CountVectorizer(stop_words=stopwords.words('english'))),
        ('tfidf', TfidfTransformer()),
        ('clf', CalibratedClassifierCV(
            base_estimator=RandomForestClassifier(random_state=0),
            cv=2, method='isotonic'))
    ])
    calibrated_random_forest = GridSearchCV(cal_pipeline, cal_parameters, n_jobs=-1, verbose=1, cv=10)
    calibrated_random_forest.fit(x_train, y_train)

    return calibrated_random_forest


def plot_roc_curve(clf, x_1, y_1, x_2, y_2, name_1, name_2, pos_label=2, ax=None):

    y_pred_1 = clf.predict_proba(x_1)[:, 1]
    fpr_1, tpr_1, _ = roc_curve(y_1, y_pred_1, pos_label=pos_label)

    y_pred_2 = clf.predict_proba(x_2)[:, 1]
    fpr_2, tpr_2, _ = roc_curve(y_2, y_pred_2, pos_label=pos_label)

    if ax is None:
        plt.figure(figsize=(7, 5))
        ax = plt.gca()

    ax.plot(fpr_1, tpr_1, label=name_1 + ', AUC=%.2f' % roc_auc_score(y_1, y_pred_1))
    ax.plot(fpr_2, tpr_2, label=name_2 + ', AUC=%.2f' % roc_auc_score(y_2, y_pred_2))

    ax.legend()
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.plot([0, 1], [0, 1], 'k--')

    if ax is None:
        plt.savefig('ROC_train_test.png')


def plot_calibration_curve(clf, calibrated_clf,
                           X_test, y_test,
                           bins=5, classifier_type='Random forest'):

    fig = plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    for clf, name in [(clf, classifier_type),
                      (calibrated_clf, classifier_type + ' + Sigmoid')]:

        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        clf_score = brier_score_loss(y_test, prob_pos, pos_label=np.max(y_test))
        print("%s:" % name)
        print("\tBrier: %1.3f" % clf_score)
        print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
        print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
        print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=bins)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s (%1.3f)" % (name, clf_score))

        ax2.hist(prob_pos, range=(0, 1), bins=bins, label=name,
                 histtype="step", lw=2, density=True)

        ax1.set_ylabel("Fraction of positives")
        ax1.set_ylim([-0.05, 1.05])
        ax1.legend(loc="lower right")
        ax1.set_title('Calibration plots  (reliability curve)')

        ax2.set_xlabel("Mean predicted value")
        ax2.set_ylabel("Normalised Count")
        ax2.legend(loc="upper center", ncol=2)

        plt.tight_layout()

    plt.savefig('calibration_rf_%d_bins.png' % bins)


def plot_calibration_curve_easy_hard(calibrated_clf,
                                     easy_x, hard_x,
                                     easy_y, hard_y,
                                     bins=5):

    fig = plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    clf = calibrated_clf

    X = [easy_x, hard_x]
    Y = [easy_y, hard_y]

    names = ['easy cases', 'hard cases']

    for (x, y), name in zip(zip(X, Y), names):

        y_pred = clf.predict(x)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(x)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(x)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        clf_score = brier_score_loss(y, prob_pos, pos_label=np.max(y))
        print("%s:" % name)
        print("\tBrier: %1.3f" % clf_score)
        print("\tPrecision: %1.3f" % precision_score(y, y_pred))
        print("\tRecall: %1.3f" % recall_score(y, y_pred))
        print("\tF1: %1.3f\n" % f1_score(y, y_pred))

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y, prob_pos, n_bins=bins)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s (%1.3f)" % (name, clf_score))

        ax2.hist(prob_pos, range=(0, 1), bins=bins, label=name,
                 histtype="step", lw=2, density=True)

        ax1.set_ylabel("Fraction of positives")
        ax1.set_ylim([-0.05, 1.05])
        ax1.legend(loc="lower right")
        ax1.set_title('Calibration plots  (reliability curve)')

        ax2.set_xlabel("Mean predicted value")
        ax2.set_ylabel("Normalised Count")
        ax2.legend(loc="upper center", ncol=2)

        plt.savefig('calibration_rf_easy_hard_%d_bins.png' % bins)


def compute_all_metrics(clf, x_1, y_1, x_2, y_2, name_1, name_2):

    results = pd.DataFrame()
    results['metric'] = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1']

    y_pred_1 = clf.predict_proba(x_1)[:, 1]
    y_pred_2 = clf.predict_proba(x_2)[:, 1]

    y_p_1 = clf.predict(x_1)
    y_p_2 = clf.predict(x_2)

    results[name_1] = [roc_auc_score(y_1, y_pred_1),
                       accuracy(clf, x_1, y_1, name=None, print_flag=False),
                       precision_score(y_1, y_p_1),
                       recall_score(y_1, y_p_1),
                       f1_score(y_1, y_p_1)]

    results[name_2] = [roc_auc_score(y_2, y_pred_2),
                       accuracy(clf, x_2, y_2, name=None, print_flag=False),
                       precision_score(y_2, y_p_2),
                       recall_score(y_2, y_p_2),
                       f1_score(y_2, y_p_2)]

    return results

