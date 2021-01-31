import pandas as pd
import numpy as np
import shap
import fatf
import fatf.transparency.predictions.surrogate_explainers as fatf_surrogates
import fatf.vis.lime as fatf_vis_lime
from treeinterpreter import treeinterpreter as ti
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TEST_SIZE = 0.2
fatf.setup_random_seed(RANDOM_STATE)

def get_rf_feature_importances(grid_search_rf):
    fimps = pd.DataFrame()
    fimps['feature'] = grid_search_rf.best_estimator_['vect'].get_feature_names()
    fimps['contribution'] = grid_search_rf.best_estimator_['clf'].feature_importances_
    fimps['magnitude'] = np.abs(fimps.contribution)
    fimps.sort_values('magnitude', inplace=True, ascending=False)
    fimps['rank_rf'] = range(len(fimps))

    return fimps


def wordcloud(fimps, ax=None):

    def color_func(word, *args, **kwargs):
        row = fimps.loc[fimps.feature == word]
        if row.contribution.item() < 0:
            return 'orange'
        else:
            return 'blue'

    cloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="white").fit_words(dict(zip(fimps['feature'],
                                                                   fimps['magnitude'])))

    cloud.recolor(color_func=color_func)

    if ax is None:
        plt.figure(figsize=(15, 10))
        ax = plt.gca()

    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")


def run_tree_interpreter(grid_search_rf, data):
    X = grid_search_rf.best_estimator_['vect'].transform(data).toarray()
    X = grid_search_rf.best_estimator_['tfidf'].transform(X).toarray()
    prediction, bias, contributions = ti.predict(grid_search_rf.best_estimator_['clf'], X)

    return prediction, bias, contributions


def get_ti_feature_contributions_for_instance_i(i, contributions, grid_search_rf):
    result = pd.DataFrame()
    result['feature'] = grid_search_rf.best_estimator_['vect'].get_feature_names()
    result['contribution'] = contributions[i, :, 0]
    result['magnitude'] = np.absolute(contributions[i, :, 0])

    result = result.loc[~result.feature.apply(lambda x: any(char.isdigit() for char in x))]
    result.sort_values(by='magnitude', inplace=True, ascending=False)
    result['rank_ti'] = range(len(result))

    return result


def get_ti_feature_contributions_average(contributions, grid_search_rf):
    result = pd.DataFrame()

    result['contribution'] = (pd.DataFrame(contributions[:, :, 0],
                                           columns=grid_search_rf.best_estimator_['vect']
                                           .get_feature_names()
                                           ).mean(axis=0))

    result = result.reset_index().rename(columns={'index': 'feature'})
    result['magnitude'] = np.absolute(result['contribution'])
    result = result.loc[~result.feature.apply(lambda x: any(char.isdigit() for char in x))]
    result.sort_values(by='magnitude', inplace=True, ascending=False)
    result['rank_ti'] = range(len(result))

    return result

def get_lime_explanation_instance(grid_search_clf, data, index_to_explain, ns=500):
    
    X = grid_search_clf.best_estimator_['vect'].fit_transform(data).toarray()
    X = grid_search_clf.best_estimator_['tfidf'].fit_transform(X).toarray()

    lime = fatf_surrogates.TabularBlimeyLime(
        X,
        grid_search_clf.best_estimator_['clf'],
        feature_names=grid_search_clf.best_estimator_['vect'].get_feature_names(),
        class_names=['np', 'pc']
    )

    lime_explanation = lime.explain_instance(
        X[index_to_explain, :], samples_number=ns
    )

    result = pd.DataFrame()
    result['feature'] = grid_search_clf.best_estimator_['vect'].get_feature_names()
    result['contribution'] = [
        lime_explanation['pc'][key] for key in lime_explanation['pc'].keys()
    ]
    result['magnitude'] = [np.abs(c) for c in result['contribution']]
    result.sort_values('magnitude', ascending=False, inplace=True)
    result['rank_lime'] = range(len(result))

    return result

def get_lime_explanation_average(grid_search_clf, data, ns=500):
    
    X = grid_search_clf.best_estimator_['vect'].fit_transform(data).toarray()
    X = grid_search_clf.best_estimator_['tfidf'].fit_transform(X).toarray()

    lime = fatf_surrogates.TabularBlimeyLime(
        X,
        grid_search_clf.best_estimator_['clf'],
        feature_names=grid_search_clf.best_estimator_['vect'].get_feature_names(),
        class_names=['np', 'pc']
    )

    result = pd.DataFrame()
    result['feature'] = grid_search_clf.best_estimator_['vect'].get_feature_names()
    average_contribution = np.zeros(len(result.feature))

    for i in range(len(X)):

        index_to_explain = i
        lime_explanation = lime.explain_instance(
            X[index_to_explain, :], samples_number=ns
        )
        for ki, key in enumerate(lime_explanation['pc'].keys()):
            average_contribution[ki] += lime_explanation['pc'][key] 


    result['contribution'] = average_contribution / len(X)
    result['magnitude'] = [np.abs(c) for c in result['contribution']]
    result.sort_values('magnitude', ascending=False, inplace=True)
    result['rank_lime'] = range(len(result))

    return result

def get_shap_value_average(grid_search_clf, data):

    X = grid_search_clf.best_estimator_['vect'].fit_transform(data).toarray()
    X = grid_search_clf.best_estimator_['tfidf'].fit_transform(X).toarray()

    features = grid_search_clf.best_estimator_['vect'].get_feature_names()

    X_train_df = pd.DataFrame()

    for i, fi in enumerate(features):
        X_train_df[fi] = X[:,i]

    explainer = shap.Explainer(grid_search_clf.best_estimator_['clf'])
    shap_values = explainer(X_train_df)

    result = pd.DataFrame()
    result['feature'] = features
    result['contribution'] = shap_values.values[:,:,1].mean(axis=0)
    result['magnitude'] = [np.abs(c) for c in result.contribution]
    result.sort_values('magnitude', ascending=False, inplace=True)
    result['rank_shap'] = range(len(result))

    return result

def get_shap_values(grid_search_clf, documents):

    X = grid_search_clf.best_estimator_['vect'].fit_transform(documents).toarray()
    X = grid_search_clf.best_estimator_['tfidf'].fit_transform(X).toarray()

    features = grid_search_clf.best_estimator_['vect'].get_feature_names()

    X_train_L, X_test_L = train_test_split(
        X, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_train_df = pd.DataFrame()

    for i, fi in enumerate(features):
        X_train_df[fi] = X_train_L[:,i]

    explainer = shap.Explainer(grid_search_clf.best_estimator_['clf'])
    shap_values = explainer(X_train_df)

    return shap_values

