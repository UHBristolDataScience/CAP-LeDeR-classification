import pandas as pd
import numpy as np
from treeinterpreter import treeinterpreter as ti
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt


def get_rf_feature_importances(grid_search_rf):
    fimps = pd.DataFrame()
    fimps['feature'] = grid_search_rf.best_estimator_['vect'].get_feature_names()
    fimps['contribution'] = grid_search_rf.best_estimator_['clf'].feature_importances_
    fimps['magnitude'] = np.abs(fimps.contribution)
    fimps.sort_values('magnitude', inplace=True, ascending=False)

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
    result.sort_values(by='contribution', inplace=True)

    return result


def get_ti_feature_contributions_average(contributions, grid_search_rf):
    result = pd.DataFrame()

    result['contribution'] = (pd.DataFrame(contributions[:, :, 0],
                                           columns=grid_search_rf.best_estimator_['vect']
                                           .get_feature_names()
                                           ).mean(axis=0))

    result = result.reset_index().rename(columns={'index': 'feature'})
    result['magnitude'] = np.absolute(result['contribution'])

    return result
