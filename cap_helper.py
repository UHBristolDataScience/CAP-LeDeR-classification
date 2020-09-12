import pandas as pd
import numpy as np
import re

DATA_DIR = '../data/'

"""
Notes on time conversion:
Where days are months are missing from the date it defaults to 1st day/month. So times before DOD will be overestimated. One option would be to use years rather than months.
Probably missing some patterns below, but this should cover most cases (notably 21 Jan 2020 will omit leave a hanging 21). There is one typo written '22/112007' which fails silently (just replaces date with a space). And other dates in formt 21-01-2020 which fail.
"""


def load_data():

    return pd.read_excel(DATA_DIR + '20191028_committee_reviews_nlp_code.xlsx',
                         sheet_name='_20191028_committee_reviews_nlp')


def concatenate_feature_columns(df, columns=None):

    if columns is None:
        # create column with concatenation of all columns for any case,
        # except for the ones we are trying to predict (the last two)
        cols = df.columns
        cols = cols[2:27]  # This excludes 'cp1id', 'cp1vig date completed', 'cp1vig summary'
        cols = [c for c in cols if 'palliative' not in c]  # remove palliative column because confounded
    else:
        cols = columns

    df['combined'] = df[cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    return df


def add_dates(df):

    deaths = pd.read_csv(DATA_DIR + '20200423_extra_dod_dodx.txt')
    deaths = deaths.merge(pd.read_csv(DATA_DIR + '20200429_cap_id_lookup.txt'), on='cp1random_id_5_char')
    df = df.merge(deaths, on='cp1id')

    df['cnr19datedeath'] = pd.to_datetime(df['cnr19datedeath'], dayfirst=True)
    df['cnr_date_pca_diag'] = pd.to_datetime(df['cnr_date_pca_diag'], dayfirst=True)
    df['cp1vig date completed'] = pd.to_datetime(df['cp1vig date completed'], dayfirst=True)

    return df


def add_reviewer_ids(df):

    reviewers = pd.read_csv(DATA_DIR + '20191119_committee_reviews_nlp_code_update.csv')
    df = df.merge(reviewers, on='cp1id')

    return df


def get_reviewer_counts(df):

    return (df[['cp1id', 'vig_author']].groupby('vig_author')
                                       .count()
                                       .rename(columns={'cp1id': 'number of reviews'}))


def get_date_diff_string(date1, date2, months=True):

    if months:
        approximate_months = np.round((date1 - date2).days/30.417)
        if approximate_months > 0:
            diff_string = "moaf"
        else:
            diff_string = "mobf"
        return str(int(np.abs(approximate_months))) + diff_string
    else:
        approximate_years = np.round((date1 - date2).days/365.25)
        if approximate_years > 0:
            diff_string = "yeaf"
        else:
            diff_string = "yebf"
        return str(int(np.abs(approximate_years))) + diff_string


def replace_single_matches(text, pattern, relative_to, verbose=False):

    match = re.search(pattern, text)
    while match is not None:

        try:
            repl = get_date_diff_string(pd.to_datetime(match.group(), dayfirst=True), relative_to)

        except ValueError:
            if verbose:
                print("Invaldid date format: ", match)
            repl = ' '

        text = text[0:match.span()[0]] + repl + text[match.span()[1]:]
        match = re.search(pattern, text)

    return text


def replace_dates(text, relative_to=pd.to_datetime('01/01/2010')):

    patterns = ['\d{1,2}\/\d{1,2}\/\d{4}',  # format:21/02/2020
                '\d{1,2}\/\d{4}',           # format:02/2020
                '(?:January|February|March|April|May|June|July|August|September|October|November|December)[\s-]\d{1,2}[\s-]\d{2,4}',  # format: January 02 2020
                '(?:January|February|March|April|May|June|July|August|September|October|November|December)[\s-]\d{2,4}',  # format: January 2020
                '(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\s-]\d{2,4}'  # format: Jan 2020
                '(?:19|20)\d{2}']  # format: 2020

    for pattern in patterns:
        text = replace_single_matches(text, pattern, relative_to)

    return text


def convert_dates_relative(df):

    df['combined'] = [replace_dates(row.combined, row['cnr19datedeath']) for ri, row in df.iterrows()]
    return df


def get_easy_and_hard_cases(df, subset_x, subset_y):

    easy = [1, 5]
    hard = [2, 4]

    easy_x = []
    hard_x = []
    easy_y = []
    hard_y = []

    for i, xi in enumerate(subset_x):
        if df.loc[subset_y.index[i]].cp1do_cod_route in easy:
            easy_x.append(xi)
            easy_y.append(subset_y.iloc[i])
        elif df.loc[subset_y.index[i]].cp1do_cod_route in hard:
            hard_x.append(xi)
            hard_y.append(subset_y.iloc[i])
        else:
            print("Not easy or hard: ", i)

    return easy_x, hard_x, easy_y, hard_y
