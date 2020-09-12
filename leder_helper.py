import pandas as pd
import numpy as np

DATA_DIR = ("//ads.bris.ac.uk/folders/Social Sciences and Law/Deaf_Studies/Deaf/nfrc/LeDeR confidential/Data Entry/"
            "Avon/reviews_leder_revised_codes/python_notebooks/")

"""
Notes on time conversion:
Where days are months are missing from the date it defaults to 1st day/month. So times before DOD will be overestimated. One option would be to use years rather than months.
Probably missing some patterns below, but this should cover most cases (notably 21 Jan 2020 will omit leave a hanging 21). There is one typo written '22/112007' which fails silently (just replaces date with a space). And other dates in formt 21-01-2020 which fail.
"""

def load_data():

    df = pd.read_csv(DATA_DIR + 'master_nvivo_df.csv', low_memory=False)
    df.drop(columns=['Unnamed: 0'], inplace=True)

    return df


def build_labels(df):

    poor_practice_indicators = [col for col in df.columns if ('Poor' in col and 'atext' not in col)]
    good_practice_indicators = [col for col in df.columns if ('Good' in col and 'atext' not in col)]

    df['Poor_practice_flag'] = df[poor_practice_indicators].sum(axis=1) > 0.0
    df['Good_practice_flag'] = df[good_practice_indicators].sum(axis=1) > 0.0

    return df


def get_practice_counts(df):

    return (df.groupby(by=['Poor_practice_flag', 'Good_practice_flag'])[['AAA_screening']]
              .count()
              .rename(columns={'AAA_screening': 'count'}))


def get_practice_columns(df, practice_type='all'):

    columns = None
    if practice_type == 'all':
        columns = [col for col in df.columns if 'practice' in col and 'Nursing_GP' not in col]
    elif practice_type == 'good':
        columns = [col for col in df.columns if 'Good' in col]
    elif practice_type == 'good':
        columns = [col for col in df.columns if 'Poor' in col]

    return columns


def concatenate_feature_columns(df, columns=None):

    if columns is None:
        text_columns = [col for col in df.columns if col not in get_practice_columns(df) and 'atext' in col]

    else:
        text_columns = columns

    df['combined'] = df[text_columns].replace(np.nan, '').apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    return df


def get_reviewer_counts(df):

    return (df[['AAA_screening', 'coder_atext']].groupby('coder_atext')
                                                .count()
                                                .rename(columns={'AAA_screening': 'number of reviews'}))


