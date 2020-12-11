# Adapted from kgu/covid_xprize_pipeline/predictor_robojudge.ipynb

import os
import pandas as pd
import numpy as np
from predictor import XPrizePredictor

if __name__ == "__main__":
    # Main source for the training data
    DATA_FILE = 'data/OxCGRT_to_Nov_15.csv'
    FEAT_FILE = 'data/happiness_0.csv'

    if os.path.exists(FEAT_FILE):
        feat_df = pd.read_csv(FEAT_FILE,index_col=0)
        print('loaded extra feats')

    predictor = XPrizePredictor(DATA_FILE, static_df=feat_df)
    predictor.train()

