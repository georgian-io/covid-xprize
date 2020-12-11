# Adapted from kgu/covid_xprize_pipeline/predictor_robojudge.ipynb

import os
import pandas as pd
import numpy as np
from predictor import XPrizePredictor

def predict(start_date: str,
            end_date: str,
            path_to_ips_file: str,
            output_file_path,
            model_path) -> None:
    #os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    # !!! YOUR CODE HERE !!!
    if FEAT_FILE and os.path.exists(FEAT_FILE):
        print('used extra features')
        static_feat_df = pd.read_csv(FEAT_FILE, index_col=0)
    else:
        print('did not use extra features')
        static_feat_df = None

    predictor = XPrizePredictor("data/OxCGRT_to_Nov_15.csv", saved_model_path=model_path,
                                static_df= static_feat_df)
    # Generate the predictions
    preds_df = predictor.predict(start_date, end_date, path_to_ips_file)
    # Create the output path
    # Save to a csv file
    preds_df.to_csv(output_file_path, index=False)
    print(f"Saved predictions to {output_file_path}")

if __name__ == "__main__":
    FEAT_FILE = 'data/happiness_0.csv'

    if os.path.exists(FEAT_FILE):
        feat_df = pd.read_csv(FEAT_FILE,index_col=0)
        print('loaded extra feats')
    else:
        feat_df = None

    #predictor._save_predictor(SAVE_PATH)
    output_file = "val_georgian.csv"
    predict("2020-11-16", "2020-12-07", "data/OxCGRT_Nov_16_to_Dec_7.csv", output_file, "checkpoints/epoch=11.ckpt")

