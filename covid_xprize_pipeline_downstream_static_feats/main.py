import os
from predictor import XPrizePredictor

DATA_FILE = 'data/OxCGRT_to_Nov_15.csv'
NPIS_INPUT_FILE = 'data/OxCGRT_Nov_16_to_Dec_7.csv'
SAVE_PATH = 'save/for_inference/model.pt'
start_date = "2020-11-16"
end_date = "2020-12-07"

if __name__ == '__main__':
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    # predictor = XPrizePredictor(DATA_FILE)
    # predictor.train()
    # predictor._save_predictor(SAVE_PATH)
    predictor2 = XPrizePredictor(DATA_FILE, saved_model_path=SAVE_PATH)
    preds_df = predictor2.predict(start_date, end_date, NPIS_INPUT_FILE)

    print('here')