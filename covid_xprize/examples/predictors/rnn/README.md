### Steps to train and test RNN models with static features

1. Training: `python model_train.py`, trained model in the last epoch will be saved automatically by pytorch-lightning

2. Predicting: Replace path to the trained model in line 42 and run `python model_predict.py`, prediction file will be saved as "val_georgian.csv"

3. Evaluate: Run Chang's evaluation script with the prediction file, `python evaluate.py -f val_georgian.csv`

### Code for one hot encoding action features: 

1. line 136 of predictor.py (`def one_hot_action_input`) link: https://github.com/georgianpartners/covid-xprize/blob/3c36d6f6e04dc7b7b8a8ab8ad9d92c61f371e067/covid_xprize/examples/predictors/rnn/predictor.py#L136

2. line 420 of predictor.py (`def one_hot_encoding_action_list`) link: https://github.com/georgianpartners/covid-xprize/blob/3c36d6f6e04dc7b7b8a8ab8ad9d92c61f371e067/covid_xprize/examples/predictors/rnn/predictor.py#L420

Note that training and predicting both use one hot encoding of the action features.
