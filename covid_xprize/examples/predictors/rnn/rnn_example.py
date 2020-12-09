import os
import sys
sys.path.append("..")

import torch
from torch.utils.data import TensorDataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything

import numpy as np

from data_utils import _prepare_dataframe, _most_affected_geos, _create_country_samples

class Simple_RNN(pl.LightningModule):

    def __init__(self, nb_context, nb_action, nb_lookback, train_data_size, val_data_size):
        super().__init__()
        rnn_context_hidden_size = nb_context*4
        rnn_action_hidden_size = nb_action*4

        self.train_data_size = train_data_size
        self.val_data_size = val_data_size

        self.rnn_context = torch.nn.RNN(input_size=nb_context, hidden_size=rnn_context_hidden_size)
        self.rnn_action = torch.nn.RNN(input_size=nb_action, hidden_size=rnn_action_hidden_size)
        self.linear_context = torch.nn.Linear(rnn_context_hidden_size, 1)
        self.linear_action = torch.nn.Linear(rnn_action_hidden_size, 1)
        self.linear_combine_contexdt_action = torch.nn.Linear(nb_lookback*2, 1)
        self.l1_loss = torch.nn.L1Loss(reduction="mean")

    def predict(self, x_context, x_action):
        context_out = self.rnn_context(x_context)
        context_out = context_out[0]
        context_out = self.linear_context(context_out)
        context_out = context_out.squeeze(-1)

        action_out = self.rnn_action(x_action)
        action_out = action_out[0]
        action_out = self.linear_action(action_out)
        action_out = action_out.squeeze(-1)

        final_out = self.linear_combine_contexdt_action(torch.cat((context_out, action_out), dim=1))
        final_out = final_out.squeeze(-1)
        return final_out

    def forward(self, x_context, x_action, y):
        y_hat = self.predict(x_context, x_action)
        loss = self.l1_loss(y, y_hat)
        return loss

    def training_step(self, batch, batch_idx):
        x_context, x_action, y = batch[0], batch[1], batch[2]
        train_loss = self(x_context, x_action, y)
        self.log("training_loss", train_loss, on_step=False, on_epoch=True, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        x_context, x_action, y = batch[0], batch[1], batch[2]
        val_loss = self(x_context, x_action, y)
        self.log("validation_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss

    def validation_epoch_end(self, validation_step_outputs):
        avg_val_loss = 0
        for output in validation_step_outputs:
            avg_val_loss += output
        avg_val_loss = avg_val_loss / self.val_data_size
        self.log("avg validation loss", avg_val_loss)
        print(f"\n average validation loss: {avg_val_loss}")
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


class RNN_Data_Module(pl.LightningDataModule):

    def __init__(self, train_dataloader):
        super().__init__()
        self.train_dataloader = train_dataloader

    def train_dataloader(self):
        return self.train_dataloader

if __name__ == "__main__":
    USE_GPU = True
    MAX_NB_COUNTRIES = 20
    NB_LOOKBACK_DAYS = 21
    DATA_FILE_PATH = "/home/ubuntu/zilun/covid-xprize/covid_xprize/examples/predictors/data/OxCGRT_latest.csv"

    SEED = 32
    MAX_EPOCHS = 5

    seed_everything(SEED)

    df = _prepare_dataframe(DATA_FILE_PATH)
    print(df.head())

    geos = _most_affected_geos(df, MAX_NB_COUNTRIES, NB_LOOKBACK_DAYS)
    country_samples = _create_country_samples(df, geos)
    print("Finished creating numpy arrays")

    # Aggregate data for training
    all_X_context_list = [country_samples[c]['X_train_context']
                          for c in country_samples]
    all_X_action_list = [country_samples[c]['X_train_action']
                         for c in country_samples]
    all_y_list = [country_samples[c]['y_train']
                  for c in country_samples]
    X_context = np.concatenate(all_X_context_list)
    X_action = np.concatenate(all_X_action_list)
    y = np.concatenate(all_y_list)
    print("Finished creating training data")

    # Clip outliers
    MIN_VALUE = 0.
    MAX_VALUE = 2.
    X_context = np.clip(X_context, MIN_VALUE, MAX_VALUE)
    y = np.clip(y, MIN_VALUE, MAX_VALUE)
    nb_context = X_context.shape[-1]
    nb_action = X_action.shape[-1]

    # Convert data in numpy arrays to tensors
    X_context = torch.Tensor(X_context)
    X_action = torch.Tensor(X_action)
    y = torch.Tensor(y)

    # Create training and validation data loader
    train_data_size = int(len(y)*0.95)
    val_data_size = len(y) - train_data_size
    all_dataset = TensorDataset(X_context, X_action, y)
    train_dataset, val_dataset = torch.utils.data.random_split(all_dataset,
                                                               [train_data_size, val_data_size],
                                                               generator=torch.Generator().manual_seed(42))
    train_dataloader = DataLoader(train_dataset, num_workers=4)
    val_dataloader = DataLoader(val_dataset, num_workers=4)

    # Create Pytorch Lightning trainer
    simple_rnn = Simple_RNN(
                            nb_context = nb_context,
                            nb_action = nb_action,
                            nb_lookback = NB_LOOKBACK_DAYS,
                            train_data_size = train_data_size,
                            val_data_size = val_data_size)
    dm = RNN_Data_Module(train_dataloader)

    trainer = Trainer(max_epochs = MAX_EPOCHS,
                      gpus = 1 if USE_GPU else 0,
                      deterministic=True,
                     )

    # Start training
    trainer.fit(simple_rnn, train_dataloader, val_dataloader)
