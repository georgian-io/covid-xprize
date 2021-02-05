import os
import pandas as pd

IP_FILE = "/home/ubuntu/mburhanpurkar/covid-xprize/" \
                  "2020-08-01_2020-08-04_prescriptions_example.csv"
hdf = pd.read_csv(IP_FILE,
                                        parse_dates=['Date'],
                                                              encoding="ISO-8859-1",
                                                                                dtype={"RegionName": str},
                                                                                                  error_bad_lines=True)

for country in hdf['CountryName'].unique():
    os.system(f"/home/ubuntu/anaconda3/envs/tianshou/bin/python run_sac_extra_states2.py {country}")
