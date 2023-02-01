from collections import namedtuple
import random
import glob
import os
import datetime
import tqdm
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


Task = namedtuple("Task", ["x", "y", "task_info"])


class NOAA_GSOD_MetaDset(Dataset):
    def __init__(
        self,
        train_days=20,
        val_days=20,
        dset_dir="/noaa_gsod/",
        split="train",
    ):
        self.dset_dir = dset_dir
        self.train_days = train_days
        self.val_days = val_days
        self.files = self._preprocess(glob.glob(os.path.join(dset_dir, "*/*.csv")))

        assert split in ["train", "val", "test"]
        self.split = split
        # deterministic shuffle
        old_state = random.getstate()
        random.seed(402)
        random.shuffle(self.files)
        random.setstate(old_state)
        val = slice(0, 5000)
        test = slice(5000, 6000)
        train = slice(6000, None)
        if self.split == "train":
            self.files = self.files[train]
        elif self.split == "val":
            self.files = self.files[val]
        else:
            self.files = self.files[test]

    def _preprocess(self, files):
        if self.val_days == 20 and self.train_days == 20:
            with open("20x20_noaa_shortcut.csv", "r") as f:
                clean_files = [x.strip() for x in f.readlines()]
            print("Skipping preprocessing using pre-saved results...")
        else:
            clean_files = []
            removed = 0
            print("Removing tasks with too few datapoints for specified task size...")
            for file in tqdm.tqdm(files):
                raw_df = pd.read_csv(file)
                df = raw_df[raw_df.TEMP != 9999.9]
                if len(df) >= self.train_days + self.val_days:
                    clean_files.append(file)
                else:
                    removed += 1
            print(f"Removed {(removed / float(len(files))) * 100:.3f}% of tasks")
        return clean_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        file = self.files[i]
        raw_df = pd.read_csv(file)
        year = Path(file).parents[0].parts[-1]
        station_name = raw_df["NAME"].values[0]
        # step 1: get rid of days that do not have a temperature reading
        df = raw_df[raw_df.TEMP != 9999.9]
        assert len(df) >= self.train_days + self.val_days
        # step 2: pick the days we will use
        df = df.sample(n=self.train_days + self.val_days, replace=False)

        # step 3: remove the columns we are not using
        # 'attribute' columns record the number of measurments that
        # were used to determine the daily summary
        df = df.drop(
            columns=[
                "STATION",
                "NAME",
                "TEMP_ATTRIBUTES",
                "DEWP",
                "DEWP_ATTRIBUTES",
                "PRCP_ATTRIBUTES",
                "SLP_ATTRIBUTES",
                "STP_ATTRIBUTES",
                "VISIB_ATTRIBUTES",
                "WDSP_ATTRIBUTES",
                "MAX",
                "MIN",
                "MAX_ATTRIBUTES",
                "MIN_ATTRIBUTES",
                "LATITUDE",
                "LONGITUDE",
            ]
        )

        # step 4: convert dates to integers 0-365
        # representing the days that have passed
        # since the start of the year
        def convert_date(date_time_obj):
            date = date_time_obj.date()
            year = date.year
            jan1 = datetime.date(year, 1, 1)
            time_since_jan1 = date - jan1
            return float(time_since_jan1.days) / 364.0

        dates = pd.to_datetime(df["DATE"])
        dates = dates.map(convert_date)
        df["DATE"] = dates

        # step 5: convert FRSHTT column into 6 separate one-hot columns
        # which indicate presence of Fog, Rain, Snow, Hail, Thunder, Tornado respectively.
        def one_hot_frshtt(frshtt):
            frshtt = str(frshtt).strip()
            cols = [0, 0, 0, 0, 0, 0]
            for i, var in enumerate(frshtt):
                if var == "1":
                    cols[i] = 1
            return np.array(cols)

        result = df["FRSHTT"].map(one_hot_frshtt)
        one_hot = np.concatenate(result.values).reshape(-1, 6).astype(np.float32)
        df = df.drop(columns=["FRSHTT"])
        df = df.astype(np.float32)
        df[["FOG", "RAIN", "SNOW", "HAIL", "THUNDER", "TORNADO"]] = one_hot

        # step 6: fix missing values. Missing measurments are usually given a large
        # constant in this dataset, e.g. missing windspeed is listed as 999.9 mph winds.
        # For stability we find values like this and replace them with 0.
        df = df.replace(99.99, 0.0)
        df = df.replace(999.9, 0.0)
        df = df.replace(9999.9, 0.0)

        # step 7: change sea pressure from millibars to bars
        df["SLP"] /= 1000.0
        df["STP"] /= 1000.0

        # step 8: change elevation from meters to kilometers
        df["ELEVATION"] /= 1000.0

        # step 9: fill NaNs that occasionally arise
        df = df.replace(np.nan, 0.0)

        # step 10: split temperature labels
        df = df.astype(np.float32)
        y = df["TEMP"]
        x = df.drop(columns=["TEMP"])

        # step 11: create torch tensors, do train/val split
        x = torch.Tensor(x.values)
        y = torch.Tensor(y.values).unsqueeze(1)
        perm = torch.randperm(x.shape[0])
        x = x[perm]
        y = y[perm]

        return Task(x=x, y=y, task_info=f"{station_name}, {year}")


if __name__ == "__main__":
    dset = NOAA_GSOD_MetaDset(split="val")
    for i in range(100):
        test = dset[i]
        print(test.task_info)
