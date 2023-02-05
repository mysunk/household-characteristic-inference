""" Define BaseDataset
"""

import pandas as pd
import datetime


class BaseDataset:
    """Base class of Energy time series data and corresponding label
    """
    
    def wh_to_kwh(self, energy: pd.DataFrame) -> pd.DataFrame:
        """Unit of energy consumption unified to kWh

        Args:
            energy (pd.DataFrame): energy consumption with Wh

        Returns:
            pd.DataFrame: energy consumption with kWh
        """
        energy = energy / 1000
        return energy
    
    def trunc_data(self, energy: pd.DataFrame, trunc_date: datetime.datetime):
        """To prevent seasonal effects, truncate data from January 1st

        Args:
            energy (pd.DataFrame): energy consumption
            trunc_date (datetime.datetime): start date

        Returns:
            pd.DataFrame: truncated energy consumption
        """
        energy = energy.loc[trunc_date:, :]
        return energy
    
    def load_energy(self):
        pass

    def load_metadata(self):
        pass

    def preprocess_energy(self):
        pass

    def preprocess_label(self):
        pass

    def allign_dataset(self):
        pass
    
    def quantize_label(self):
        pass
