""" Define BaseDataset
"""

import pandas as pd
import numpy as np
import datetime
from typing import Optional, Tuple


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
    
    def align_dataset(self, energy: pd.DataFrame, info: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align the order of the labels in the infodata with the data in the energy dataset.
        
        Args:
            energy: Pandas DataFrame with the energy data
            info: Pandas DataFrame with the infodata information
        
        Returns:
            Tuple of two Pandas DataFrames: aligned energy data and infodata
        """
        # Initialize list to store valid columns in both dataframes
        valid_col = []
        
        for col in energy.columns:
            if col in info.index:
                valid_col.append(col)
        
        info = info.T[valid_col].T
        energy = energy[valid_col]
        
        # Return the aligned energy data and infodata
        return energy, info
    
    def trunc_data(self, energy: pd.DataFrame, start_date: Optional[datetime.datetime] = None, end_date: Optional[datetime.datetime] = None):
        """To prevent seasonal effects, truncate data from January 1st

        Args:
            energy (pd.DataFrame): energy consumption
            start_date (datetime.datetime): start date
            end_date (datetime.datetime): end date.

        Returns:
            pd.DataFrame: truncated energy consumption
        """
        energy = energy.loc[start_date:end_date, :]
        return energy

    def replace_invalid_value(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        replace invalid energy consumption value which is equals to 0
        because the sensor measurement cannot be zero.

        Args:
            df (pd.DataFrame): Energy consumption data

        Returns:
            pd.DataFrame: replaced energy consumption data
        """
        df[df == 0] = np.nan
        return df
    
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
