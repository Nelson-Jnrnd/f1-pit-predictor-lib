import os
import pandas as pd
import numpy as np

type_dict ={
    'LapStartTime': np.float32,
    'LapNumber': np.uint8,
    'LapTime': np.float32,
    'DriverNumber': "category",
    'Team' : "category",
    'Compound': "category",
    'TyreLife': np.uint8,
    'TrackStatus': "category",
    'Stint': np.uint8,
    'DistanceToDriverAhead': np.float32,
    'DriverAhead': "category",
    'PitStatus': "category",
    'IsAccurate': np.bool_,
    'Track': "category",
    'NumberOfPitStops': np.uint8,
    'Position' : np.uint8,
    'GapToLeader' : np.float32,
    'IntervalToPositionAhead' : np.float32,
    'LapsToLeader' : np.uint8,
    'TotalLaps' : np.uint8,
    'AirTemp': np.float32,
    'Humidity': np.float32,
    'Pressure': np.float32,
    'TrackTemp': np.float32,
    'WindDirection': np.float32,
    'WindSpeed': np.float32
    }

def load_from_csv(years, base_path='data', file_name='season.csv'):
    """Load data from csv files.

    Parameters
    ----------
    years : list of int
        List of years to load data for.
    base_path : str, optional
        Path to data folder.

    Returns
    -------
    DataFrame
        Dataframe containing the race data for the given years.
    """
    data = pd.DataFrame()

    for year in years:
        path = os.path.join(base_path, str(year), file_name)
        year_csv = pd.read_csv(
                path,
                dtype=type_dict,
        )
        data = pd.concat([
            data,
            year_csv
        ])
    return data