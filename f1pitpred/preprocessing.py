import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

## Rainfall -------------------------------------------------------------------

def _process_rainfall(df): # Removes races with rain
    rain = df.groupby(['Year', 'RoundNumber', 'DriverNumber'])['Compound'].transform(lambda x: x[x.str.contains('INTERMEDIATE|WET')].count())
    return df[rain == 0].reset_index(drop=True)

## Pitstops -------------------------------------------------------------------
def _process_pitstops(df):
    df['PitStatus'] = df.groupby(['Year', 'RoundNumber', 'DriverNumber'])['PitStatus'].shift(-1, fill_value='NoPit')
    return df
## Incomplete races -----------------------------------------------------------
def _incomplete_races(df):
    return df.groupby(['Year', 'RoundNumber', 'DriverNumber']).filter(lambda x: x['LapNumber'].max() + 3 >= x['TotalLaps'].max()).reset_index(drop=True)
## TrackName ------------------------------------------------------------------
def _process_track_name(df):
    df['Track'] = df['Track'].str.replace(' ', '_')
    return df
## TrackStatus ----------------------------------------------------------------

def _trackStatus_to_binary(df):
    trackStatus = df['TrackStatus']
    status = pd.Series(
        np.zeros(6, dtype=np.bool_),
        index=['Green', 'Yellow', 'SC', 'Red', 'VSC', 'SC_ending']
    )
    if "1" in trackStatus:
        status['Green'] = True
    if "2" in trackStatus:
        status['Yellow'] = True
    if "4" in trackStatus:
        status['SC'] = True
    if "5" in trackStatus:
        status['Red'] = True
    if "6" in trackStatus:
        status['VSC'] = True
    if "7" in trackStatus:
        status['SC_ending'] = True
    return status

def _process_trackStatus(df):
    trackStatuses = df.apply(_trackStatus_to_binary, axis=1)
    return pd.concat([df.drop('TrackStatus', axis=1), trackStatuses], axis=1)

## Missing Data ----------------------------------------------------------------

def _process_missing_values(df):
    # TODO fill the missing values better
    df.fillna({
        'DistanceToDriverAhead': -1,
        'GapToLeader': -1,
        'IntervalToPositionAhead': -1,
    }, inplace=True)

    # drop all rows with missing laptime
    df.dropna(subset=['LapTime'], inplace=True)
    return df[df['LapNumber'] > 1].reset_index(drop=True)

## Datatypes -------------------------------------------------------------------

def _process_datatypes(df):
    # boolean
    df['Green'] = df['Green'].astype('bool')
    df['Yellow'] = df['Yellow'].astype('bool')
    df['SC'] = df['SC'].astype('bool')
    df['Red'] = df['Red'].astype('bool')
    df['VSC'] = df['VSC'].astype('bool')
    df['SC_ending'] = df['SC_ending'].astype('bool')
    df['IsAccurate'] = df['IsAccurate'].astype('bool')
    df['Rainfall'] = df['Rainfall'].astype('bool')
    # category
    df['DriverNumber'] = df['DriverNumber'].astype('category')
    df['Team'] = df['Team'].astype('category')
    #df['Compound'] = df['Compound'].astype('category')
    df['DriverAhead'] = df['DriverAhead'].astype('category')
    #df['Track'] = df['Track'].astype('category')
    # float
    df['LapStartTime'] = df['LapStartTime'].astype('float32')
    df['LapTime'] = df['LapTime'].astype('float32')
    df['DistanceToDriverAhead'] = df['DistanceToDriverAhead'].astype('float32')
    df['GapToLeader'] = df['GapToLeader'].astype('float32')
    df['IntervalToPositionAhead'] = df['IntervalToPositionAhead'].astype('float32')
    df['AirTemp'] = df['AirTemp'].astype('float32')
    df['Humidity'] = df['Humidity'].astype('float32')
    df['Pressure'] = df['Pressure'].astype('float32')
    df['TrackTemp'] = df['TrackTemp'].astype('float32')
    df['WindDirection'] = df['WindDirection'].astype('float32')
    df['WindSpeed'] = df['WindSpeed'].astype('float32')
    # int
    df['LapNumber'] = df['LapNumber'].astype('uint8')
    df['TyreLife'] = df['TyreLife'].astype('uint8')
    df['Stint'] = df['Stint'].astype('uint8')
    df['NumberOfPitStops'] = df['NumberOfPitStops'].astype('uint8')
    df['Position'] = df['Position'].astype('uint8')
    df['LapsToLeader'] = df['LapsToLeader'].astype('uint8')
    df['TotalLaps'] = df['TotalLaps'].astype('uint8')
    return df

## Add target ------------------------------------------------------------------

def _process_target(df):
    df['is_pitting'] = df['PitStatus'] == 'InLap'
    df['is_pitting'] = df['is_pitting'].astype('bool')
    return df

## Remove features -------------------------------------------------------------

def _process_remove_features(df):
    df.drop(['LapStartTime', 'DriverNumber', 'Team', 'DriverAhead', 
    'AirTemp', 'Humidity', 'Pressure', 'Rainfall', 'TrackTemp', 'WindDirection', 'WindSpeed',
    'PitStatus', 'IsAccurate', 'Year', 'RoundNumber'], axis=1, inplace=True)
    return df

## Feature encoding ------------------------------------------------------------

def _process_feature_encoding(df):
    categorical_features = ['Compound', 'Track']
    one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    one_hot_encoder.fit(df[categorical_features])
    one_hot_encoded = one_hot_encoder.transform(df[categorical_features])
    one_hot_encoded = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(categorical_features))
    df = df.join(one_hot_encoded)
    df.drop(categorical_features, axis=1, inplace=True)
    return df, one_hot_encoder

def _process_feature_encoding_new(df, encoder):
    categorical_features = ['Compound', 'Track']
    one_hot_encoded = encoder.transform(df[categorical_features])
    one_hot_encoded = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_features))
    df = df.join(one_hot_encoded)
    df.drop(categorical_features, axis=1, inplace=True)
    return df

def preprocess_pre_split(df):
    df = df.copy()
    df = _process_rainfall(df)
    df = _incomplete_races(df)
    df = _process_pitstops(df)
    df = _process_track_name(df)
    df = _process_missing_values(df)
    return df

def preprocess_post_split(df):
    df = _process_trackStatus(df)
    df = _process_datatypes(df)
    df = _process_remove_features(df)
    return df

def preprocess_post_split_train(df):
    df = df.copy()
    df = _process_target(df)
    df, encoder = _process_feature_encoding(df)
    df = preprocess_post_split(df)
    return df, encoder

def preprocess_post_split_test(df, encoder):
    df = df.copy()
    df = _process_target(df)
    df = _process_feature_encoding_new(df, encoder)
    df = preprocess_post_split(df)
    return df

def preprocess_new_data(df, encoder):
    df = df.copy()
    df = _process_rainfall(df)
    df = _incomplete_races(df)
    df = _process_track_name(df)
    df = _process_missing_values(df)
    df = _process_trackStatus(df)
    df = _process_datatypes(df)
    df = _process_target(df)
    df = _process_remove_features(df)
    df = _process_feature_encoding_new(df, encoder)
    return df

## Train test split ------------------------------------------------------------

def _get_races_grouped(df):
    return df.groupby(['Year', 'RoundNumber', 'DriverNumber'])

def get_train_test_split(df, test_size, return_groups=False, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    groups = _get_races_grouped(df).groups
    groups_keys = list(groups.keys())
    np.random.shuffle(groups_keys)
    test_groups = groups_keys[:int(len(groups_keys) * test_size)]
    train_groups = groups_keys[int(len(groups_keys) * test_size):]
    test = df[df.apply(lambda x: (x['Year'], x['RoundNumber'], x['DriverNumber']) in test_groups, axis=1)].reset_index(drop=True)
    train = df[df.apply(lambda x: (x['Year'], x['RoundNumber'], x['DriverNumber']) in train_groups, axis=1)].reset_index(drop=True)
    if return_groups:
        return train, test, train.groupby(['Year', 'RoundNumber', 'DriverNumber']).groups, test.groupby(['Year', 'RoundNumber', 'DriverNumber']).groups
    return train, test

def get_preprocessed_train_test_split(df, test_size, return_groups=False, random_state=None):
    df = preprocess_pre_split(df)
    train, test, train_groups, test_groups = get_train_test_split(df, test_size, return_groups=True, random_state=random_state)
    train, encoder = preprocess_post_split_train(train)
    test = preprocess_post_split_test(test, encoder)
    train.dropna(inplace=True)
    test.dropna(inplace=True)
    if return_groups:
        return train, test, train_groups, test_groups
    return train, test

def get_x_y(df):
    return df.drop(['is_pitting'], axis=1), df['is_pitting']