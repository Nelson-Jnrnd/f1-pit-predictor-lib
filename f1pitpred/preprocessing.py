import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

## Rainfall -------------------------------------------------------------------

def _process_rainfall(df): # Removes races with rain
    rain = df.groupby(['Year', 'RoundNumber', 'DriverNumber'])['Compound'].transform(lambda x: x[x.str.contains('INTERMEDIATE|WET')].count())
    return df[rain == 0].reset_index(drop=True)

## Pitstops -------------------------------------------------------------------
def _process_pitstops(df):
    df['PitStatusShift'] = df.groupby(['Year', 'RoundNumber', 'DriverNumber'])['PitStatus'].shift(-1, fill_value='NoPit')
    return df

## Tires ----------------------------------------------------------------------
def _process_tires(df):
    df['NextCompound'] = df.groupby(['Year', 'RoundNumber', 'DriverNumber'])['Compound'].shift(-2)
    return df
## Incomplete races -----------------------------------------------------------
def _incomplete_races(df):
    return df.groupby(['Year', 'RoundNumber', 'DriverNumber']).filter(lambda x: x['LapNumber'].max() + 3 >= x['TotalLaps'].max()).reset_index(drop=True)
## TrackName ------------------------------------------------------------------

### Remove unusual races
def _remove_unusual_races(df): # Remove races with more than 4 pitstops
    nb_pitstops = df.groupby(['Year', 'RoundNumber', 'DriverNumber'])['NumberOfPitStops'].transform(lambda x: x.max())
    return df[nb_pitstops <= 4].reset_index(drop=True)

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
    df['DriverAhead'] = df['DriverAhead'].astype('str')
    # TODO fill the missing values better
    df.fillna({
        'DistanceToDriverAhead': -1,
        'GapToLeader': -1,
        'IntervalToPositionAhead': -1,
        'DriverAhead': -1
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
    df['is_pitting'] = df['PitStatusShift'] == 'InLap'
    df['is_pitting'] = df['is_pitting'].astype('bool')
    df = df.loc[df['PitStatusShift'] != 'OutLap']
    df = df.loc[df['PitStatus'] != 'OutLap']
    return df

## Add features ----------------------------------------------------------------

def _process_add_features(df):
    #df['RacePercentage'] = df['LapNumber'] / df['TotalLaps']
    #df['LapTimeDiff'] = df.groupby(['Year', 'RoundNumber', 'DriverNumber'])['LapTime'].diff()
    return df

## Remove features -------------------------------------------------------------

def _get_features_to_remove():
    return ['LapStartTime', 'DriverNumber', 'Team', 'DriverAhead', 
    'AirTemp', 'Humidity', 'Pressure', 'Rainfall', 'TrackTemp', 'WindDirection', 'WindSpeed',
    'PitStatus', 'PitStatusShift', 'IsAccurate', 'Year', 'RoundNumber', 'NumberOfPitStops'] #'LapNumber', 'TotalLaps']

def _process_remove_features(df):
    df.drop(_get_features_to_remove(), axis=1, inplace=True)
    return df

## Feature encoding ------------------------------------------------------------

def _process_feature_encoding(df):
    if 'NextCompound' in df.columns:
        categorical_features = ['Compound', 'Track', 'NextCompound']
    else:
        categorical_features = ['Compound', 'Track']
    one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    one_hot_encoder.fit(df[categorical_features])
    one_hot_encoded = one_hot_encoder.transform(df[categorical_features])
    one_hot_encoded = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(categorical_features))
    df = df.join(one_hot_encoded)
    df.drop(categorical_features, axis=1, inplace=True)
    return df, one_hot_encoder

def _process_feature_encoding_new(df, encoder):
    if 'NextCompound' in df.columns:
        categorical_features = ['Compound', 'Track', 'NextCompound']
    else:
        categorical_features = ['Compound', 'Track']
    one_hot_encoded = encoder.transform(df[categorical_features])
    one_hot_encoded = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_features))
    df = df.join(one_hot_encoded)
    df.drop(categorical_features, axis=1, inplace=True)
    return df

def preprocess_pre_split(df, target, drop=True):   
    df = df.copy()
    df = _process_rainfall(df)
    df = _incomplete_races(df)
    df = _remove_unusual_races(df)
    if target == 'pit':
        df = _process_pitstops(df)
    elif target == 'tire':
        df = _process_pitstops(df)
        df = _process_tires(df)
        if drop:
            df = df.loc[df['PitStatusShift'] == 'InLap'].reset_index(drop=True)
    df = _process_track_name(df)
    df = _process_missing_values(df)
    df = _process_target(df)
    df = _process_trackStatus(df)
    df = _process_add_features(df)
    return df

def preprocess_post_split_train(df):
    df = df.copy()
    df, encoder = _process_feature_encoding(df)
    return df, encoder

def preprocess_post_split_test(df, encoder):
    df = df.copy()
    df = _process_feature_encoding_new(df, encoder)
    return df

def preprocess_new_data(df, encoder, target='pit'):
    df = df.copy()
    df = _process_rainfall(df)
    df = _incomplete_races(df)
    if target == 'pit':
        df = _process_pitstops(df)
    elif target == 'tire':
        df = _process_pitstops(df)
        df = _process_tires(df)
    df = _process_track_name(df)
    df = _process_missing_values(df)
    df = _process_feature_encoding_new(df, encoder)
    df = _process_trackStatus(df)
    df = _process_datatypes(df)
    df = _process_target(df)
    df = _process_add_features(df)
    df = _process_remove_features(df)
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

def get_preprocessed_train_test_split(df, test_size, return_groups=False, random_state=None, target='pit', drop=True):
    df = preprocess_pre_split(df, target, drop=drop)
    train, test, train_groups, test_groups = get_train_test_split(df, test_size, return_groups=True, random_state=random_state)
    train, encoder = preprocess_post_split_train(train)
    test = preprocess_post_split_test(test, encoder)
    train = _process_remove_features(train)
    test = _process_remove_features(test)
    train.dropna(inplace=True)
    test.dropna(inplace=True)
    if return_groups:
        return train, test, encoder, train_groups, test_groups
    return train, test, encoder

def get_x_y_pit(df):
    return df.drop(['is_pitting'], axis=1), df['is_pitting']

def get_x_y_tires(df):
    return df.drop(['is_pitting', 'NextCompound_SOFT', 'NextCompound_MEDIUM', 'NextCompound_HARD', 'NextCompound_nan'], axis=1), df[['NextCompound_SOFT', 'NextCompound_MEDIUM', 'NextCompound_HARD']]


## Time series management ------------------------------------------------------

def create_sequences(data, sequence_length):
    # Sort the data by 'LapStartTime' to ensure temporal order
    data = data.sort_values('LapStartTime')

    # Group the data by races using 'DriverNumber', 'RoundNumber', and 'Year'
    grouped = data.groupby(['DriverNumber', 'RoundNumber', 'Year'])

    # Initialize lists to store sequences and corresponding targets
    sequences = []
    targets = []

    # Process each race group separately
    for _, race_group in grouped:
        # Extract lap data for the race
        laps_data = race_group.values

        # Calculate the total number of sequences that can be created for this race
        num_sequences = len(laps_data) - sequence_length

        # Create overlapping sequences for this race
        for i in range(num_sequences):
            sequence = laps_data[i:i+sequence_length]
            target = laps_data[i+sequence_length]  # LapNumber of the next lap (target)

            # Append the sequence and target to the respective lists
            #sequences.append(pd.DataFrame(sequence, columns=data.columns))
            sequences.append(sequence)
            #targets.append(pd.Series(target, index=data.columns))
            targets.append(target)
        
    # Convert the lists to numpy arrays
    sequences = np.array(sequences)
    targets = np.array(targets)


    return sequences, targets

def get_preprocessed_sequences(df, test_size, sequence_length, return_groups=False, random_state=None, target='pit', drop=False):
    """"
    Returns sequences and targets for train and test sets

    Parameters
    ----------
    df : pandas.DataFrame
        Data to be split
    test_size : float
        Fraction of data to be used for test set
    sequence_length : int
        Length of the sequences to be created
    return_groups : bool, default False
        Whether to return the groups used for the train test split
    random_state : int, default None
        Random state for reproducibility
    target : str, default 'pit'
        Target variable to be used for the sequences
    Returns
    -------
    sequences_train : list
        List of sequences for the train set
    targets_train : list
        List of targets for the train set
    sequences_test : list
        List of sequences for the test set
    targets_test : list
        List of targets for the test set
    encoder : sklearn.preprocessing.LabelEncoder
        Encoder used to encode the categorical variables
    """
    df = preprocess_pre_split(df, target, drop=False)
    train, test, train_groups, test_groups = get_train_test_split(df, test_size, return_groups=True, random_state=random_state)
    train, encoder = preprocess_post_split_train(train)
    test = preprocess_post_split_test(test, encoder)
    train.dropna(inplace=True)
    test.dropna(inplace=True)

    sequences_train, targets_train = create_sequences(train, sequence_length)
    sequences_test, targets_test = create_sequences(test, sequence_length)
    
    cols = train.columns.to_numpy()
    target_col = ['is_pitting']
    if target == 'tire':
        target_col = ['is_pitting', 'NextCompound_SOFT', 'NextCompound_MEDIUM', 'NextCompound_HARD', 'NextCompound_nan']
        if drop:
            pit_col_id = np.where(cols == 'is_pitting')[0][0]
            pitting_train = np.where(targets_train[:, pit_col_id] == 1)
            pitting_test = np.where(targets_test[:, pit_col_id] == 1)
            sequences_train = sequences_train[pitting_train]
            sequences_test = sequences_test[pitting_test]
            targets_train = targets_train[pitting_train]
            targets_test = targets_test[pitting_test]

    cols_id_delete = list(map(lambda col : np.where(cols == col)[0][0], _get_features_to_remove() + target_col))
    
    sequences_train = np.delete(sequences_train, cols_id_delete, axis=2)
    sequences_test = np.delete(sequences_test, cols_id_delete, axis=2)
    
    if target == 'tire':
        target_col = ['is_pitting', 'NextCompound_SOFT', 'NextCompound_MEDIUM', 'NextCompound_HARD']
    col = np.where(np.isin(cols, target_col))[0]
    targets_train = targets_train[:, col]
    targets_test = targets_test[:, col]

    if target == 'pit':
        targets_train = np.ravel(targets_train)
        targets_test = np.ravel(targets_test)

    sequences_train = np.array(sequences_train).astype(np.float32)
    targets_train = np.array(targets_train).astype(np.float32)
    sequences_test = np.array(sequences_test).astype(np.float32)
    targets_test = np.array(targets_test).astype(np.float32)
    
    return sequences_train, targets_train, sequences_test, targets_test, encoder


def get_preprocessed_sequences_new(df, encoder, seq_size, target='pit'):
    preproc1 = preprocess_pre_split(df, target)
    preproc2 = preprocess_post_split_test(preproc1, encoder)
    preproc2.dropna(inplace=True)
    
    seq_x, _ = create_sequences(preproc2, seq_size)
    
    cols = preproc2.columns.to_numpy()
    cols_id_delete = list(map(lambda col : np.where(cols == col)[0][0], _get_features_to_remove() + ['is_pitting']))
    
    seq_x = np.delete(seq_x, cols_id_delete, axis=-1)
    
    return seq_x, np.delete(cols, cols_id_delete)