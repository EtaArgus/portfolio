import os
import numpy as np
import pandas as pd
import glob
from sklearn.model_selection import train_test_split


os.chdir('C:\\Users\Charles\Documents\SOR\Dissertation\Model')


def vectorize(number_classes, j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((number_classes, 1))
    e[j] = 1.0
    return e


def read_parameters_data():
    # Read csv files
    path = r'C:\\Users\Charles\Documents\SOR\Dissertation\Model\Data\csv'
    all_files = glob.glob(os.path.join(path, "*.csv"))
    df_from_each_file = (pd.read_csv(f) for f in all_files)
    data_aux = pd.concat(df_from_each_file, ignore_index=True)
    # Change name of a column
    data_aux = data_aux.rename(columns={'objID': 'OBJID',
                                        'Unnamed: 1': 'dered_g-dered_r',
                                        'Unnamed: 2':
                                        'dered_r-dered_i'})
    data_aux['concentration'] = data_aux['petroR90_i'] / data_aux['petroR50_i']
    data = data_aux.drop(data_aux.columns[[8, 9]],
                         axis=1)
    return(data)


def prepare_data_gz2(cut, task):
    input_parameters = read_parameters_data()
    if task == 't01':
        # read data
        read_gz2 = pd.read_csv('Data\zoo2MainSpecz.csv', usecols=[
            'dr7objid',
            'gz2class',
            't01_smooth_or_features_a01_smooth_weighted_fraction',
            't01_smooth_or_features_a02_features_or_disk_weighted_fraction',
            't01_smooth_or_features_a03_star_or_artifact_weighted_fraction'])
        # change column names
        read_gz2 = read_gz2.rename(columns={
            'dr7objid': 'OBJID', 't01_smooth_or_features_a01_smooth'
            '_weighted_fraction': 't01_elliptical',
            't01_smooth_or_features_a02_features_or_disk_weighted_fraction':
            't01_spiral', 't01_smooth_or_features_a03_star_or_artifact_'
            'weighted_fraction': 't01_artifact'})
        # apply cut
        read_gz2 = read_gz2[((read_gz2['t01_elliptical'] >= cut) |
                             (read_gz2['t01_spiral'] >= cut) |
                             (read_gz2['t01_artifact'] >= cut))]
        # define GROUP column
        read_gz2.loc[read_gz2[[
            't01_elliptical', 't01_spiral', 't01_artifact']].idxmax(axis=1) ==
            't01_elliptical', 'GROUP'] = 0
        read_gz2.loc[read_gz2[[
            't01_elliptical', 't01_spiral', 't01_artifact']].idxmax(axis=1) ==
            't01_spiral', 'GROUP'] = 1
        read_gz2.loc[read_gz2[[
            't01_elliptical', 't01_spiral', 't01_artifact']].idxmax(axis=1) ==
            't01_artifact', 'GROUP'] = 2
    elif task == 't07':
        # read data
        read_gz2 = pd.read_csv('Data\zoo2MainSpecz.csv', usecols=[
            'dr7objid',
            'gz2class',
            't07_rounded_a16_completely_round_weighted_fraction',
            't07_rounded_a17_in_between_weighted_fraction',
            't07_rounded_a18_cigar_shaped_weighted_fraction'])
        # change column names
        read_gz2 = read_gz2.rename(columns={
            'dr7objid': 'OBJID',
            't07_rounded_a16_completely_round_weighted_fraction':
            't07_completely', 't07_rounded_a17_in_between_weighted_fraction':
            't07_between', 't07_rounded_a18_cigar_shaped_weighted_fraction':
            't07_shaped'})
        # apply cut
        read_gz2 = read_gz2[((read_gz2['t07_completely'] >= cut) |
                             (read_gz2['t07_between'] >= cut) |
                             (read_gz2['t07_shaped'] >= cut))]
        # define GROUP column
        read_gz2.loc[read_gz2[[
            't07_completely', 't07_between', 't07_shaped']].idxmax(axis=1) ==
            't07_completely', 'GROUP'] = 0
        read_gz2.loc[read_gz2[[
            't07_completely', 't07_between', 't07_shaped']].idxmax(axis=1) ==
            't07_between', 'GROUP'] = 1
        read_gz2.loc[read_gz2[[
            't07_completely', 't07_between', 't07_shaped']].idxmax(axis=1) ==
            't07_shaped', 'GROUP'] = 2
    elif task == 't09':
        # read data
        read_gz2 = pd.read_csv('Data\zoo2MainSpecz.csv', usecols=[
            'dr7objid',
            'gz2class',
            't09_bulge_shape_a25_rounded_weighted_fraction',
            't09_bulge_shape_a26_boxy_weighted_fraction',
            't09_bulge_shape_a27_no_bulge_weighted_fraction'])
        # change column names
        read_gz2 = read_gz2.rename(columns={
            'dr7objid': 'OBJID',
            't09_bulge_shape_a25_rounded_weighted_fraction': 't09_rounded',
            't09_bulge_shape_a26_boxy_weighted_fraction': 't09_boxy',
            't09_bulge_shape_a27_no_bulge_weighted_fraction': 't09_no_bulge'})
        # apply cut
        read_gz2 = read_gz2[((read_gz2['t09_rounded'] >= cut) |
                             (read_gz2['t09_boxy'] >= cut) |
                             (read_gz2['t09_no_bulge'] >= cut))]
        # define GROUP column
        read_gz2.loc[read_gz2[[
            't09_rounded', 't09_boxy', 't09_no_bulge']].idxmax(axis=1) ==
            't09_rounded', 'GROUP'] = 0
        read_gz2.loc[read_gz2[[
            't09_rounded', 't09_boxy', 't09_no_bulge']].idxmax(axis=1) ==
            't09_boxy', 'GROUP'] = 1
        read_gz2.loc[read_gz2[[
            't09_rounded', 't09_boxy', 't09_no_bulge']].idxmax(axis=1) ==
            't09_no_bulge', 'GROUP'] = 2
    elif task == 't10':
        read_gz2 = pd.read_csv('Data\zoo2MainSpecz.csv', usecols=[
            'dr7objid',
            'gz2class',
            't10_arms_winding_a28_tight_weighted_fraction',
            't10_arms_winding_a29_medium_weighted_fraction',
            't10_arms_winding_a30_loose_weighted_fraction'])
        read_gz2 = read_gz2.rename(columns={
            'dr7objid': 'OBJID',
            't10_arms_winding_a28_tight_weighted_fraction': 't10_tight',
            't10_arms_winding_a29_medium_weighted_fraction': 't10_medium',
            't10_arms_winding_a30_loose_weighted_fraction': 't10_loose'})
        # apply cut
        read_gz2 = read_gz2[((read_gz2['t10_tight'] >= cut) |
                             (read_gz2['t10_medium'] >= cut) |
                             (read_gz2['t10_loose'] >= cut))]
        # define GROUP column
        read_gz2.loc[read_gz2[[
            't10_tight', 't10_medium', 't10_loose']].idxmax(axis=1) ==
            't10_tight', 'GROUP'] = 0
        read_gz2.loc[read_gz2[[
            't10_tight', 't10_medium', 't10_loose']].idxmax(axis=1) ==
            't10_medium', 'GROUP'] = 1
        read_gz2.loc[read_gz2[[
            't10_tight', 't10_medium',
            't10_loose']].idxmax(axis=1) == 't10_loose', 'GROUP'] = 2
    elif task == 't11':
        # read data
        read_gz2 = pd.read_csv('Data\zoo2MainSpecz.csv', usecols=[
            'dr7objid',
            'gz2class',
            't11_arms_number_a31_1_weighted_fraction',
            't11_arms_number_a32_2_weighted_fraction',
            't11_arms_number_a33_3_weighted_fraction',
            't11_arms_number_a34_4_weighted_fraction',
            't11_arms_number_a36_more_than_4_weighted_fraction',
            't11_arms_number_a37_cant_tell_weighted_fraction'])
        # change column names
        read_gz2 = read_gz2.rename(columns={
            'dr7objid': 'OBJID',
            't11_arms_number_a31_1_weighted_fraction': 't11_1',
            't11_arms_number_a32_2_weighted_fraction': 't11_2',
            't11_arms_number_a33_3_weighted_fraction': 't11_3',
            't11_arms_number_a34_4_weighted_fraction': 't11_4',
            't11_arms_number_a36_more_than_4_weighted_fraction': 't11_more',
            't11_arms_number_a37_cant_tell_weighted_fraction': 't11_dunno'})
        # apply cut
        read_gz2 = read_gz2[
            ((read_gz2['t11_1'] >= cut) | (read_gz2['t11_2'] >= cut) |
             (read_gz2['t11_3'] >= cut) | (read_gz2['t11_4'] >= cut) |
             (read_gz2['t11_more'] >= cut) | (read_gz2['t11_dunno'] >= cut))]
        # define grouops column
        read_gz2.loc[read_gz2[[
            't11_1', 't11_2', 't11_3', 't11_4', 't11_more',
            't11_dunno']].idxmax(axis=1) == 't11_1', 'GROUP'] = 0
        read_gz2.loc[read_gz2[[
            't11_1', 't11_2', 't11_3', 't11_4', 't11_more',
            't11_dunno']].idxmax(axis=1) == 't11_2', 'GROUP'] = 1
        read_gz2.loc[read_gz2[[
            't11_1', 't11_2', 't11_3', 't11_4', 't11_more',
            't11_dunno']].idxmax(axis=1) == 't11_3', 'GROUP'] = 2
        read_gz2.loc[read_gz2[[
            't11_1', 't11_2', 't11_3', 't11_4', 't11_more',
            't11_dunno']].idxmax(axis=1) == 't11_4', 'GROUP'] = 3
        read_gz2.loc[read_gz2[[
            't11_1', 't11_2', 't11_3', 't11_4', 't11_more',
            't11_dunno']].idxmax(axis=1) == 't11_more', 'GROUP'] = 4
        read_gz2.loc[read_gz2[[
            't11_1', 't11_2', 't11_3', 't11_4', 't11_more',
            't11_dunno']].idxmax(axis=1) == 't11_dunno', 'GROUP'] = 5
    # merge with parameters data
    df_gz2 = pd.merge(read_gz2, input_parameters, on='OBJID', how='left')
    # drop na and outliers
    df_gz2 = df_gz2.dropna().drop(df_gz2[(df_gz2['dered_g-dered_r'] < 0)
                                         | (df_gz2['dered_g-dered_r'] >= 2)
                                         | (df_gz2['mCr4_i'] < 0)].index)

    return(df_gz2)


def prepare_data_gz1(cut, nrows):
    input_parameters = read_parameters_data()
    chunk = pd.read_csv('Data/GalaxyZoo1_DR_table2.csv', nrows=nrows,
                        usecols=['OBJID', 'P_EL', 'P_CW', 'P_ACW', 'P_EDGE',
                                 'P_DK', 'P_MG'])
    # calculate P_CS
    chunk['P_CS'] = chunk['P_CW'] + chunk['P_ACW'] + chunk['P_EDGE']
    # filter by cut and P_MG
    chunk = chunk[((chunk['P_EL'] >= cut) | (chunk['P_CS'] >= cut) |
                   (chunk['P_DK'] >= cut))]
    # select necessary columns
    chunk = chunk[['OBJID', 'P_EL', 'P_CS', 'P_DK']]
    # define GROUP column
    chunk.loc[chunk[['P_EL', 'P_CS', 'P_DK']].idxmax(axis=1) ==
              'P_EL', 'GROUP'] = 0
    chunk.loc[chunk[['P_EL', 'P_CS', 'P_DK']].idxmax(axis=1) ==
              'P_CS', 'GROUP'] = 1
    chunk.loc[chunk[['P_EL', 'P_CS', 'P_DK']].idxmax(axis=1) ==
              'P_DK', 'GROUP'] = 2
    # merge input parameters
    chunk = pd.merge(chunk, input_parameters, on='OBJID', how='left')
    # drop nas and outliers
    chunk = chunk.dropna().drop(chunk[(chunk['dered_g-dered_r'] < 0)
                                      | (chunk['dered_g-dered_r'] >= 2)
                                      | (chunk['mCr4_i'] < 0)].index)
    return(chunk)


def data_wrapper(data, split=0.4, nclass=3, nvar=11):
    # training and test split
    variables = [
        'dered_g-dered_r', 'deVAB_i', 'concentration',
        'dered_r-dered_i', 'expAB_i']
    train, test = train_test_split(data, test_size=split)
    train = (train[variables].as_matrix(),
             train[['GROUP']].astype(int).as_matrix())
    test = (test[variables].as_matrix(),
            test[['GROUP']].astype(int).as_matrix())
    # format data
    training_inputs = [np.reshape(x, (nvar, 1)) for x in train[0]]
    training_results = [vectorize(nclass, y) for y in train[1]]
    training_data = zip(training_inputs, training_results)
    test_inputs = [np.reshape(x, (nvar, 1)) for x in test[0]]
    test_data = zip(test_inputs, test[1])
    return (training_data, test_data)


# def data_wrapper(data, split=0.4, nclass=3, nvar=11):
#     # training and test split
#     train, test = train_test_split(data, test_size=split)
#     train = (train[[
#         'dered_g-dered_r', 'dered_r-dered_i', 'deVAB_i', 'expAB_i', 'lnLexp_i',
#         'lnLdeV_i', 'lnLstar_i', 'mRrCc_i', 'mCr4_i', 'texture_i',
#         'concentration']].as_matrix(),
#         train[['GROUP']].astype(int).as_matrix())
#     test = (test[[
#         'dered_g-dered_r', 'dered_r-dered_i', 'deVAB_i', 'expAB_i', 'lnLexp_i',
#         'lnLdeV_i', 'lnLstar_i', 'mRrCc_i', 'mCr4_i', 'texture_i',
#         'concentration']].as_matrix(), test[['GROUP']].astype(int).as_matrix())
#     # format data
#     training_inputs = [np.reshape(x, (nvar, 1)) for x in train[0]]
#     training_results = [vectorize(nclass, y) for y in train[1]]
#     training_data = zip(training_inputs, training_results)
#     test_inputs = [np.reshape(x, (nvar, 1)) for x in test[0]]
#     test_data = zip(test_inputs, test[1])
#     return (training_data, test_data)
