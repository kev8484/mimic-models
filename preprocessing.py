import numpy as np
import pandas as pd


def identify_copd_admits(icd_path):
    """ Identify admissions that have a COPD diagnosis in any dx position.
    """
    # strict copd coding
    strict_icd9 = [
        "49120",
        "49121",
        "49122",
        "49320",
        "49321",
        "49322",
        "496",
    ]

    # regular copd coding
    reg_icd9 = [
        "4911",
        "4920",
        "4928",
    ]

    print("Loading dx codes...")
    df = pd.read_csv(icd_path)
    copd_hadmids = df[df.ICD9_CODE.isin(
        strict_icd9 + reg_icd9)].HADM_ID.unique()

    return df, copd_hadmids


def identify_30d_readmits(pt_path, admit_path, icd_df, copd_ids):
    """ Identify readmissions and flag 30 day readmits
    """

    patients = pd.read_csv(pt_path, parse_dates=['DOB', 'DOD', 'DOD_HOSP'])

    admission_cols = [
        'HADM_ID',
        'ADMISSION_TYPE',
        'ADMITTIME',
        'DISCHTIME',
        'DEATHTIME',
        'EDREGTIME',
        'EDOUTTIME',
        'HOSPITAL_EXPIRE_FLAG',
        'HAS_CHARTEVENTS_DATA',
    ]
    print("Loading admission events...")
    admits = pd.read_csv(admit_path, parse_dates=[
                         'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'EDREGTIME', 'EDOUTTIME', ], usecols=admission_cols)

    # concat primary dx onto admissions
    admits = admits.merge(icd_df, on=['HADM_ID']).drop_duplicates(
        subset=["HADM_ID"])

    # get rid of spurrious admissions and ignore newborns
    admits = admits[(admits['DISCHTIME'] > admits['ADMITTIME'])
                    & (admits.ADMISSION_TYPE != "NEWBORN")]

    # add age information
    admits = admits.merge(
        patients[['SUBJECT_ID', 'DOB']], on='SUBJECT_ID', how='left')
    admits['age'] = admits.apply(lambda x: (
        x['ADMITTIME'].date() - x['DOB'].date()).days // 365.242, axis=1)

    # tag copd admissions
    admits['copd'] = admits.HADM_ID.isin(copd_ids)

    # get the type and time of the next admission
    admits.sort_values(by=['SUBJECT_ID', 'ADMITTIME'], inplace=True)
    admits['next_admit_time'] = admits.groupby(
        'SUBJECT_ID').ADMITTIME.shift(-1)
    admits['next_admit_type'] = admits.groupby(
        'SUBJECT_ID').ADMISSION_TYPE.shift(-1)
    # if the next admission is elective, nullify and back fill
    admits.loc[admits.next_admit_type ==
               "ELECTIVE", 'next_admit_time'] = pd.NaT
    admits.loc[admits.next_admit_type ==
               "ELECTIVE", 'next_admit_type'] = np.nan
    admits[['next_admit_time', 'next_admit_type']] = admits.groupby(
        ['SUBJECT_ID'])[['next_admit_time', 'next_admit_type']].fillna(method='bfill')

    # compute readmission stats
    admits['readmit_time'] = admits.groupby('SUBJECT_ID').apply(
        lambda x: x['next_admit_time'] - x['DISCHTIME']).reset_index(level=0, drop=True)
    admits['30d_readmit'] = (
        admits['readmit_time'].dt.total_seconds() < 30 * 24 * 3600).astype(int)

    return admits


def retrieve_discharge_notes(dc_path, admits_df):
    chunk_reader = pd.read_csv(
        dc_path,
        chunksize=100000,
        usecols=['SUBJECT_ID', 'HADM_ID', 'CHARTDATE',
                 'CATEGORY', 'DESCRIPTION', 'TEXT', ]
    )
    chunk_li = []
    for chunk in chunk_reader:
        chunk_li.append(chunk[(chunk['CATEGORY'] == 'Discharge summary')])

    notes = pd.concat(chunk_li, ignore_index=True)
    # keep only one discharge summary per admission
    notes = notes.sort_values(
        by=['SUBJECT_ID', 'HADM_ID', 'CHARTDATE']).groupby(['HADM_ID']).nth(-1)
    cols = [
        'HADM_ID',
        'SUBJECT_ID',
        'age',
        'copd',
        'HOSPITAL_EXPIRE_FLAG',
        'ADMISSION_TYPE',
        'ADMITTIME',
        'DISCHTIME',
        'DEATHTIME',
        'next_admit_time',
        'next_admit_type',
        '30d_readmit',
    ]
    notes = notes.merge(admits_df[cols], on=[
                        'SUBJECT_ID', 'HADM_ID'], how='inner')

    return notes
