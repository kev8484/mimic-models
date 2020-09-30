import argparse
from pathlib import Path

import numpy as np
import pandas as pd


S3_PATH_NOTES = "s3://mimic-iii-physionet/NOTEEVENTS.csv.gz"
S3_PATH_DIAGNOSES = "s3://mimic-iii-physionet/DIAGNOSES_ICD.csv.gz"
S3_PATH_PATIENTS = "s3://mimic-iii-physionet/PATIENTS.csv.gz"
S3_PATH_ADMISSIONS = "s3://mimic-iii-physionet/ADMISSIONS.csv.gz"


def identify_copd_admits(mimic_path):
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
    if mimic_path is None:
        df = pd.read_csv(S3_PATH_DIAGNOSES)
    else:
        df = pd.read_csv(mimic_path / Path("DIAGNOSES_ICD.csv"))
    copd_hadmids = df[df.ICD9_CODE.isin(
        strict_icd9 + reg_icd9)].HADM_ID.unique()

    return df, copd_hadmids


def identify_30d_readmits(mimic_path, icd_df, copd_ids):
    """ Identify readmissions and flag 30 day readmits
    """
    if mimic_path is None:
        patients = pd.read_csv(S3_PATH_PATIENTS, parse_dates=[
                               'DOB', 'DOD', 'DOD_HOSP'])
    else:
        patients = pd.read_csv(mimic_path / Path("PATIENTS.csv"),
                               parse_dates=['DOB', 'DOD', 'DOD_HOSP'])

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
    if mimic_path is None:
        admits = pd.read_csv(
            S3_PATH_ADMISSIONS,
            parse_dates=['ADMITTIME', 'DISCHTIME',
                         'DEATHTIME', 'EDREGTIME', 'EDOUTTIME'],
            usecols=admission_cols
        )
    else:
        admits = pd.read_csv(
            mimic_path / Path("ADMISSIONS.csv"),
            parse_dates=['ADMITTIME', 'DISCHTIME',
                         'DEATHTIME', 'EDREGTIME', 'EDOUTTIME'],
            usecols=admission_cols
        )

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


def retrieve_discharge_notes(mimic_path, admits_df):
    """ Retrieve discharge summaries and attach 30d readmit outcomes
    """
    if mimic_path is None:
        chunk_reader = pd.read_csv(
            S3_PATH_NOTES,
            chunksize=100000,
            usecols=['SUBJECT_ID', 'HADM_ID', 'CHARTDATE',
                     'CATEGORY', 'DESCRIPTION', 'TEXT', ]
        )
    else:
        chunk_reader = pd.read_csv(
            mimic_path / Path("NOTEEVENTS.csv"),
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


def preprocess_notes(notes, char_limit=2500):

    text_data = notes[notes['HOSPITAL_EXPIRE_FLAG'] == 0]['TEXT']\
        .fillna(' ')\
        .str.replace('\n', ' ')\
        .str.replace('\r', ' ')\
        .str.replace(r'\[\*\*(\d{4}-\d{1,2}-\d{1,2})\*\*\]',
                     '<DATE>', regex=True)\
        .str.replace(r'\[\*\*([\da-zA-Z() \(\)]*?(?:Name|name)[\da-zA-Z \(\)]*?)\*\*\]',
                     '<NAME>', regex=True)\
        .str.replace(r'\[\*\*([\d-]*?)\*\*\]',
                     '<NUMBER>', regex=True)\
        .str.replace(r'\[\*\*(Hospital.*?)\*\*\]',
                     '<HOSPITAL>', regex=True)\
        .str.replace(r'\[\*\*(.*?)\*\*\]',
                     '<UNK>', regex=True)\
        .str[:char_limit]

    labels = notes[notes['HOSPITAL_EXPIRE_FLAG'] == 0]['30d_readmit']

    return text_data, labels


def save_data(path, text_data, labels, char_limit):
    text_data.to_csv(
        path / Path(f"mimic_discharge_summaries_{char_limit}chars.csv"),
        index=False
    )

    labels.to_csv(
        path / Path("mimic_30d_readmit_labels.csv"),
        index=False
    )


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-dir", type=Path, default=None,
                        help="The local directory containing MIMIC data files, otherwise download from AWS")
    parser.add_argument("--char-limit", type=int, default=2500,
                        help="Number of characters for truncating output text data")
    parser.add_argument("--out-dir", type=Path, default=Path.cwd(),
                        help="Output directory to write processed data and label files to")
    args = parser.parse_args()
    return args


def main(args):
    args = parse_args(args)
    mimic_files = [
        "DIAGNOSES_ICD.csv",
        "PATIENTS.csv",
        "ADMISSIONS.csv",
        "NOTEEVENTS.csv",
    ]
    if args.mimic_dir is not None:
        found_files = [f for f in mimic_files if (
            args.mimic_dir / Path(f)).exists()]

        if len(found_files) != len(mimic_files):
            raise RuntimeError(
                "Unable to locate MIMIC data files in specified directory.")
    print("Loading diagnosis codes...")
    diag_df, copd_ids = identify_copd_admits(args.mimic_dir)
    print("Loading admissions...")
    admits = identify_30d_readmits(args.mimic_dir, diag_df, copd_ids)
    print("Loading discharge notes...")
    notes = retrieve_discharge_notes(args.mimic_dir, admits)
    print("Preprocessing text data...")
    text_data, labels = preprocess_notes(notes, args.char_limit)
    print("Saving file...")
    save_data(args.out_dir, text_data, labels, args.char_limit)

    print(
        f"MIMIC data prep complete, {notes.shape[0]} discharge summaries processed.")


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
