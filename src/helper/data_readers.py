"""Every readingData() version in this repo, gathered in ONE file.

These are DIFFERENT functions (they read different year CSVs in different ways),
so each has its own name. The old `readingData` in each utils file is now a
redirect import of the matching function here, so existing imports keep working.
Which version is "right" is undecided (ponder-later) -- nothing here changes behavior.

  readingData_utils         utils.py                    5 years, by position in glob
  readingData_pnn           utils_pnn.py                10 years, by position in glob
  readingData_coolturtles   utils_from_coolturtles.py   10 years, by 'YYYY_YYYY' in filename
  readingData_esb_rotation  utils_mse_crps.py           named ESB years (ESB_ROTATION_YEARS)
"""
from .esb_years import ESB_ROTATION_YEARS  # used by readingData_esb_rotation

# TODO(#15): add readingData_laguna_madre() here once #15 is merged.


def readingData_utils(path_to_data):
    '''readingData() reads the data for each year'''

    import glob
    import pandas as pd

    print('inside readingData in utils. path_to_data:',path_to_data)
    csvs = glob.glob(f"{path_to_data}/*csv")
    
    # Reading years with extra rows (5 days back and 5 days after)
    #print(csvs)
    data_year1 = pd.read_csv(csvs[11])
    data_year2 = pd.read_csv(csvs[1])
    data_year3 = pd.read_csv(csvs[2])
    data_year4 = pd.read_csv(csvs[3])
    data_year5 = pd.read_csv(csvs[4])
    # data_year6 = pd.read_csv(csvs[5])
    # data_year7 = pd.read_csv(csvs[6])
    # data_year8 = pd.read_csv(csvs[7])
    # data_year9 = pd.read_csv(csvs[8])
    # data_year10 = pd.read_csv(csvs[10])
    
    return data_year1, data_year2, data_year3, data_year4, data_year5

def readingData_pnn(path_to_data):
    '''readingData() reads the data for each year'''

    import glob
    import pandas as pd

    print('inside readingData in utils. path_to_data:',path_to_data)
    csvs = glob.glob(f"{path_to_data}/*csv")
    
    # Reading years with extra rows (5 days back and 5 days after)
    #print(csvs)
    data_year1 = pd.read_csv(csvs[11])
    data_year2 = pd.read_csv(csvs[1])
    data_year3 = pd.read_csv(csvs[2])
    data_year4 = pd.read_csv(csvs[3])
    data_year5 = pd.read_csv(csvs[4])
    data_year6 = pd.read_csv(csvs[5])
    data_year7 = pd.read_csv(csvs[6])
    data_year8 = pd.read_csv(csvs[7])
    data_year9 = pd.read_csv(csvs[8])
    data_year10 = pd.read_csv(csvs[10])
    
    return data_year1, data_year2, data_year3, data_year4, data_year5, data_year6, data_year7, data_year8, data_year9, data_year10

def readingData_coolturtles(path_to_data):
    '''readingData() reads the data for each year'''

    import glob
    import pandas as pd
    import os
    import re

    csvs = glob.glob(f"{path_to_data}/*.csv")
    # print(csvs)
    print(f"Found {len(csvs)} CSV files in the directory.")

    pattern = re.compile(r'(\d{4})_(\d{4})')

    year_dict = {}

    for csv in csvs:
        match = pattern.search(csv)
        if match:
            year_range = f"{match.group(1)}_{match.group(2)}"  # e.g., "2022_2023"
            year_dict[year_range] = pd.read_csv(csv)  # Read CSV into DataFrame


    # csvs = glob.glob(f"{path_to_data}/*csv")
    data_year1 = year_dict['2022_2023']  # Read the first CSV into a DataFrame
    data_year2 = year_dict['2012_2013']  # Read the second CSV into a DataFrame
    data_year3 = year_dict['2013_2014']  # Read the third CSV into a DataFrame
    data_year4 = year_dict['2014_2015']  # Read the fourth CSV into a DataFrame
    data_year5 = year_dict['2015_2016']  # Read the fifth CSV into a DataFrame
    data_year6 = year_dict['2016_2017']  # Read the sixth CSV into a DataFrame
    data_year7 = year_dict['2017_2018']  # Read the seventh CSV into a DataFrame
    data_year8 = year_dict['2018_2019']  # Read the eighth CSV into a DataFrame
    data_year9 = year_dict['2019_2020']  # Read the ninth CSV into a DataFrame
    data_year10 = year_dict['2021_2022']  # Read the tenth CSV into a DataFramw
    
    return data_year1, data_year2, data_year3, data_year4, data_year5, data_year6, data_year7, data_year8, data_year9, data_year10

def readingData_esb_rotation(path_to_data, rotation_years=None):
    '''
    -------------------------------------------------------------------------
                                def readingData
    input:
            path_to_data - folder containing the year CSVs
            rotation_years - ordered year names (no .csv) for year2..year5.
                             Defaults to ESB_ROTATION_YEARS.
    purpose:
            load the NAMED years, in the order given. Files in the folder
            that are not named get ignored; a named file that is missing
            is an error.
    output:
            data_list - one dataframe per name in rotation_years
    -------------------------------------------------------------------------
    '''

    import os
    import glob
    import pandas as pd

    if rotation_years is None:
        rotation_years = ESB_ROTATION_YEARS

    # Key each CSV by its filename stem: a year is found by NAME, not position.
    csv_files = sorted(glob.glob(os.path.join(path_to_data, "*.csv")))
    
    # dict comprehension that builds a lookup table mapping year name → full path
    by_name = {os.path.splitext(os.path.basename(f))[0]: f for f in csv_files}

    missing = set(rotation_years) - set(by_name)
    if missing:
        raise FileNotFoundError(
            f"rotation year(s) {sorted(missing)} not found in {path_to_data}. "
            f"Present: {sorted(by_name)}"
        )

    # Ignoring unnamed files is what lets esb_2025_2026 sit in this folder
    # harmlessly -- but say so, since ignored must never mean silently dropped.
    ignored = sorted(set(by_name) - set(rotation_years))
    print(f"Found {len(csv_files)} CSVs. Rotation: {list(rotation_years)}")
    if ignored:
        print("Ignored (not in rotation):", ignored)

    return [pd.read_csv(by_name[name]) for name in rotation_years]
