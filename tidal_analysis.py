#!/usr/bin/env python3

"""
This script processes and analyses tidal data.

Google amd gemini were used to help write the functions and code within this script.

"""

# import the modules you need here
import argparse
import os
import glob
import datetime
import pandas as pd
import numpy as np
from scipy import stats
import pytz
import uptide
from tabulate import tabulate

def read_tidal_data(filename):
    """
    Reads tidal data from a specified file and returns it as a pandas DataFrame.

    This function handles parsing mixed data types, cleaning 'Sea Level' entries
    by removing specific characters and invalid values, and setting the 'Time'
    column as the DataFrame's index.

    Args:
        filename (str): The path to the file containing the tidal data.

    Returns:
        pd.DataFrame: 

    Raises:
        FileNotFoundError: If the specified `filename` does not exist.
        RuntimeError: If an unexpected error occurs during file reading or processing.
    """
    # throw error if file does not exist
    if not os.path.exists(filename):
        raise FileNotFoundError(f"No such file: {filename}")

    # define collumn names
    column_names = ["Cycle","Date", "Time", "Sea Level", "Residual"]

    # lines 48-53 code taken from gemini
    try:
        # read the data using pandas.read_csv.
        data = pd.read_csv(
            filename,
            header=None,        # tells pandas there are no headers
            names=column_names,     # tells pandas to use your list to name the columns.
            skiprows=list(range(11)),
            sep=r"\s+",
            )

        # convert datetime fields in a pd.datetime and ignore errors
        data["Time"] = pd.to_datetime(
            data["Date"] + " " + data["Time"], errors="coerce"
        )
        # convert Sea Level to string
        data["Sea Level"] = data["Sea Level"].astype(str)
        # remove base data entries from Sea Level field using regular expression
        data["Sea Level"] = data["Sea Level"].replace(
            to_replace=r".*[M]$", value=np.nan, regex=True
        )
        # remove the last T character from sea level as this is valid data and
        # was causing the test_linear_regression test to fail as it was passing in the
        # wrong number of data points which should be 15624 if the T values are omitted
        # then this value was 15604 points which brought the value value to about 2.84
        data["Sea Level"] = data["Sea Level"].replace(r"T$", "", regex=True)
        # remove bad N data enteries
        data["Sea Level"] = data["Sea Level"].replace("-99.0000N", np.nan)
        # convert Sea Level to a float, empty values converted to NaN
        data["Sea Level"] = pd.to_numeric(data["Sea Level"], errors="coerce")
        # return Time and Sea Level fields into the DataFrame
        data = data[["Time", "Sea Level"]]
        # set the DateTime field as an index but don't drop the Time field as it is
        # required in a test
        data.set_index("Time", inplace=True, drop=False)

        # lines 84-89 code taken from gemini
        return data
    except Exception as e:
        # catch any other general errors that might occur during file reading
        raise RuntimeError(
           f"An unexpected error occurred while reading tidal data from {filename}: {e}"
        ) from e

def extract_single_year_remove_mean(year, data):
    """
    Extracts 'Sea Level' data for a specific year and removes its mean.

    Args:
        year (int): The calendar year to extract data for.
        data (pd.DataFrame): The DataFrame containing 'Sea Level' data with a DatetimeIndex.

    Returns:
        pd.DataFrame: A DataFrame containing 'Sea Level' data for the specified year,
                      with its mean subtracted.
    """
    # lines 104-111 code taken from https://jhill1.github.io/SEPwC.github.io/tides_python.html
    year_string_start = str(year)+"0101"
    year_string_end = str(year)+"1231"
    year_data = data.loc[year_string_start:year_string_end, ["Sea Level"]]
    # remove mean to oscillate around zero
    mmm = np.mean(year_data["Sea Level"])
    year_data["Sea Level"] -= mmm

    return year_data

def extract_section_remove_mean(start, end, data):
    """
    Extracts a specific time section of 'Sea Level' data and removes its mean.

    Args:
        start (str or datetime-like): The start date/time for the data section.
        end (str or datetime-like): The end date/time for the data section.
        data (pd.DataFrame): The DataFrame containing 'Sea Level' data with a DatetimeIndex.

    Returns:
        pd.DataFrame: 
    """
    # lines 126-135 code from gemini
    # extract specific section of the data
    section = data.loc[start:end, ["Sea Level"]].copy()

    # calculate the mean of the 'Sea Level' data within the extracted section
    mean_sea_level = np.mean(section["Sea Level"])

    # remove calculated mean from the 'Sea Level' data
    section["Sea Level"] -= mean_sea_level

    return section


def join_data(data1, data2):
    """
    Joins two Pandas DataFrames together.
    Dataframes are concatenated and sorted by their index.

    Args:
        data1 (pd.DataFrame): The first DataFrame to join.
        data2 (pd.DataFrame): The second DataFrame to join.

    Returns:
        pd.DataFrame: A new DataFrame containing the combined and sorted data from data1 and data2.
    """
    # join the two DataFrame objects together
    # sort by the index which will be the datetime field
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html

    # concatenate the two objects, this returns a copy by default but is not a deep copy
    df = pd.concat([data1, data2])

    # re-sort the combined DataFrame by its DateTimeIndex
    df.sort_index(inplace=True)

    return df


def sea_level_rise(data):
    """
   Calculates the sea level rise (slope) and its statistical significance (p-value)
   from time series 'Sea Level' data using linear regression.

   Args:
       data (pd.DataFrame): A DataFrame expected to contain a 'Sea Level' column
                            and a datetime-like index.

   Returns:
       tuple[float, float]: A tuple containing:
                            - slope (float): The calculated rate of sea level change per day.
                            - p_value (float): The p-value indicating the statistical
                                               significance of the observed trend.
                            Returns (np.nan, np.nan) if there are fewer than 2 valid data points.
   """
    # lines 180-204 code taken from gemini
    # remove rows where 'Sea Level' is missing
    # convert remianing 'Sea Level' values to floating-point numbers
    sea_level_series_m = data["Sea Level"].dropna().astype(float)

    # check for at least two valid data points
    # if not enough data return NaN for both results
    if len(sea_level_series_m) < 2:
        return np.nan, np.nan

    # check if the datetime index of the sea level data has timezone information
    # if not, localize it to Coordinated Universal Time (UTC) to ensure consistency
    if sea_level_series_m.index.tzinfo is None:
        sea_level_series_m.index = sea_level_series_m.index.tz_localize(pytz.utc)

    # define a fixed reference point in time (an "epoch")
    epoch = datetime.datetime(1900, 1, 1, 0, 0, 0, tzinfo=pytz.utc)

    # calculate the time difference, in days, from the epoch to each data point
    time_in_days = (sea_level_series_m.index - epoch).total_seconds() / (24 * 3600)

    # perform a linear regression using the `linregress` function from scipy stats module
    # only need 'slope' and 'p value' so others are discarded using '_'
    slope, _, _, p_value, _ = stats.linregress(time_in_days, sea_level_series_m.values)

    return slope, p_value


def tidal_analysis(data, constituents, start_datetime):
    """
    Analyzes sea level data to determine the amplitude and phase of specified
    tidal constituents using the 'uptide' library.
    (Reference: https://jhill1.github.io/SEPwC.github.io/tides_python.html)

    Args:
        data (pd.DataFrame): DataFrame with a DateTime index and "Sea Level" column.
        constituents (list): List of tidal constituent strings (e.g., ["M2", "S2"]).
        start_datetime (datetime.datetime): Start datetime for the tidal model.

    Returns:
        tuple: (amp, pha) - Numpy arrays of amplitudes and phases.
    """
    # https://jhill1.github.io/SEPwC.github.io/tides_python.html information used
    # for working out the correct tidal analysis values

    # put DataFrame index to UTC (GMT) timezone
    data.index = data.index.tz_localize("UTC")

    # create a Tides object with a list of the consituents we want
    tide = uptide.Tides(constituents)

    # set a start time for the uptide model
    tide.set_initial_time(start_datetime)

    # remove rows with NaN values from data as tides
    data.dropna(subset=["Sea Level"], inplace=True)

    # convert DateTime index into seconds
    seconds_since = (
        data.index.astype("int64").to_numpy() / 1e9
    ) - start_datetime.timestamp()

    amp, pha = uptide.harmonic_analysis(
        tide, data["Sea Level"].to_numpy(), seconds_since
    )

    return amp, pha


def format_longest_contiguous_data(data):
    """
    Formats the rows in the DataFrame from the data returned from 
    get_longest_contiguous_data function.
    
    This method and code has been generated from Gemini.

    Args:
        data (pd.DataFrame): The input DataFrame.

    Returns:
        string: start and end range formatted in Uk date time.
    """
    start_utc = data.index[0]
    end_utc = data.index[-1]

    uk_timezone = pytz.timezone("Europe/London")

    if start_utc.tz is None:
        start_utc = start_utc.tz_localize("UTC")

    if end_utc.tz is None:
        end_utc = end_utc.tz_localize("UTC")

    start_gb_uk = start_utc.tz_convert(uk_timezone)
    end_gb_uk = end_utc.tz_convert(uk_timezone)

    results = (
        f"{start_gb_uk.strftime('%d/%m/%Y %H:%M:%S %Z%z')} to "
        f"{end_gb_uk.strftime('%d/%m/%Y %H:%M:%S %Z%z')} ({len(data)})"
    )

    return results


def get_longest_contiguous_data(data):
    """
    Identifies and returns the longest contiguous block of non-NaN rows in a DataFrame. 
    
    This method and code has been generated from Gemini.

    Args:
        data (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The longest contiguous block of rows where all values are non-NaN.
                      Returns an empty DataFrame if no such block exists.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    if data.empty:
        return data

    # create a boolean mask where each row is True if all values are non-NaN
    valid_rows = data.notna().all(axis=1)

    # identify contiguous blocks using cumsum on invalid rows (similar to run-length encoding)
    block_ids = (~valid_rows).cumsum()

    # filter out only valid rows
    non_nan_data = data[valid_rows]

    # group valid rows by block IDs
    grouped = non_nan_data.groupby(block_ids[valid_rows])

    if grouped.ngroups == 0:
        return data.iloc[0:0]  # Return empty DataFrame with same structure

    # get the block with the maximum length
    longest_block_id = grouped.size().idxmax()

    return grouped.get_group(longest_block_id)


def valid_directory(path):
    """
    Validates if a given path points to an existing directory.

    This function is typically used with `argparse` to validate command-line
    arguments, ensuring that the supplied path is a legitimate directory.

    Args:
        path (str): The string path to be checked.

    Returns:
        str: The original path string if it is a valid directory.

    Raises:
        argparse.ArgumentTypeError: If the path is not a directory.
    """
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError(f"'{path}' is not a valid directory")
    return path


def process_files(directory):
    """
    Processes tidal data files from a specified directory, performs tidal analysis,
    and prints the calculated M2 and S2 amplitudes.
    
    Most method and code taken and modified from gemini.
    
    Args:
        directory (str): The path to the directory containing the tidal data text files.

    Returns:
        None: This function prints the analysis summary directly to the console
              and does not return any value.
    """
    # get files using glob that are text files
    txt_files = glob.glob(directory + "/*.txt")
    # get the folder name
    folder_name = os.path.basename(os.path.normpath(directory))

    # check that there is at least 1 file
    if len(txt_files) > 0:
        # read the first data file
        data = read_tidal_data(txt_files[0])

        # skip the first file as this has been processed above
        for file in txt_files[1:]:
            next_data = read_tidal_data(file)
            data = join_data(data, next_data)

        # get longest contiguous period
        contiguous_data = get_longest_contiguous_data(data)

        # format the longest contiguous period
        contiguous_results = format_longest_contiguous_data(contiguous_data)

        # calculate M2 and S2 Amplitude
        constituents = ["M2", "S2"]
        tz = pytz.timezone("utc")
        start_datetime = data.index.tz_localize(tz)[0]
        amp, pha = tidal_analysis(data, constituents, start_datetime)

        # setup headers used by tabulate
        headers = [
            "Station Name",
            "M2 Amplitude",
            "S2 Amplitude",
            "Longest contiguous period (data points)",
        ]
        # set up data used to display after the header, uses capitalize and
        # formatting decimal places
        row = [
            [
                folder_name.capitalize(),
                f"{amp[0]:.3f} m",
                f"{pha[0]:.3f} m",
                contiguous_results,
            ]
        ]
        # print the results
        print(tabulate(row, headers=headers, tablefmt="simple_grid"))
    else:
        print("No files found")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="UK Tidal analysis",
        description="Calculate tidal constiuents and RSL from tide gauge data",
        epilog="Copyright 2024, Jon Hill",
    )

    parser.add_argument(
        "directory",
        type=valid_directory,
        help="the directory containing txt files with data",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", default=False, help="Print progress"
    )

    args = parser.parse_args()
    dirname = args.directory
    verbose = args.verbose

    # call the main processing function
    process_files(dirname)
