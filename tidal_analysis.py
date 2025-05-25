#!/usr/bin/env python3

"""
This script processes and analyses tidal data using python functions
"""

# import the modules you need here
import argparse
import os
import datetime
import pandas as pd
import numpy as np
from scipy import stats
import pytz
import uptide

def read_tidal_data(filename):
    """
    Reads tidal data from a specified file and returns it as a pandas DataFrame

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

    # lines 43-48 code taken from gemini
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

        # lines 79-84 code taken from gemini
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
    # lines 99-106 code taken from https://jhill1.github.io/SEPwC.github.io/tides_python.html
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
    # lines 121-130 code from gemini
    # extract specific section of the data
    section = data.loc[start:end, ["Sea Level"]].copy()

    # calculate the mean of the 'Sea Level' data within the extracted section
    mean_sea_level = np.mean(section["Sea Level"])

    # remove calculated mean from the 'Sea Level' data
    section["Sea Level"] -= mean_sea_level

    return section


def join_data(data1, data2):
    """
    Joins two Pandas DataFrames together
    Dataframes are concatenated and then sorted by their index

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
    # lines 175-196 code taken from gemini
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


def get_longest_contiguous_data(data):
    """
    Identifies and returns the longest contiguous block of non NaN data.

    This function processes a pandas Series or DataFrame to find the longest
    sequence of consecutive non-null (non-NaN) values
    Args:
        data (pd.Series or pd.DataFrame): The input data, which can be a pandas Series
                                         or DataFrame. The function will operate on
                                         the non-null status of the data.

    Returns:
        pd.Series or pd.DataFrame: The longest contiguous block of non-NaN data
                               from the input. Returns an empty Series/DataFrame
                               if no non-NaN data is found.
    """
    # lines 260-275 code from gemini
    # handle empty input data
    if data.empty:
        return data

    # create a boolean mask: true for non nan and false for nan
    is_not_nan = data.notna()
    # generate block ID's
    block_ids = (~is_not_nan).cumsum()
    # filter out NaN values from original data
    non_nan_data = data[is_not_nan]
    # group the non NaN data by the generated block ID's
    grouped_blocks = non_nan_data.groupby(block_ids[is_not_nan])
    # find the ID of the block with the maximum length
    longest_block_id = grouped_blocks.size().idxmax()

    return grouped_blocks.get_group(longest_block_id)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                     prog="UK Tidal analysis",
                     description="Calculate tidal constiuents and RSL from tide gauge data",
                     epilog="Copyright 2024, Jon Hill"
                     )

    parser.add_argument("directory",
                    help="the directory containing txt files with data")
    parser.add_argument('-v', '--verbose',
                    action='store_true',
                    default=False,
                    help="Print progress")

    args = parser.parse_args()
    dirname = args.directory
    verbose = args.verbose
