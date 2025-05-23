#!/usr/bin/env python3

# import the modules you need here
import argparse
import os 
import datetime
import pandas as pd 
import numpy as np
import pytz
import uptide 

def read_tidal_data(filename):
    # throw error if file does not exist 
    if not os.path.exists(filename):
        raise FileNotFoundError(f"No such file: {filename}")
    
    # define collumn names  
    column_names = ["Cycle","Date", "Time", "Sea Level", "Residual"]

    # lines 20-25 code taken from gemini
    try:
        # read the data using pandas.read_csv.
        data = pd.read_csv(
            filename,
            header=None,              # tells pandas there are no headers 
            names=column_names,       # tells pandas to use your list to name the columns.
            skiprows=list(range(11)),
            sep=r"\s+",
            )
        na_values=['M', 'N', 'T'] # this converts 'M', 'N', 'T' to NaN.
        
        # code from gemini 
        print(data.head()) 

        # convert datetime fields in a pd.datetime and ignore errors
        data["Time"] = pd.to_datetime(
            data["Date"] + " " + data["Time"], errors="coerce"
        )
        # convert Sea Level to string 
        data["Sea Level"] = data["Sea Level"].astype(str)
        # remove base data entries from Sea Level field using regular expression
        data["Sea Level"] = data["Sea Level"].replace(
            to_replace=r".*[MT]$", value=np.nan, regex=True
        )
        # remove bad N data enteries
        data["Sea Level"] = data["Sea Level"].replace("-99.0000N", np.nan)
        # convert Sea Level to a float
        data["Sea Level"] = pd.to_numeric(data["Sea Level"], errors="coerce")
        # return Time and Sea Level fields into the DataFrame
        data = data[["Time", "Sea Level"]]
        # set the DateTime field as an index but don't drop the Time field as it is 
        # required in a test.
        data.set_index("Time", inplace=True, drop=False)     

        # lines 55-58 code taken from gemini 
        return data                                                                
    except Exception as e:          
        # catch any other general errors that might occur during file reading
        raise RuntimeError(f"An unexpected error occurred while reading tidal data from {filename}: {e}") from e
    

def extract_single_year_remove_mean(year, data):
        
    # lines 64-71 code taken from https://jhill1.github.io/SEPwC.github.io/tides_python.html
    year_string_start = str(year)+"0101"
    year_string_end = str(year)+"1231"
    year_data = data.loc[year_string_start:year_string_end, ["Sea Level"]]
    # remove mean to oscillate around zero
    mmm = np.mean(year_data["Sea Level"])
    year_data["Sea Level"] -= mmm
     
    return year_data


def extract_section_remove_mean(start, end, data):
    # lines 76-85 from gemini 
    # extract specific section of the data
    section = data.loc[start:end, ["Sea Level"]].copy()

    # calculate the mean of the 'Sea Level' data within the extracted section 
    mean_sea_level = np.mean(section["Sea Level"])

    # remove calculated mean from the 'Sea Level' data 
    section["Sea Level"] -= mean_sea_level

    return section


def join_data(data1, data2):
    
    # join the two DataFrame objects together 
    # sort by the index which will be the datetime field
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html

    # concatenate the two objects, this returns a copy by default but is not a deep copy
        df = pd.concat([data1, data2])

    # re-sort the combined DataFrame by its DateTimeIndex
        df.sort_index(inplace=True)

        return df


def sea_level_rise(data):

                                                     
    return 


def tidal_analysis(data, constituents, start_datetime):
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
        
        return 
    
    
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
    


