from cmath import log
import pandas as pd
import numpy as np
import sys
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

sys.path.append(os.path.abspath(os.path.join("./scripts/")))
from logger import logger

class CleaningPipeline():

    def convert_to_string(self, df, columns):
        try:
            for col in columns:
                df[col] = df[col].astype("string")
            logger.info(f"{columns} converted to string.")
        except Exception as e:
            logger.error(e)
    
    def convert_to_datetime(self, df, columns):
        try:
            for col in columns:
                df[col] = pd.to_datetime(df[col])
            logger.info(f"{columns} converted to datetime.")
        except Exception as e:
            logger.error(e)
    
    def add_custom_column(self, df, column, value):
        try:
            df[column] = value
            logger.info(f"Column '{column}' added.")
        except Exception as e:
            logger.error(e)


    def count_missing_rows(self, df):
        try:
            # Calculate total number rows with missing values
            missing_values = sum([True for idx,row in df.iterrows() if any(row.isna())])
            logger.info(f"{missing_values} rows have missing values")
            return missing_values
        except Exception as e:
            logger.error(e)
    
    def percent_missing_values(self, df):

        try:
            # Count number of missing values per column
            missingCount = df.isnull().sum()

            # Calculate total number of missing values
            totalMissing = missingCount.sum()

            logger.info(f"{totalMissing} missing values found")
            return totalMissing
        except Exception as e:
            logger.error(e)
    
    def table_for_missing_values(self, df):
        try:
            # Total missing values
            mis_val = df.isnull().sum()

            # Percentage of missing values
            mis_val_percent = 100 * mis_val / len(df)

            # dtype of missing values
            mis_val_dtype = df.dtypes

            # Make a table with the results
            mis_val_table = pd.concat([mis_val, mis_val_percent, mis_val_dtype], axis=1)

            # Rename the columns
            mis_val_table_ren_columns = mis_val_table.rename(
            columns = {0 : 'Missing Values', 1 : '% of Total Values', 2: 'Dtype'})

            # Sort the table by percentage of missing descending and remove columns with no missing values
            mis_val_table_ren_columns = mis_val_table_ren_columns[
                mis_val_table_ren_columns.iloc[:,0] != 0].sort_values(
            '% of Total Values', ascending=False).round(2)

            # Print some summary information
            logger.info("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                "There are " + str(mis_val_table_ren_columns.shape[0]) +
                " columns that have missing values.")

            if mis_val_table_ren_columns.shape[0] == 0:
                return

            # Return the dataframe with missing information
            return mis_val_table_ren_columns
        except Exception as e:
            logger.error(e)

    def get_unique_values(self, df):
        try:
            unique_values = {'Column': [], 'Unique values': []}
            for col in df:
                unique_values['Column'].append(col)
                values = df[col].value_counts().index.to_list()
                unique_values['Unique values'].append(values)
            tmp = pd.DataFrame(unique_values)
            return tmp
        except Exception as e:
            logger.error(e)
            return None
    
    def fix_missing_value(self, df, cols, value):
        try:
            for col in cols:
                count = df[col].isna().sum()
                df[col] = df[col].fillna(value)
                if type(value) == 'str':
                    logger.info(f"{count} missing values in the column {col} have been replaced by \'{value}\'.")
                else:
                    logger.info(f"{count} missing values in the column {col} have been replaced by {value}.")
        except Exception as e:
            logger.error(e)

    def convert_to_int(self, df, columns):
        try:
            for col in columns:
                df[col] = df[col].astype("int64")
            logger.info(f"{columns} converted to int.")
        except Exception as e:
            logger.error(e)

    def getMonth(self, month_list, index):
        months = ['0', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
        month_list = month_list.split(',')
        month = month_list[index]
        return months.index(month)

    def scale_column(self, df, column: str, range_tup: tuple = (0, 1)) -> pd.DataFrame:
        """
            Returns the objects DataFrames column normalized using Normalizer
            Parameters
        """
        try:
            std_column_df = pd.DataFrame(df[column])
            std_column_values = std_column_df.values
            minmax_scaler = MinMaxScaler(feature_range=range_tup)
            normalized_data = minmax_scaler.fit_transform(std_column_values)
            df[column] = normalized_data
            return df
        except Exception as e:
            logger.error(e)

    def scale_columns(self, df, columns: list, range_tup: tuple = (0, 1)) -> pd.DataFrame:
        try:
            for col in columns:
                df = self.scale_column(df, col, range_tup)
            logger.info(f"{columns} scaled between {range_tup}")
            return df
        except Exception as e:
            logger.error(e)
            return None

    def change_datatypes(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        A simple function which changes the data types of the dataframe and returns it
        """
        try:
            data_types = dataframe.dtypes
            changes = ['float64', 'int64']
            for col in data_types.index:
                if(data_types[col] in changes):
                    if(data_types[col] == 'float64'):
                        dataframe[col] = pd.to_numeric(
                            dataframe[col], downcast='float')
                    elif(data_types[col] == 'int64'):
                        dataframe[col] = pd.to_numeric(
                            dataframe[col], downcast='unsigned')
            logger.info(f"Data types converted.")       
        except Exception as e:
            logger.error(e)

        return dataframe
    
    def get_difference(self, dataset, interval=1):
        """
         A function to get the difference of scaled data
        """
        try:
            diff = list()
            for i in range(interval, len(dataset)):
                value = dataset[i] - dataset[i - interval]
                diff.append(value)
            logger.info(f"{len(diff)} fetched")
            return pd.Series(diff)
        except Exception as e:
            logger.error(e)
            return None




    def run_pipeline(self, df, date_cols=[], string_cols=[] ):
        self.convert_to_string(df, string_cols)
        self.convert_to_datetime(df, date_cols)
        self.add_custom_column(df, 'Year', df['Date'].apply(lambda x: x.year))
        self.add_custom_column(df, 'Month', df['Date'].apply(lambda x: x.month))
        self.add_custom_column(df, 'DayOfMonth', df['Date'].apply(lambda x: x.day))
        self.add_custom_column(df, 'WeekOfYear', df['Date'].apply(lambda x: x.weekofyear))
        self.add_custom_column(df, 'weekday', df['DayOfWeek'].apply(lambda x: 0 if (x in [6, 7]) else 1))
    
    def run_pipeline_for_store(self, df, string_cols=[], int_cols=[]):
        self.convert_to_string(df, string_cols)
        self.convert_to_int(df, int_cols)
        df['PromoInterval0'] = df.PromoInterval.apply((lambda x: self.getMonth(x, 0)))
        df['PromoInterval1'] = df.PromoInterval.apply((lambda x: self.getMonth(x, 1)))
        df['PromoInterval2'] = df.PromoInterval.apply((lambda x: self.getMonth(x, 2)))
        df['PromoInterval3'] = df.PromoInterval.apply((lambda x: self.getMonth(x, 3)))


