from datetime.datetime import strptime
import re

import numpy as np
import pandas as pd

pd.options.mode.copy_on_write = True

# TODO: remove dicts from global namespace
cases_table_from_dict_to_excel = {
    "date": "Дата",
    "sars_total_cases": "Всего_орви",
    "sars_cases_age_group_0": "0 - 2_орви",
    "sars_cases_age_group_1": "3 - 6_орви",
    "sars_cases_age_group_2": "7 - 14_орви",
    "sars_cases_age_group_3": "15 и ст._орви",
    "total_population": "Всего",
    "population_age_group_0": "0 - 2",
    "population_age_group_1": "3 - 6",
    "population_age_group_2": "7 - 14",
    "population_age_group_3": "15 и ст.",
}

pcr_table_from_excel_to_python = {
    "date": "Дата",
    "tested_total": "Число образцов тестированных на грипп",
    "tested_strain_0": "A (субтип не определен)",
    "tested_strain_1": "A(H1)pdm09",
    "tested_strain_2": "A(H3)",
    "tested_strain_3": "B",
}


def date_extract(input_string):
    matching = re.search(r"(\d{2}\.\d{2}\.\d{4})", input_string)
    if matching:
        date_string = matching.group(1)
        date_object = strptime(date_string, "%d.%m.%Y")
        return date_object
    else:
        raise Exception("Incorrect date format!")



class EpidData:
    REGIME_TOTAL = "total"
    REGIME_AGE = "age"
    REGIME_STRAIN = "strain"


    def __init__(
        self, 
        city: str, 
        path: str,
        start_time: str, 
        end_time: str
    )-> None:
        """
        EpidData class

        Download epidemiological excel data file from the subdirectory of epid_data.
        epid_data directory looks like 'epid_data/{city}/epid_data.xlsx'.

        :param city: Name of city  
        :param path: path to directory 'epid_data'
        :param start: Start date for extraction
            String of the form "mm-dd-yy"
        :param end: End date for extraction
            String of the form "mm-dd-yy"
        :param regime: Name of regime. Allowed strings: 'total', 'age', 'strain'. 
        """
        self.strain_dict = {
            "A (субтип не определен)": 0,
            "A(H1)pdm09": 1,
            "A(H3)": 2,
            "B": 3,
        }
        self.strains_number = 4
        self.cases_df = None
        self.pcr_df = None
        self.returned_df = None

        self.start_time = strptime(start_time, "%d-%m-%Y")
        self.end_time = strptime(end_time, "%d-%m-%Y")
        self.data_folder = path.rstrip('/') + f'/data/{city}/'


        self.__read__all_data_to_dataframes()


    def __read_data_to_dataframe(self, file, table) -> pd.DataFrame:
        """
        Download excel file from epid_data folder
        :return: pd.DataFrame
        """
        excel_file = pd.read_excel( self.data_folder + file )
        df = pd.DataFrame(columns=table.keys())

        for col_name in df.columns:
            df[col_name] = excel_file[
                table[col_name]
            ]

        df["datetime"] = df["date"].apply(date_extract)
        
        return df.fillna(float("nan"))


    def __read__all_data_to_dataframes(self) -> None:
        """
        Download excel file from epid_data folder
        :return:
        """

        # read cases.xlsx file
        self.cases_df = self.__read_data_to_dataframe(
            "cases.xlsx", 
            cases_table_from_dict_to_excel 
        )


        # read pcr.xlsx file
        self.pcr_df = self.__read_data_to_dataframe(
            "pcr.xlsx", 
            pcr_table_from_excel_to_python
        )
        
        for strain_index in range(self.strains_number):
            str_ind = str(strain_index)
            self.cases_df["rel_strain_" + str_ind] = (
                self.pcr_df["tested_strain_" + str_ind] 
                / self.pcr_df["tested_total"]
            )
            self.cases_df["real_cases_strain_" + str_ind] = (
                self.cases_df["rel_strain_" + str_ind]
                * self.cases_df["sars_total_cases"]
            ).round()


    def __get_time_period(self) -> None:
        """
        Get data from the desired time interval
        :return:
        """
        self.returned_df = self.cases_df[
            (self.cases_df["datetime"] > self.start_time)
            & (self.cases_df["datetime"] < self.end_time)
        ]


    def __transform_data_for_regime(self, regime) -> None:
        # TODO: think about nan values. In epidemic data nan != 0.
        # That is why using method fillna(0) is slightly incorrect.
        self.returned_df["total_cases"] = self.returned_df.fillna(0)[[
            "real_cases_strain_1", 
            "real_cases_strain_2", 
            "real_cases_strain_3"
        ]].sum(axis=1)


        if regime == self.REGIME_TOTAL:
            self.returned_df = self.returned_df[[
                "datetime", 
                "total_cases", 
                "total_population"
            ]]


        if regime == self.REGIME_AGE:
            # sum up cases from age groups: 0-2, 3-6, 7-14 because we work with 0-14 and 15+
            # TODO: think about nan values. In epidemic data nan != 0.
            self.returned_df["sars_cases_age_group_0-2"] = self.returned_df.fillna(0)[[
                    "sars_cases_age_group_0",
                    "sars_cases_age_group_1",
                    "sars_cases_age_group_2",
            ]].sum(axis=1)

            rel_cases_age_group_0_2 = (
                self.returned_df["sars_cases_age_group_0-2"]
                / self.returned_df["sars_total_cases"]
            )

            rel_cases_age_group_3 = (
                self.returned_df["sars_cases_age_group_3"]
                / self.returned_df["sars_total_cases"]
            )

            # check if the sum of relative diseases is not equal to 1
            to_assert = -1 + abs(
                rel_cases_age_group_0_2.iloc[1]
                + rel_cases_age_group_3.iloc[1]
            )

            assert to_assert < 1e-5

            # final calculated cases
            self.returned_df["age_group_0-2_cases"] = (
                rel_cases_age_group_0_2
                * self.returned_df["total_cases"]
            )
            self.returned_df["age_group_3_cases"] = (
                rel_cases_age_group_3
                * self.returned_df["total_cases"]
            )

            self.returned_df = self.returned_df[[
                    "datetime",
                    "age_group_0-2_cases",
                    "age_group_3_cases",
                    "total_population",
            ]]


    def get_wave_data(self, regime)-> pd.DataFrame:
        self.__get_time_period()
        assert isinstance(self.returned_df, pd.DataFrame)
        self.__transform_data_for_regime(regime)

        return self.returned_df


    def get_rho(self) -> int:
        return self.returned_df["total_population"].iloc[0]


    def prepare_for_plot(self) -> np.array:
        return np.array(self.returned_df.drop(columns=[
            "datetime", 
            "total_population"
        ]))


    def prepare_for_calibration(self) -> np.array:
        return self.prepare_for_plot().T.flatten()
