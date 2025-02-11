from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
import requests

from .epid_data import EpidData

pd.options.mode.copy_on_write = True

# TODO: remove dicts from global namespace
table_from_dict = {
    "YEAR": "datetime",
    "REGION_NAME": "region_name",
    "DISTRICT_NAME": "district_name",
    "ARI_TOTAL": "sars_total_cases",
    "ARI_0_2": "sars_cases_age_group_0",
    "ARI_3_6": "sars_cases_age_group_1",
    "ARI_7_14": "sars_cases_age_group_2",
    "ARI_15_64": "sars_cases_age_group_3",
    "ARI_65": "sars_cases_age_group_4",
    "POP_TOTAL": "total_population",
    "POP_0_2": "population_age_group_0",
    "POP_3_6": "population_age_group_1",
    "POP_7_14": "population_age_group_2",
    "POP_15_64": "population_age_group_3",
    "POP_65": "population_age_group_4",
    "SWB_TOTAL": "tested_total",
    "A_TOTAL": "tested_strain_0",
    "PDM_TOTAL": "tested_strain_1",
    "H3_TOTAL": "tested_strain_2",
    "B_TOTAL": "tested_strain_3",
}

# ARI_ - ОРВИ
# POP_ - Население

# SWB_ - Число образцов на грипп
# A_ - Положительные на грипп А (не субтипировано)
# PDM_ - Положительные на грипп H1pdm09
# H3_ - Положительные на грипп H3
# B_ - Положительные на грипп B


def date_creation(input):
    return datetime.strptime(f'{input["YEAR"]}-W{input["WEEK"]}-1', "%G-W%V-%u")


class InfluenzaData(EpidData):
    """
    InfluenzaData class

    Download epidemiological data from https://db.influenza.spb.ru
    """

    url = (
        "https://db.influenza.spb.ru/scripts/report/rmancgi.exe"
        + "?reportname=get_csv&id=aripcr&byear={}&bweek={}&eyear={}&eweek={}&auth={}"
    )

    def __init__(
        self, city: str, begin_year: int, begin_week: int, end_year: int, end_week: int
    ) -> None:

        self.strain_dict = {
            "A (субтип не определен)": 0,
            "A(H1)pdm09": 1,
            "A(H3)": 2,
            "B": 3,
        }
        self.strain_dict = {
            "A (субтип не определен)": 0,
            "A(H1)pdm09": 1,
            "A(H3)": 2,
            "B": 3,
        }
        self.strains_number = 4
        self.df = None
        self.pcr_df = None
        self.returned_df = None

        # получаем данные
        response = requests.get(
            self.url.format(begin_year, begin_week, end_year, end_week, self.secret)
        )
        data = response.content.decode("utf-8")

        self.df = pd.read_csv(StringIO(data), sep="|")

        # преобразуем в даты
        self.df["YEAR"] = self.df.apply(date_creation, axis=1)

        self.df = self.df.loc[:, list(table_from_dict.keys())]

        self.df = self.df.rename(columns=table_from_dict).fillna(float("nan"))

        for strain_index in range(self.strains_number):
            self.df[f"rel_strain_{strain_index}"] = (
                self.df[f"tested_strain_{strain_index}"] / self.df["tested_total"]
            )
            self.df[f"real_cases_strain_{strain_index}"] = (
                self.df[f"rel_strain_{strain_index}"] * self.df["sars_total_cases"]
            ).round()

        self.df = self.df.drop(
            columns=[
                "tested_total",
                "tested_strain_0",
                "tested_strain_1",
                "tested_strain_2",
                "tested_strain_3",
            ]
        )

    def get(self):
        return self.df

    # TODO
    def __transform_data_for_type(self, type: str) -> None:

        self.returned_df["total_cases"] = self.returned_df.fillna(0)[
            ["real_cases_strain_1", "real_cases_strain_2", "real_cases_strain_3"]
        ].sum(axis=1)

        if type == self.REGIME_TOTAL:
            self.returned_df = self.returned_df[
                ["datetime", "total_cases", "total_population"]
            ]

        elif type == self.REGIME_AGE:

            self.returned_df["sars_cases_age_group_0-2"] = self.returned_df.fillna(0)[
                [
                    "sars_cases_age_group_0",
                    "sars_cases_age_group_1",
                    "sars_cases_age_group_2",
                ]
            ].sum(axis=1)

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
                rel_cases_age_group_0_2.iloc[1] + rel_cases_age_group_3.iloc[1]
            )

            assert to_assert < 1e-5

            # final calculated cases
            self.returned_df["age_group_0-2_cases"] = (
                rel_cases_age_group_0_2 * self.returned_df["total_cases"]
            )
            self.returned_df["age_group_3_cases"] = (
                rel_cases_age_group_3 * self.returned_df["total_cases"]
            )

            self.returned_df = self.returned_df[
                [
                    "datetime",
                    "age_group_0-2_cases",
                    "age_group_3_cases",
                    "total_population",
                ]
            ]

    def __set_timedelta(self):
        deltatime = self.returned_df["datetime"]
        deltadays = (deltatime.iloc[1] - deltatime.iloc[0]).days

        self.returned_df.attrs = {"time_step": "week" if deltadays == 7 else "day"}

    def get_wave_data(self, type: str) -> pd.DataFrame:
        """
        Obtaining data for the epidemiological wave

        :param type: Name of type

        :return: Epidemiological wave
        """
        self.__get_time_period()
        assert isinstance(self.returned_df, pd.DataFrame)
        self.__transform_data_for_type(type)
        self.__set_timedelta()

        return self.returned_df

    def get_rho(self) -> int:
        """
        Get number of people in population
        :return: Number of people
        """
        return self.returned_df["total_population"].iloc[0]

    def prepare_for_plot(self) -> np.array:
        """
        Obtaining data in graphing format
        :return: Plot data
        """
        return np.array(self.returned_df.drop(columns=["datetime", "total_population"]))

    def get_data(self) -> np.array:
        """
        Obtaining data for calibration
        :return: Data for calibration
        """
        return self.returned_df.drop(columns=["total_population"])

    def get_duration(self) -> int:
        return (
            len(self.returned_df) * 7
            if self.returned_df.attrs["time_step"] == "week"
            else len(self.returned_df)
        )
