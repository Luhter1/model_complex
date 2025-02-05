from datetime import timedelta

import numpy as np
import pandas as pd

from ..models import Model
from ..utils import ModelParams


class Forecast:

    @classmethod
    def forecast(
        self,
        model: Model,
        data: pd.DataFrame,
        duration: timedelta,
    ):

        ci_pars = model.get_ci_params()
        calibration_duration = len(data) # if self.returned_df.attrs["discretisation"] == "week" else len(self.returned_df)
        group_num = len(ci_pars[0].initial_infectious)

        if data.attrs["discretisation"] == "week":
            calibration_duration *= 7
            get_newly_infected_base_on_discretisation = model.get_weekly_newly_infected_by_group
            # округляем вверх кол-во недель, чтобы не выравнивать numpy матрицу
            duration = calibration_duration + (duration.days+6)//7 * 7 
            count_of_min_mean_max_points = duration//7
        else:
            get_newly_infected_base_on_discretisation = model.get_daily_newly_infected_by_group
            duration = calibration_duration + duration.days
            count_of_min_mean_max_points = duration

        min_mean_max = np.array(
            [
                [[float("inf"), 0, float("-inf")] for _ in range(count_of_min_mean_max_points)]
                for j in range(group_num)
            ]
        )


        for pars in ci_pars:

            model.simulate(
                pars=pars, 
                modeling_duration=duration
            )

            new_result = get_newly_infected_base_on_discretisation()

            for i in range(group_num):
                min_mean_max[i, :, 0] = np.minimum(min_mean_max[i, :, 0], new_result[i])
                min_mean_max[i, :, 2] = np.maximum(min_mean_max[i, :, 2], new_result[i])

        model.simulate(
            pars=model.get_best_params(), 
            modeling_duration=duration
        )

        new_result = get_newly_infected_base_on_discretisation()

        for i in range(group_num):
            min_mean_max[i, :, 1] = new_result[i]


        return min_mean_max
