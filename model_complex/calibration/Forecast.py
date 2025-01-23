from datetime import timedelta

import numpy as np

from ..models import Model
from ..utils import ModelParams


class Forecast:

    @classmethod
    def forecast(
        self,
        model: Model,
        calibration_duration: int,
        duration: timedelta,
    ):

        ci_pars = model.get_ci_params()
        # округляем вверх кол-во недель, чтобы не выравнивать numpy матрицу
        duration = calibration_duration + (duration.days+6)//7 * 7 

        group_num = len(ci_pars[0].initial_infectious)

        min_mean_max = np.array(
            [
                [[float("inf"), 0, float("-inf")] for _ in range(duration//7)]
                for j in range(group_num)
            ]
        )


        for pars in ci_pars:

            model.simulate(
                pars=pars, 
                modeling_duration=duration
            )

            new_result = model.get_weekly_newly_infected_by_group()

            for i in range(group_num):
                min_mean_max[i, :, 0] = np.minimum(min_mean_max[i, :, 0], new_result[i])
                min_mean_max[i, :, 2] = np.maximum(min_mean_max[i, :, 2], new_result[i])

        model.simulate(
            pars=model.get_best_params(), 
            modeling_duration=duration
        )

        new_result = model.get_weekly_newly_infected_by_group()

        for i in range(group_num):
            min_mean_max[i, :, 1] = new_result[i]


        return min_mean_max
