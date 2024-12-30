import numpy as np

from ..models import Model
from ..utils import ModelParams


class Forecast:
    @classmethod
    def forecast(
        data: list,
        model: Model,
        model_params: ModelParams,
        duration: int,
    ) -> np.array:

        data_size = duration + (len(data) // len(model_params.initial_infectious))
        res = np.array(
            [
                [[float("inf"), float("-inf")] for _ in range(data_size)]
                for j in range(len(model_params.initial_infectious))
            ]
        )

        # to avoid creating new_params again at each loop
        new_params = ModelParams(
            alpha=[0],
            beta=[0],
            initial_infectious=[0],
            population_size=0,
        )

        for a, b in zip(zip(*model_params.alpha), zip(*model_params.beta)):

            new_params.alpha = a
            new_params.beta = b
            new_params.initial_infectious = model_params.initial_infectious
            new_params.population_size = model_params.population_size

            model.simulate(model_params=new_params, modeling_duration=data_size)

            new_res = np.array(model.get_result())

            for i in range(len(model_params.initial_infectious)):
                res[i, :, 0] = np.minimum(res[i, :, 0], new_res[i])
                res[i, :, 1] = np.maximum(res[i, :, 1], new_res[i])

        return res
