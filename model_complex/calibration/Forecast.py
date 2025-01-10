import numpy as np

from ..models import BRModel
from ..utils import ModelParams

class Forecast:

    # TODO:     @classmethod
    def __init__(
        self,
        data: list,
        model: BRModel,
        init_infectious: list[int],
        alpha: list[int],
        beta: list[int],
        rho: int,
        duration: int,
    ) -> None:
        
        self.data = data
        self.model = model
        self.init_infectious = init_infectious
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.duration = duration


    # TODO: добавить усреднение
    def forecast(self):
        data_size = len(self.data)//len(self.init_infectious) + self.duration

        res = np.array(
                [[[float('inf'), float('-inf')] for _ in range(data_size)] for j in range(len(self.init_infectious))]
            )

        simulate_pars = ModelParams(
            alpha=[0],
            beta=[0],
            population_size=self.rho,
            initial_infectious=self.init_infectious
        )

        for a, b in zip(zip(*self.alpha), zip(*self.beta)):

            simulate_pars.alpha = a
            simulate_pars.beta = b
            
            self.model.simulate(
                pars=simulate_pars,
                modeling_duration=data_size
            )

            new_res = list(self.model.get_daily_newly_infected().values())

            for i in range(len(self.init_infectious)):
                res[i, :, 0] = np.minimum(res[i, :, 0], new_res[i])
                res[i, :, 1] = np.maximum(res[i, :, 1], new_res[i])

        return res