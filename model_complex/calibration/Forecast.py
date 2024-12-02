import numpy as np

from ..models import BRModel

class Forecats:

    # alpha, beta
    # data
    # на сколько строить прогноз
    # model
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



    def forecast(self):
        data_size = len(self.data)//len(self.init_infectious) + self.duration

        res = np.array(
                [[[float('inf'), float('-inf')] for _ in range(data_size)] for j in range(len(self.init_infectious))]
            )

        for a, b in zip(zip(*self.alpha), zip(*self.beta)):

            self.model.simulate(
                alpha=a, 
                beta=b, 
                initial_infectious=self.init_infectious, 
                rho=self.rho, 
                modeling_duration=data_size
            )

            new_res = np.array(self.model.get_result())

            for i in range(len(self.init_infectious)):
                res[i, :, 0] = np.minimum(res[i, :, 0], new_res[i])
                res[i, :, 1] = np.maximum(res[i, :, 1], new_res[i])

        return res