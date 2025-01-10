import numpy as np

from ...utils import ModelParams

class Model:
    GROUPS_NUMBER = 0

    br_func_array = [0.1, 0.1, 1, 0.9, 0.55, 0.3, 0.15, 0.05]

    is_ci_ready = False
    is_calibrated = False
    best_calibration_params: ModelParams = None
    ci_params: list[ModelParams] = None
    newly_infected = None


    def __init__(self):
        """
        Interface for all Models
        """
        self.alpha_dim = 0
        self.beta_dim = 0


    def simulate(
        self, 
        pars: ModelParams,
        modeling_duration: int
    ):
        pass


    def br_function(self, day: int) -> int:
        """
        Baroyan-Rvachev function

        :param day: Illness day

        :return: human virulence
        """

        if day >= len(self.br_func_array):
            return 0
        return self.br_func_array[day]


    def params(self):
        """
        TODO
        """
        return (self.alpha_dim, self.beta_dim)


    def get_result(self):
        return self.get_daily_newly_infected()


    def get_daily_newly_infected(self):
        data_arrays = np.array_split(
            self.newly_infected, 
            self.GROUPS_NUMBER
        )

        return {index: data_arrays[index] for index in range(self.GROUPS_NUMBER)}
