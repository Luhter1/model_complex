import numpy as np

from ...utils import ModelParams


class Model:
    GROUPS_NUMBER = 0

    br_func_array = [0.1, 0.1, 1, 0.9, 0.55, 0.3, 0.15, 0.05]

    is_ci_ready = False
    is_calibrated = False
    calibration_params: ModelParams = None
    ci_params: list[ModelParams] = None
    newly_infected = None

    def __init__(self):
        """
        Interface for all Models
        """
        self.alpha_dim = 0
        self.beta_dim = 0

    def simulate(self, pars: ModelParams, modeling_duration: int):
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

    def get_daily_newly_infected(self):
        data_arrays = np.array_split(self.newly_infected, self.GROUPS_NUMBER)

        return {index: data_arrays[index] for index in range(self.GROUPS_NUMBER)}


    def get_weekly_newly_infected(self):
        return np.array(self.newly_infected).reshape(-1, 7).sum(axis=1)
    
    def get_weekly_newly_infected_by_group(self):
        return self.get_weekly_newly_infected().reshape(self.alpha_dim, -1)

    def set_best_params(self, best_params: ModelParams):
        self.calibration_params = best_params
        self.is_calibrated = True

    def set_ci_params(self, ci_params: list[ModelParams]):
        self.ci_params = ci_params
        self.is_ci_ready = True

    def get_best_params(self) -> ModelParams:
        if self.is_calibrated:
            return self.calibration_params
        else:
            raise Exception("Model is not calibrated!")

    def get_ci_params(self) -> list[ModelParams]:
        if self.is_ci_ready:
            return self.ci_params
        else:
            raise Exception("Model does not have set of parameters for CI construction!")
