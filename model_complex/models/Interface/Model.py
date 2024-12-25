from ...utils import ModelParams

import numpy as np

class Model:
    """
    Interface for models from Model complex
    """

    br_func_array = [0.1, 0.1, 1, 0.9, 0.55, 0.3, 0.15, 0.05]

    is_calibrated = False
    ready_for_ci = False
    best_calibration_params: ModelParams = None
    ci_params: list[ModelParams] = None


    def __init__(self) -> None:
        self.alpha_dim = 0
        self.beta_dim = 0


    def simulate(
        self,
        model_params: ModelParams,
        modeling_duration: int
    ) -> None:
        """
        Launch simulation

        :param model_params: class that contains model parameters
        :param modeling_duration: duration of modeling

        :return:
        """
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


    def alpha_beta_dims(self) -> tuple:
        """
        Dimensionality of parameters alpha and beta

        :param day:

        :return: tuple of alpha and beta dimensions
        """
        return (self.alpha_dim, self.beta_dim)


    def _chunk(self, lst, size) -> list:
        return [lst[i : i + size] for i in range(0, len(lst), size)]


    def get_result(self):
        """
        TODO
        """
        size = len(self.newly_infected) // self.alpha_len

        return self._chunk(self.newly_infected, size)


    def get_newly_infected(self) -> np.array:
        """
        TODO
        """
        return self.newly_infected


    def set_best_params_after_calibration(
            self, 
            best_params: ModelParams
        ) -> None:
        """
        TODO
        """
        self.best_calibration_params = best_params
        self.is_calibrated = True


    def set_ci_params(self, ci_params: ModelParams) -> None:
        """
        TODO
        """
        self.ci_params = ci_params
        self.ready_for_ci = True


    def get_best_params(self) -> ModelParams:
        """
        TODO
        """        
        if self.is_calibrated:
            return self.best_calibration_params
        else:
            raise Exception('Model is not calibrated!')


    def get_ci_params(self) -> list[ModelParams]:
        """
        TODO
        """        
        if self.ready_for_ci:
            return self.ci_params
        else:
            raise Exception(
                    'Model does not have set of parameters' 
                    + 'for CI construction!'
                )