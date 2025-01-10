from ...utils import ModelParams

class Model:
    """
    Interface for all Models
    """

    br_func_array = [0.1, 0.1, 1, 0.9, 0.55, 0.3, 0.15, 0.05]

    is_ci_ready = False
    is_calibrated = False
    best_calibration_params: ModelParams = None
    ci_params: list[ModelParams] = None
    newly_infected = None

    def __init__(self):
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


    def _chunk(self, lst, size):
        return [lst[i : i + size] for i in range(0, len(lst), size)]


    def get_result(self):
        size = len(self.newly_infected) // self.alpha_dim

        return self._chunk(self.newly_infected, size)
