import numpy as np

from ...utils import ModelParams
from ..Interface import Model


class AgeGroupBRModel(Model):
    """
    Baroyan-Rvachev model for case of several age groups
    """

    def __init__(self) -> None:
        self.alpha_dim = 2
        self.beta_dim = 4

    def simulate(self, model_params: ModelParams, modeling_duration: int) -> None:
        """
        Launch simulation using Baroyan-Rvachev model for case of several age groups

        :param model_params: class that contains model parameters
        :param modeling_duration: duration of modeling

        :return:
        """
        assert len(model_params.alpha) == self.alpha_dim
        assert len(model_params.beta) == self.beta_dim
        assert len(model_params.initial_infectious) == self.alpha_dim
        assert np.all(np.array(model_params.alpha) > 0) and np.all(
            np.array(model_params.alpha) < 1
        )
        assert np.all(np.array(model_params.beta) > 0) and np.all(
            np.array(model_params.beta) < 1
        )

        self.newly_infected = [
            self.__simulate_one_group(self, model_params, modeling_duration, group_num)
            for group_num in range(2)
        ]

    def __simulate_one_group(
        self, model_params: ModelParams, modeling_duration: int, group_num: int
    ) -> list:
        # SETTING UP INITIAL CONDITIONS
        initial_susceptible = int(
            model_params.alpha[group_num] * model_params.population_size
        )

        total_infected = np.zeros(modeling_duration)
        newly_infected = np.zeros(modeling_duration)
        susceptible = np.zeros(modeling_duration)

        total_infected[0] = model_params.initial_infectious[0]
        newly_infected[0] = model_params.initial_infectious[0]
        susceptible[0] = initial_susceptible

        # SIMULATION
        for day in range(modeling_duration - 1):
            total_infected[day] = min(
                sum(
                    newly_infected[day - tau] * self.br_function(tau)
                    for tau in range(len(self.br_func_array))
                    if (day - tau) >= 0
                ),
                model_params.population_size,
            )

            newly_infected[day + 1] = min(
                sum(
                    model_params.beta[2 * i + group_num]
                    * susceptible[day]
                    * total_infected[day]
                    / model_params.population_size
                    for i in range(2)
                ),
                susceptible[day],
            )

            susceptible[day + 1] = susceptible[day] - newly_infected[day + 1]

        return list(newly_infected)
