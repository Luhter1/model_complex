import numpy as np
import pymc as pm

from ..Interface import Model
from ...utils import ModelParams


class TotalBRModel(Model):
    """
    Baroyan-Rvachev model for total case
    """

    def __init__(self) -> None:
        self.alpha_dim = 1
        self.beta_dim = 1


    def simulate(
        self,
        model_params: ModelParams,
        modeling_duration: int
    ) -> None:
        """
        Launch simulation using Baroyan-Rvachev model for total case

        :param model_params: class that contains model parameters
        :param modeling_duration: duration of modeling

        :return:
        """
        assert len(model_params.alpha) == self.alpha_dim
        assert len(model_params.beta) == self.beta_dim
        assert len(model_params.initial_infectious) == self.alpha_dim
        assert (np.all(np.array(model_params.alpha) > 0) 
            and np.all(np.array(model_params.alpha) < 1))
        assert (np.all(np.array(model_params.beta) > 0) 
            and np.all(np.array(model_params.beta) < 1))
        
        # SETTING UP INITIAL CONDITIONS
        initial_susceptible = int(model_params.alpha[0] 
                                  * model_params.population_size)

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
                (
                    model_params.beta[0] * susceptible[day] 
                    * total_infected[day] / model_params.population_size
                ),
                susceptible[day]
            )

            susceptible[day + 1] = susceptible[day] - newly_infected[day + 1]

        self.newly_infected = newly_infected
