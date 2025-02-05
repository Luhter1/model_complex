import numpy as np
from scipy.optimize import dual_annealing
from sklearn.metrics import r2_score

from ...models import Model
from ...utils import ModelParams


class Annealing:

    @classmethod
    def calibrate(
        self,
        model: Model,
        data: np.array,
        discretisation: str,
        model_pars: ModelParams,
    ):
        alpha_len, beta_len = model.params()
        duration = (len(data) // alpha_len)
        get_newly_infected_base_on_discretisation = model.get_daily_newly_infected
        
        if discretisation == "week":
            duration *= 7
            get_newly_infected_base_on_discretisation = model.get_weekly_newly_infected

        simulate_pars = ModelParams(
            alpha=[0],
            beta=[0],
            population_size=model_pars.population_size,
            initial_infectious=model_pars.initial_infectious,
        )

        lw = [0] * (alpha_len + beta_len)
        up = [1] * (alpha_len + beta_len)

        def AnnealingModel(x):

            alpha = x[:alpha_len]
            beta = x[alpha_len:]

            simulate_pars.alpha = alpha
            simulate_pars.beta = beta

            model.simulate(
                pars=simulate_pars,
                modeling_duration=duration
            )

            return -r2_score(data, get_newly_infected_base_on_discretisation())

        ret = dual_annealing(AnnealingModel, bounds=list(zip(lw, up)))

        simulate_pars.alpha = ret.x[:alpha_len]
        simulate_pars.beta = ret.x[alpha_len:]

        model.set_ci_params([])
        model.set_best_params(simulate_pars)
