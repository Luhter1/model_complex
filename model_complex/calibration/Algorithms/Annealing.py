import numpy as np
from scipy.optimize import dual_annealing
from sklearn.metrics import r2_score

from ...models import Model
from ...utils import ModelParams


class Annealing:

    @classmethod
    def calibrate(
        self,
        rho: int,
        model: Model,
        init_infectious: list[int],
        data: np.array,
    ):

        alpha_len, beta_len = model.params()
        duration = (len(data) // alpha_len) * 7

        simulate_pars = ModelParams(
            alpha=[0],
            beta=[0],
            population_size=rho,
            initial_infectious=init_infectious,
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

            return -r2_score(data, model.get_weekly_newly_infected())

        ret = dual_annealing(AnnealingModel, bounds=list(zip(lw, up)))

        simulate_pars.alpha = ret.x[:alpha_len]
        simulate_pars.beta = ret.x[alpha_len:]

        model.set_ci_params([])
        model.set_best_params(simulate_pars)
