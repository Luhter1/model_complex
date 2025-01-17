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
        # time_stamp: pd.DataFrame,
    ):

        alpha_len, beta_len = model.params()

        simulate_pars = ModelParams(
            alpha=[0],
            beta=[0],
            population_size=rho,
            initial_infectious=init_infectious,
            # time_stamp=time_stamp
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
                modeling_duration=len(data) // alpha_len,
            )

            return -r2_score(data, model.newly_infected)

        ret = dual_annealing(AnnealingModel, bounds=list(zip(lw, up)))

        alpha = ret.x[:alpha_len]
        beta = ret.x[alpha_len:]

        # запускаем, чтобы в модели были результаты с лучшими параметрами
        simulate_pars.alpha = alpha
        simulate_pars.beta = beta

        model.set_ci_params([])
        model.set_best_params(simulate_pars)

        return alpha, beta
