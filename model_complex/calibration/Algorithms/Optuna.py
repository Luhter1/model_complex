import numpy as np
import optuna
from sklearn.metrics import r2_score

from ...models import Model
from ...utils import ModelParams


class Optuna:

    @classmethod
    def calibrate(
        self,
        model: Model,
        data: np.array,
        discretisation: str,
        model_pars: ModelParams,
        n_trials=1000,
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

        def OptunaModel(trial):

            alpha = [trial.suggest_float(f"alpha_{i}", 0, 1) for i in range(alpha_len)]
            beta = [trial.suggest_float(f"beta_{i}", 0, 1) for i in range(beta_len)]

            simulate_pars.alpha = alpha
            simulate_pars.beta = beta

            model.simulate(
                pars=simulate_pars,
                modeling_duration=duration,
            )

            return r2_score(data, get_newly_infected_base_on_discretisation())

        study = optuna.create_study(direction="maximize")
        study.optimize(OptunaModel, n_trials=n_trials)

        simulate_pars.alpha = [study.best_params[f"alpha_{i}"] for i in range(alpha_len)]
        simulate_pars.beta = [study.best_params[f"beta_{i}"] for i in range(beta_len)]

        model.set_ci_params([])
        model.set_best_params(simulate_pars)
