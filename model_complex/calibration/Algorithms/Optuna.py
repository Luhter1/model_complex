from sklearn.metrics import r2_score
import numpy as np
import optuna

from ...models import BRModel
from ...utils import ModelParams


class Optuna:

    @classmethod
    def calibrate(
            self, 
            rho: int, 
            model: BRModel, 
            init_infectious: list[int], 
            data: np.array,
            n_trials=1000
        ):
    
        alpha_len, beta_len = model.params()

        simulate_pars = ModelParams(
            alpha=[0],
            beta=[0],
            population_size=rho,
            initial_infectious=init_infectious
        )

        def OptunaModel(trial):

            alpha = [trial.suggest_float(f"alpha_{i}", 0, 1) for i in range(alpha_len)]
            beta = [trial.suggest_float(f"beta_{i}", 0, 1) for i in range(beta_len)]

            simulate_pars.alpha = alpha
            simulate_pars.beta = beta
            
            model.simulate(
                pars=simulate_pars,
                modeling_duration=len(data) // alpha_len,
            )

            return r2_score(data, model.newly_infected)

        study = optuna.create_study(direction="maximize")
        study.optimize(OptunaModel, n_trials=n_trials)

        alpha = [study.best_params[f"alpha_{i}"] for i in range(alpha_len)]
        beta = [study.best_params[f"beta_{i}"] for i in range(beta_len)]

        # запускаем, чтобы в модели были результаты с лучшими параметрами
        simulate_pars.alpha = alpha
        simulate_pars.beta = beta
        
        model.simulate(
            pars=simulate_pars,
            modeling_duration=len(data) // alpha_len,
        )
        
        return alpha, beta