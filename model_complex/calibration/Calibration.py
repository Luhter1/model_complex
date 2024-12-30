import numpy as np
import optuna
import pymc as pm
from scipy.optimize import dual_annealing
from sklearn.metrics import r2_score

from ..models import Model
from ..utils import ModelParams

optuna.logging.set_verbosity(optuna.logging.ERROR)


class Calibration:
    """
    Calibration class
    """

    def __init__(
        self, model: Model, data: list, population_size: int, init_infectious: list
    ) -> None:
        """
        Calibration class

        TODO

        :param init_infectious: Number of initial infected people
        :param model: Model for calibration
        :param data: Observed data for calibrating process
        :param population_size: People's population
        """
        self.population_size = population_size
        self.init_infectious = init_infectious
        self.model = model
        self.data = data

    def abc_calibration(self, sample=100, epsilon=3000, with_rho=False):
        """
        TODO

        """
        # !!!! Надо ли сохранять with_rho
        alpha_len, beta_len = self.model.alpha_beta_dims()

        population_size = self.population_size

        # ф-я симуляции
        def simulation_func(rng, alpha, beta, pop_size, size=None):

            new_params = ModelParams(
                alpha=alpha,
                beta=beta,
                initial_infectious=self.init_infectious,
                population_size=pop_size,
            )

            self.model.simulate(
                model_params=new_params,
                modeling_duration=len(self.data) // alpha_len,
            )
            return self.model.get_newly_infected()

        # модель
        with pm.Model() as model:
            alpha = pm.Uniform(name="alpha", lower=0, upper=1, shape=(alpha_len,))
            beta = pm.Uniform(name="beta", lower=0, upper=1, shape=(beta_len,))

            if with_rho:
                population_size = pm.Uniform(
                    name="population_size", lower=with_rho[0], upper=with_rho[1]
                )

            sim = pm.Simulator(
                "sim",
                simulation_func,
                list(alpha) + [0] * (beta_len - alpha_len),
                beta,
                population_size,
                epsilon=epsilon,
                observed=self.data,
            )

            idata = pm.sample_smc(progressbar=False)

        posterior = idata.posterior.stack(samples=("draw", "chain"))

        alpha = [
            np.random.choice(posterior["alpha"][i], size=sample) for i in range(alpha_len)
        ]
        beta = [
            np.random.choice(posterior["beta"][i], size=sample) for i in range(beta_len)
        ]

        new_params = ModelParams(
            alpha=[a.mean() for a in alpha],
            beta=[b.mean() for b in beta],
            initial_infectious=self.init_infectious,
            population_size=self.population_size,
        )

        self.model.simulate(
            model_params=new_params,
            modeling_duration=len(self.data) // alpha_len,
        )

        return alpha, beta

    def optuna_calibration(self, n_trials=1000):
        """
        TODO

        """

        alpha_len, beta_len = self.model.alpha_beta_dims()

        def model(trial):

            alpha = [trial.suggest_float(f"alpha_{i}", 0, 1) for i in range(alpha_len)]
            beta = [trial.suggest_float(f"beta_{i}", 0, 1) for i in range(beta_len)]

            self.model.simulate(
                alpha=alpha,
                beta=beta,
                initial_infectious=self.init_infectious,
                population_size=self.population_size,
                modeling_duration=len(self.data) // alpha_len,
            )

            return r2_score(self.data, self.model.get_newly_infected())

        study = optuna.create_study(direction="maximize")
        study.optimize(model, n_trials=n_trials)

        alpha = [study.best_params[f"alpha_{i}"] for i in range(alpha_len)]
        beta = [study.best_params[f"beta_{i}"] for i in range(beta_len)]

        # запускаем, чтобы в модели были результаты с лучшими параметрами
        new_params = ModelParams(
            alpha=alpha,
            beta=beta,
            initial_infectious=self.init_infectious,
            population_size=self.population_size,
        )

        self.model.simulate(
            model_params=new_params,
            modeling_duration=len(self.data) // alpha_len,
        )

        return alpha, beta

    def annealing_calibration(self):
        """
        TODO

        """

        alpha_len, beta_len = self.model.alpha_beta_dims()

        lw = [0] * (alpha_len + beta_len)
        up = [1] * (alpha_len + beta_len)

        def model(x):

            alpha = x[:alpha_len]
            beta = x[alpha_len:]

            new_params = ModelParams(
                alpha=alpha,
                beta=beta,
                initial_infectious=self.init_infectious,
                population_size=self.population_size,
            )

            self.model.simulate(
                model_params=new_params,
                modeling_duration=len(self.data) // alpha_len,
            )

            return -r2_score(self.data, self.model.get_newly_infected())

        ret = dual_annealing(model, bounds=list(zip(lw, up)))

        alpha = ret.x[:alpha_len]
        beta = ret.x[alpha_len:]

        # запускаем, чтобю в модели были результаты с лучшими параметрами
        new_params = ModelParams(
            alpha=alpha,
            beta=beta,
            initial_infectious=self.init_infectious,
            population_size=self.population_size,
        )

        self.model.simulate(
            model_params=new_params,
            modeling_duration=len(self.data) // alpha_len,
        )

        return alpha, beta

    def mcmc_calibration(
        self,
        sample=100,
        epsilon=10000,
        with_rho=False,  # [50_000, 500_000] - если True
        with_initi=False,  # [1, 1_000] - если True
        tune=2500,
        draws=500,
        chains=4,
    ):
        """
        Parameters:
            - with_rho -- tune population size
            - with_initi -- tune initial infected
            - tune -- number of mcmc warmup samples
            - draws -- number of mcmc draws
            - chains -- number of chains
        """

        alpha_len, beta_len = self.model.alpha_beta_dims()
        init_infectious = self.init_infectious
        population_size = self.population_size

        def simulation_func(
            rng, alpha, beta, population_size, init_infectious, size=None
        ):

            new_params = ModelParams(
                alpha=alpha,
                beta=beta,
                initial_infectious=init_infectious,
                population_size=population_size,
            )

            self.model.simulate(
                model_params=new_params,
                modeling_duration=len(self.data) // alpha_len,
            )

            return self.model.get_newly_infected()

        with pm.Model() as pm_model:
            alpha = pm.Uniform(name="alpha", lower=0, upper=1, shape=(alpha_len,))
            beta = pm.Uniform(name="beta", lower=0, upper=1, shape=(beta_len,))

            if with_rho:
                population_size = pm.Uniform(
                    name="population_size", lower=with_rho[0], upper=with_rho[1]
                )

            if with_initi:
                init_infectious = pm.Uniform(
                    name="init_infectious",
                    lower=with_initi[0],
                    upper=with_initi[1],
                    shape=(alpha_len,),
                )

            sim = pm.Simulator(
                "sim",
                simulation_func,
                list(alpha) + [0] * (beta_len - alpha_len),
                beta,
                population_size,
                list(init_infectious) + [0] * (beta_len - len(init_infectious)),
                epsilon=epsilon,
                observed=self.data,
            )

            # Differential evolution (DE) Metropolis sampler
            # step=pm.DEMetropolisZ(proposal_dist=pm.LaplaceProposal)
            step = pm.DEMetropolisZ()

            idata = pm.sample(
                tune=tune,
                draws=draws,
                chains=chains,
                step=step,
                progressbar=False,
            )
            idata.extend(pm.sample_posterior_predictive(idata, progressbar=False))

        posterior = idata.posterior.stack(samples=("draw", "chain"))

        alpha = [
            np.random.choice(posterior["alpha"][i], size=sample) for i in range(alpha_len)
        ]
        beta = [
            np.random.choice(posterior["beta"][i], size=sample) for i in range(beta_len)
        ]

        # запускаем, чтобю в модели были результаты с лучшими параметрами
        # self.model.simulate(
        #     alpha=[a.mean() for a in alpha],
        #     beta=[b.mean() for b in beta],
        #     initial_infectious=self.init_infectious,
        #     population_size=self.population_size,
        #     modeling_duration=len(self.data) // alpha_len,
        # )

        return alpha, beta
