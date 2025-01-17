import optuna

from ..models import Model
from .Algorithms import ABC, MCMC, Annealing, Optuna

optuna.logging.set_verbosity(optuna.logging.ERROR)


class Calibration:

    def __init__(
        self,
        init_infectious: list[int],
        model: Model,
        data: list,
        rho: int,
    ) -> None:
        """
        Calibration class

        TODO

        :param init_infectious: Number of initial infected people
        :param model: Model for calibration
        :param data: Observed data for calibrating process
        :param rho: People's population
        """
        self.rho = rho
        self.model = model
        self.init_infectious = init_infectious
        self.time_stamp = data["datetime"]
        self.data = data.drop(columns=["datetime"]).to_numpy().T.flatten()

    def abc_calibration(self, sample=100, epsilon=3000):

        return ABC.calibrate(
            rho=self.rho,
            model=self.model,
            init_infectious=self.init_infectious,
            data=self.data,
            # time_stamp=self.time_stamp,
            sample=sample,
            epsilon=epsilon,
        )

    def optuna_calibration(self, n_trials=1000):

        return Optuna.calibrate(
            rho=self.rho,
            model=self.model,
            init_infectious=self.init_infectious,
            data=self.data,
            # time_stamp=self.time_stamp,
            n_trials=n_trials,
        )

    def annealing_calibration(self):

        return Annealing.calibrate(
            rho=self.rho,
            model=self.model,
            init_infectious=self.init_infectious,
            data=self.data,
            # time_stamp=self.time_stamp,
        )

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

        return MCMC.calibrate(
            rho=self.rho,
            model=self.model,
            init_infectious=self.init_infectious,
            data=self.data,
            # time_stamp=self.time_stamp,
            sample=sample,
            epsilon=epsilon,
            with_rho=with_rho,  # [50_000, 500_000] - если True
            with_initi=with_initi,  # [1, 1_000] - если True
            tune=tune,
            draws=draws,
            chains=chains,
        )
