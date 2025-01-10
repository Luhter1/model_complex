import numpy as np
import optuna
import pymc as pm
from scipy.optimize import dual_annealing
from sklearn.metrics import r2_score

from .Algorithms import ABC, Optuna, Annealing, MCMC
from ..models import BRModel

optuna.logging.set_verbosity(optuna.logging.ERROR)


class Calibration:

    def __init__(
        self,
        init_infectious: list[int],
        model: BRModel,
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
        self.init_infectious = init_infectious
        self.model = model
        self.data = data


    def abc_calibration(self, sample=100, epsilon=3000, with_rho=False):

        return ABC.calibrate(
            rho=self.rho, 
            model=self.model, 
            init_infectious=self.init_infectious, 
            data=self.data,
            sample=sample, 
            epsilon=epsilon, 
            with_rho=with_rho
        )


    def optuna_calibration(self, n_trials=1000):

        return Optuna.calibrate(
            rho=self.rho, 
            model=self.model, 
            init_infectious=self.init_infectious, 
            data=self.data,
            n_trials=n_trials
        )


    def annealing_calibration(self):

        return Annealing.calibrate(
            rho=self.rho, 
            model=self.model, 
            init_infectious=self.init_infectious, 
            data=self.data,
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
            sample=sample,
            epsilon=epsilon,
            with_rho=with_rho,  # [50_000, 500_000] - если True
            with_initi=with_initi,  # [1, 1_000] - если True
            tune=tune,
            draws=draws,
            chains=chains,
        )
