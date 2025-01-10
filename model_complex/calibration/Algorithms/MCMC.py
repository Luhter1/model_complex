import numpy as np
import pymc as pm

from ...models import BRModel
from ...utils import ModelParams


class MCMC:

    @classmethod
    def calibrate(
        self, 
        rho: int, 
        model: BRModel, 
        init_infectious: list[int], 
        data: np.array,
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

        alpha_len, beta_len = model.params()

        simulate_pars = ModelParams(
            alpha=[0],
            beta=[0],
            population_size=0,
            initial_infectious=[0]
        )


        def simulation_func(rng, alpha, beta, rho, init_infectious, size=None):
            simulate_pars.alpha = alpha
            simulate_pars.beta = beta
            simulate_pars.population_size = rho
            simulate_pars.initial_infectious = init_infectious

            model.simulate(
                pars=simulate_pars,
                modeling_duration=len(data) // alpha_len,
            )
            return model.newly_infected

        with pm.Model() as pm_model:
            alpha = pm.Uniform(name="alpha", lower=0, upper=1, shape=(alpha_len,))
            beta = pm.Uniform(name="beta", lower=0, upper=1, shape=(beta_len,))

            if with_rho:
                rho = pm.Uniform(name="rho", lower=with_rho[0], upper=with_rho[1])

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
                rho,
                list(init_infectious) + [0] * (beta_len - len(init_infectious)),
                epsilon=epsilon,
                observed=data,
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
            np.random.choice(posterior["alpha"][i], size=sample)
            for i in range(alpha_len)
        ]
        beta = [
            np.random.choice(posterior["beta"][i], size=sample) for i in range(beta_len)
        ]

        # запускаем, чтобю в модели были результаты с лучшими параметрами
        simulate_pars.alpha = [a.mean() for a in alpha]
        simulate_pars.beta = [b.mean() for b in beta]
        simulate_pars.population_size = rho
        simulate_pars.initial_infectious = init_infectious

        model.simulate(
            pars=simulate_pars,
            modeling_duration=len(data) // alpha_len,
        )

        return alpha, beta