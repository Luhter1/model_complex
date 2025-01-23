import numpy as np
import pymc as pm

from ...models import Model
from ...utils import ModelParams


class ABC:

    @classmethod
    def calibrate(
        self,
        rho: int,
        model: Model,
        init_infectious: list[int],
        data: np.array,
        sample: int = 100,
        epsilon: int = 3000,
    ):
        alpha_len, beta_len = model.params()
        duration = (len(data) // alpha_len) * 7

        simulate_pars = ModelParams(
            alpha=[0],
            beta=[0],
            population_size=rho,
            initial_infectious=init_infectious,
        )

        def simulation_func(rng, alpha, beta, size=None):

            simulate_pars.alpha = alpha
            simulate_pars.beta = beta

            model.simulate(
                pars=simulate_pars,
                modeling_duration=duration
            )
            return model.get_weekly_newly_infected()

        with pm.Model() as PMmodel:
            alpha = pm.Uniform(name="alpha", lower=0, upper=1, shape=(alpha_len,))
            beta = pm.Uniform(name="beta", lower=0, upper=1, shape=(beta_len,))

            sim = pm.Simulator(
                "sim",
                simulation_func,
                list(alpha) + [0] * (beta_len - alpha_len),
                beta,
                epsilon=epsilon,
                observed=data,
            )

            idata = pm.sample_smc(progressbar=False)

        posterior = idata.posterior.stack(samples=("draw", "chain"))

        alpha = np.array(
            [
                np.random.choice(posterior["alpha"][i], size=sample)
                for i in range(alpha_len)
            ]
        )
        beta = np.array(
            [
                np.random.choice(posterior["beta"][i], size=sample) 
                for i in range(beta_len)
            ]
        )

        ci_pars = []

        for i in range(sample):

            ci_par = ModelParams(
                alpha=alpha[:, i],
                beta=beta[:, i],
                population_size=rho,
                initial_infectious=init_infectious,
            )

            ci_pars.append(ci_par)

        model.set_ci_params(ci_pars)

        simulate_pars.alpha = [a.mean() for a in alpha]
        simulate_pars.beta = [b.mean() for b in beta]

        model.set_best_params(simulate_pars)
