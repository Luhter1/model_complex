from model_complex import (
    Calibration, 
    EpidData, 
    FactoryBRModel
)
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def calibration_plot(st_time, end_time, path, city, method, type, save_path='./', epsilon=3000):
    epid_data = EpidData(city=city, path=path, 
                start_time=st_time, end_time=end_time)
    
    epid_data.get_wave_data(regime=type)
    data = epid_data.get_data()
    rho = epid_data.get_rho()//10
    dur = epid_data.get_duration()
    plot_data = epid_data.prepare_for_plot()

    if type == 'age':
        init_infect = [100, 100]
        model = FactoryBRModel.age_group()
        label = {0: '0-14 years', 1: '15+ years'}

    else:
        init_infect = [100]
        model = FactoryBRModel.total()
        label = {0: 'total'}
    color = {0: 'blue', 1: 'orange'}


    calibration = Calibration(init_infect, model, data, rho)

    if method == 'annealing':
        calibration.annealing_calibration()
    elif method == 'abc':
        calibration.abc_calibration(epsilon=epsilon)
    elif method == 'mcmc':
        calibration.mcmc_calibration(epsilon=epsilon)
    else:
        calibration.optuna_calibration()


    for ci_par in model.get_ci_params():
        model.simulate(
            pars=ci_par,
            modeling_duration=dur
        )

        res = model.get_weekly_newly_infected_by_group()

        for i in range(len(res)):
            plt.plot(res[i], lw=0.3, alpha=0.5, color=color[i])

    model.simulate(
        pars=model.get_best_params(),
        modeling_duration=dur
    )


    res = model.get_weekly_newly_infected_by_group()

    for i in range(len(res)):
        plt.plot(
            res[i], 
            label=f'{label[i]}, $R^2_{i}$: {round(r2_score(plot_data[:, i], res[i]),2)}', 
            color=color[i]
        )
        plt.plot(plot_data[:, i], '--o', color=color[i])

    plt.title(f"{method.capitalize()}, {type.capitalize()}")
    plt.legend()

    plt.savefig(save_path + f'{city}_{method}_{type}_{st_time}_{end_time}.png', dpi=600)
    plt.savefig(save_path + f'{city}_{method}_{type}_{st_time}_{end_time}.pdf', dpi=600)

