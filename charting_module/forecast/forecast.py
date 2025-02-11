from datetime import datetime

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from model_complex import Calibration, EpidData, FactoryBRModel, Forecast, ModelParams


def forecast_plot(
    st_time,
    end_time,
    forecast_duration,
    path,
    city,
    method,
    type,
    save_path="./",
    epsilon=3000,
):

    epid_data = EpidData(city=city, path=path, start_time=st_time, end_time=end_time)

    epid_data.get_wave_data(type=type)
    data = epid_data.get_data()
    dur = epid_data.get_duration()
    plot_data = epid_data.prepare_for_plot()
    model_params = ModelParams(
        alpha=[0],
        beta=[0],
        population_size=epid_data.get_rho() // 10,
        initial_infectious=[100],
    )

    if type == "age":
        model_params.initial_infectious = [100, 100]
        model = FactoryBRModel.age_group()
        label = {0: "0-14 years", 1: "15+ years"}

    else:
        model = FactoryBRModel.total()
        label = {0: "total"}
    color = {0: "blue", 1: "orange"}

    if data.attrs["time_step"] == "week":
        func_to_get_newly_data = model.get_weekly_newly_infected_by_group
    else:
        func_to_get_newly_data = model.get_daily_newly_infected_by_group

    calibration = Calibration(model, data, model_params)

    if method == "abc":
        calibration.abc_calibration(epsilon=epsilon)
    else:
        calibration.mcmc_calibration(epsilon=epsilon)

    end_date = datetime.strptime(end_time, "%d-%m-%Y") + forecast_duration
    end_prog = end_date.strftime("%d-%m-%Y")

    forecast_epid_data = EpidData(
        city=city, path=path, start_time=st_time, end_time=end_prog
    )
    forecast_epid_data.get_wave_data(type=type)
    forecast_plot_data = forecast_epid_data.prepare_for_plot()

    forecast_result = Forecast.forecast(model, data, forecast_duration)

    model.simulate(params=model.get_best_params(), modeling_duration=dur)

    result = func_to_get_newly_data()

    for i in range(len(result)):
        r2 = round(r2_score(plot_data[:, i], result[i]), 2)
        plt.plot(forecast_result[i, :, 1], color=color[i], alpha=0.3)
        plt.plot(
            result[i],
            label=f"{label[i]}, $R^2_{i}$: {r2}",
            color=color[i],
        )
        plt.plot(forecast_plot_data[:, i], "--o", color=color[i], alpha=0.5)
        plt.plot(plot_data[:, i], "--o", color=color[i])
        plt.fill_between(
            [j for j in range(len(forecast_result[0, :, 0]))],
            forecast_result[i, :, 0],
            forecast_result[i, :, 2],
            color=color[i],
            alpha=0.1,
        )
        plt.plot(forecast_result[i, :, 0], color=color[i])
        plt.plot(forecast_result[i, :, 2], color=color[i])

    plt.title(f"{method.upper()}, {type.capitalize()}")
    plt.legend()

    plt.savefig(save_path + f"F_{city}_{method}_{type}_{st_time}_{end_time}.png", dpi=600)
    plt.savefig(save_path + f"F_{city}_{method}_{type}_{st_time}_{end_time}.pdf", dpi=600)
    plt.clf()
