import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from model_complex import Calibration, EpidData, FactoryBRModel, ModelParams


def interval_estimation_plot(
    st_time, end_time, path, city, method, type, save_path, epsilon
):
    epid_data = EpidData(city=city, path=path, start_time=st_time, end_time=end_time)

    epid_data.get_wave_data(type=type)
    data = epid_data.get_data()
    model_params = ModelParams(
        alpha=[0],
        beta=[0],
        population_size=epid_data.get_rho() // 10,
        initial_infectious=[100],
    )

    if type == "age":
        model_params.initial_infectious = [100, 100]
        model = FactoryBRModel.age_group()
        label_alpha = {0: "0-14 years", 1: "15+ years"}

    else:
        model = FactoryBRModel.total()
        label_alpha = {0: "total"}

    calibration = Calibration(model, data, model_params)

    if method == "abc":
        calibration.abc_calibration(epsilon=epsilon)
    else:
        calibration.mcmc_calibration(epsilon=epsilon)

    alpha = []
    beta = []

    for params in model.get_ci_params():
        alpha.append(params.alpha)
        beta.append(params.beta)

    alpha, beta = np.array(alpha), np.array(beta)

    if type == "age":
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        for group in range(len(alpha[0])):
            sns.histplot(alpha[:, group], ax=axes[group], kde=True)
            axes[group].set_title(f"{label_alpha[group]} alpha")

    else:
        fig, axes = plt.subplots(1, 1, figsize=(5, 5))

        sns.histplot(alpha[:, 0], ax=axes, kde=True)
        axes.set_title(f"{label_alpha[0]} alpha")

    plt.savefig(
        save_path + f"IE_alpha_{city}_{method}_{type}_{st_time}_{end_time}.png", dpi=600
    )
    plt.savefig(
        save_path + f"IE_alpha_{city}_{method}_{type}_{st_time}_{end_time}.pdf", dpi=600
    )
    plt.clf()

    if type == "age":
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        for group in range(len(beta[0])):
            sns.histplot(beta[:, group], ax=axes[group // 2][group % 2], kde=True)

    else:
        fig, axes = plt.subplots(1, 1, figsize=(5, 5))

        sns.histplot(alpha[:, 0], ax=axes, kde=True)

    fig.suptitle("beta interval estimation")

    plt.savefig(
        save_path + f"IE_beta_{city}_{method}_{type}_{st_time}_{end_time}.png", dpi=600
    )
    plt.savefig(
        save_path + f"IE_beta_{city}_{method}_{type}_{st_time}_{end_time}.pdf", dpi=600
    )
    plt.clf()
