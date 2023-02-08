import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

def fit_and_plot(
        model,
        x: np.array,
        y: np.array,
        x_uncertainty: np.array,
        y_uncertainty: np.array, 
        num_par: int, 
        given_values: list, 
        xlabel: str, 
        ylabel: str,
        title: str,
        fitting_equation: str,
        residual_title = None,
        save_dir = None, 
        x_logscale = False, 
        y_logscale = False,
    ):
    """_summary_

    Args:
        model (class): Your model's forward method should take in (x, *params) and return y
                       Your model's backward method should take in (x, *params) and return y'
        x (np.array): x data point
        y (np.array): y data point
        y_uncertainty (np.array): the uncertainty associated with y
        num_par (int): the number of free parameters in your model
        given_values (list): should be either None or list of length num_par
        xlabel (str): name of the x axis
        ylabel (str): name of the y axis
        title (str): the title of the plot
        fitting equation (str): the fitting equation associated with the regression
        residual_title (str, optional): the title of the residual plot 
        save_dir (str, optional): directory to save your plot
        x_logscale (bool, optional): use log scale for x axis. Defaults to False.
        y_logscale (bool, optional): use log scale for y axis. Defaults to False.
    Return:
        pars (np.array): best fit parameters
        cov (np.array): covariance of fitting
    """
    init_values = given_values if given_values else [1]*num_par

    pars, cov = opt.curve_fit(model.forward, x, y, p0=init_values, sigma = y_uncertainty, absolute_sigma = True)
    for i in range(2):
        equi_uncertainty = np.sqrt(np.square(model.backward(x, *pars) * x_uncertainty) + np.square(y_uncertainty))
        pars, cov = opt.curve_fit(model.forward, x, y, p0=pars, sigma = equi_uncertainty, absolute_sigma = True)
        std_errs = np.sqrt(np.diag(cov))

    print(np.transpose([pars, std_errs]))
    
    if residual_title != None:
        fig1, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 8), dpi= 100, facecolor='w', edgecolor='k')
    else:
        fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 8), dpi= 100, facecolor='w', edgecolor='k')
    plt.rcParams.update({'font.size': '16'})
    
    width = np.max(x) - np.min(x)
    pred_x = np.linspace(np.min(x) - 0.05*width, np.max(x)+0.05*width, 1000)
    pred_y = model.forward(pred_x, *pars.tolist())

    if equi_uncertainty is not None:
        Chi_squared = np.sum((model.forward(x, *pars.tolist()) - y)**2/(equi_uncertainty**2+1e-12))/(len(y)-num_par) 

    if given_values != None:
        theory_y = model.forward(pred_x, *given_values)
    # Data and fit
    line1 = ax1.scatter(x, y, s=1)
    _ = ax1.errorbar(x, y, yerr=y_uncertainty, xerr=x_uncertainty, fmt='none')
    line2, = ax1.plot(pred_x , pred_y, color='orange', ms = 1)
    # line4, = ax1.plot([], [], ' ')
    if given_values != None:
        line3, = ax1.plot(pred_x , theory_y, color='red', ms = 1)
        ax1.legend(
            [line1, line2, line3],
            ["data points", "best-fit curve", "theoretical prediction"],
            title = "Model: " + fitting_equation + "\n" + r"$\chi^2$ per ddof" + f": {Chi_squared:.2f}")
    else:
        ax1.legend(
            [line1, line2],
            ["data points", "best-fit curve"],
            title = "Model: " + fitting_equation + "\n" + r"$\chi^2$ per ddof"+f":  {Chi_squared:.2f}"
            )

    
    if x_logscale:
        ax1.set_xscale('log')
    if y_logscale:
        ax1.set_yscale('log')

    # Axes label and title
    ax1.set_title(title)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)

    if residual_title != None:
        line5 = ax2.scatter(x, (model.forward(x, *pars.tolist()) - y)/(equi_uncertainty), s=20)
        ax2.axhline(y=0., color='r', linestyle='-')
        ax2.legend([line5], [r"$\frac{y_i - f(x_i)}{\sqrt{(f'(x_i)\sigma_x)^2 + \sigma_y^2}}$"])
        ax2.set_ylabel("normalized equivalent residual")
        ax2.set_title(residual_title)
        ax2.set_xlabel(xlabel)
        
    plt.show()
    if save_dir:
        plt.savefig(save_dir)

    return pars, cov

def montecarlo_estimator(f, xs, sigma, rollout):
    stats = []
    for i in range(rollout):
        stats.append(
            np.asarray(
                f(*[x + np.random.normal(0, s, size = x.shape) for x, s in zip(xs, sigma)])
            ).reshape(1, -1)
        )
    stats = np.concatenate(stats, axis = 0)
    std = stats.std(0)
    return std