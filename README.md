
<div align="center">

# Physics 5CL Toolkit

</div>

##Introduction
This module is developed to speed up the process of data analysis of Berkeley Physics 5 series lab. To import the utilities, go into the utils.py file and directly import the things needed.
##Explanation
The module consists of two components, a Monte Carlo uncertainty estimator and a parameter regressor. 
- *Monte Carlo Uncertainty Estimator*: to estimate the uncertainty of a complicated derived quantity, it is possible to calculate tedious partial derivatives. However, we can adopt Monte Carlo methods to estimate the uncertainty. Given $y=f(x_1, x_2, \dots, x_n)$ where each of the $x_i$ is independent measurement, we are interested in the variance $\sigma_y =\sqrt{\text{Var}_{x_i~\mathcal N(x_i, \sigma_i)}\left[f(x_1, x_2, ..., x_n)\right]}$. Assuming $f(x_1, x_2, \dots, x_n)$ is smooth enough and $\sigma_i$ is small enough, we can derive an explicit error propagation formula $\sigma_y=\sqrt{\sum_{i=1}^n\left(\frac{\partial f}{\partial x_i}\sigma_i\right)^2}$. However, as a more direct method, we can use Monte Carlo estimation of the quantity, namely, $\sigma_y=\sqrt{\sum_{x_i\sim\mathcal N(\bar x_i, \sigma_i)}^N\frac{\left(f(x_1, x_2, \dots, x_n) - f(\bar x_1, \bar x_2, \dots, \bar x_n)\right)^2}{N}}$.  The method is implemented by ```montecarlo_estimator```.

- *Parameter Regressor*: Given a functional relation $y=f(x;\theta)$ between $x$ and $y$, a set of data points $(x_i, y_i)$, and uncertainties attached to each measurements/derived quantities $(x, y)$, we find the best-fit parameter with the ```scipy.optimize.curve_fit estimator```. However it is worth noting that the estimator only takes uncertainties of $y$. To address this problem, we iterate as follows: first, we fit the curve with $y$ uncertainties and get a set of best-fit parameters fit. The best-fit parameters are used to derive $f'(x; \theta_{fit})$. The equivalent uncertainties are then updated as $\sqrt{\sigma_{y, i}^2+\left(f'(x_i; \theta_{fit})\sigma_{x, i}\right)^2}$. In principle this process can be iterated many times, but empirically, three times is enough for convergence. Then we plot the best-fit curve and the datapoints out, as well as the normalized residual of each datapoint. The parameter regressor and plot maker are implemented by ```fit_and_plot```.
## Usage
```
fit_and_plot
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
```
```
montecarlo_estimator
    Args:
        f (Callable): takes in a sequence of np.array of the same shape and return y
        xs (Sequence): an Sequence containing input np.array
        sigma (Sequence): an Sequence containing input uncertainty np.array
        rollout (int): how many times of random trials wanted

    Returns:
        std (np.array): uncertainties of each y
```