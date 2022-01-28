"""
This module generates the diagrams of our paper
"""

from experiment1 import *
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from multiprocessing import Pool
from scipy.optimize import curve_fit
import warnings


def func(x, a, b, c):
    """
    The general exponential function to fit on the experimental data

    :param float a: coefficient of exponential term
    :param float b: coefficient of the exponent
    :param float c: constant term
    :return: a*exp(b*x) + c
    :rtype: float
    """

    return a * np.exp(-b * x) + c

def generate_digrams_data(number_of_faults=1,\
                          max_number_of_queries=1000,\
                          number_of_experiment_per_query=10):
    """
    Generate the required data for our diagrams

    :param int number_of_faults: number of faults
    :param int max_number_of_queries: max number of queries specifying the maximum point on the x axis of our digrams (1, max_number_of_queries)
    :param int number_of_experiment_per_query: number of random experiments for each fixed number of queries
    """

    remained_deltas = []
    for nq in range(1, max_number_of_queries):
        mean_output, total_mean = experiment1(number_of_experiments=number_of_experiment_per_query,\
                                              number_of_faults=number_of_faults,\
                                              number_of_queries_in_each_exper=nq)
        remained_deltas.append(total_mean)
    return remained_deltas


def read_or_gen_data():
    """
    Generate (or read) the required data to plot a figure 
    for number of non-observed values with respect to the number 
    of available ciphertexts
    """

    lam = 1
    m = 2**8 - lam
    expected_number_of_queries = int(np.ceil((m*harmonic_number(m))))
    max_number_of_queries = expected_number_of_queries + 800
    #################################################################
    number_of_faults = range(1, 17)
    if not os.path.exists("candidates"):        
        number_of_experiment_per_query = 10
        with Pool(16) as pool:
            arguments = [(nf, max_number_of_queries, number_of_experiment_per_query) for nf in number_of_faults]
            candidates = pool.starmap(generate_digrams_data, arguments)
        with open('candidates', 'wb') as f:
            pickle.dump(candidates, f)
    else:
        with open('candidates', 'rb') as f:
            candidates = pickle.load(f)
    return candidates
        
def plot_diagram1():
    """
    Plot the number of non-observed values with 
    respect to the number of available ciphertexts - overview
    """

    candidates = read_or_gen_data()

    lam = 1
    m = 2**8 - lam
    expected_number_of_queries = int(np.ceil((m*harmonic_number(m))))
    max_number_of_queries = expected_number_of_queries + 800

    x_start_point = 1
    x_end_point = max_number_of_queries
    x_data = range(x_start_point, x_end_point)

    y_start_point = 0
    y_end_point = 256

    cmap = plt.get_cmap('hsv')
    colors = [cmap(i) for i in np.linspace(0, 1, 17)]
    for i in range(16):
        y_data = candidates[i][x_start_point - 1:]
        plt.plot(x_data, y_data,\
                color=colors[i], label='$\lambda = %d$' % (i + 1), linewidth=0.6)
        m = 2**8 - (i + 1)
        expect_number_of_queries = np.ceil((m*harmonic_number(m)))
        plt.plot([expect_number_of_queries]*2, [y_start_point, y_end_point],\
                    '--', color=colors[i], label='', linewidth=0.6)

    plt.legend(fontsize='xx-small', ncol=1, loc='best')

    x_tick_step = 215
    y_tick_step = 16
    plt.xticks(list(range(0, max_number_of_queries, x_tick_step)))
    plt.yticks([1] + list(range(16, 260, y_tick_step)))
    plt.grid(True)
    plt.xlabel('$N$: Number of known ciphertexts')
    plt.ylabel('Number of non-observed values')
    folder_name = "Figures"
    file_name = "overview_diagram_of_non_observed_values.svg"
    file_dir = os.path.join(folder_name, file_name)   
    plt.savefig(file_dir, format='svg', dpi=1200)
    return plt

def plot_diagram2():
    """
    Plot the number of non-observed values with respect to 
    the number of available ciphertexts - close up'
    """
    candidates = read_or_gen_data()

    lam = 1
    m = 2**8 - lam
    expected_number_of_queries = int(np.ceil((m*harmonic_number(m))))
    max_number_of_queries = expected_number_of_queries + 800

    x_start_point = 1
    x_end_point = max_number_of_queries
    x_data = range(x_start_point, x_end_point)

    y_start_point = 0
    y_end_point = 256

    cmap = plt.get_cmap('hsv')
    colors = [cmap(i) for i in np.linspace(0, 1, 17)]
    for i in range(16):
        y_data = candidates[i][x_start_point - 1:]
        plt.plot(x_data, y_data,\
                color=colors[i], label='$\lambda = %d$' % (i + 1), linewidth=0.6)
        m = 2**8 - (i + 1)
        expect_number_of_queries = np.ceil((m*harmonic_number(m)))
        plt.plot([expect_number_of_queries]*2, [y_start_point, y_end_point],\
                    '--', color=colors[i], label='', linewidth=0.6)

    plt.legend(fontsize='xx-small', ncol=2, loc='best')

    x_tick_step = 150
    y_tick_step = 1
    plt.xticks(list(range(700, 2000, x_tick_step)))
    plt.yticks(list(range(1, 20, y_tick_step)))
    plt.xlim(600, 2000)
    plt.ylim(0, 20)
    plt.grid(True)
    plt.xlabel('$N$: Number of known ciphertexts')
    plt.ylabel('Number of non-observed values')
    folder_name = "Figures"
    file_name = "close_up_diagram_of_non_observed_values.svg"
    file_dir = os.path.join(folder_name, file_name)
    plt.savefig(file_dir, format='svg', dpi=1200)  
    return plt

def fit_to_exp1():
    """
    Fit an exponential curve to derived data - overview
    """
    candidates = read_or_gen_data()

    lam = 1
    m = 2**8 - lam
    expected_number_of_queries = int(np.ceil((m*harmonic_number(m))))
    max_number_of_queries = expected_number_of_queries + 800

    warnings.filterwarnings('ignore')
    x_start_point = 1
    x_end_point = max_number_of_queries
    x_data = np.arange(x_start_point, x_end_point)

    y_start_point = 0
    y_end_point = 256

    cmap = plt.get_cmap('hsv')
    colors = [cmap(i) for i in np.linspace(0, 1, 17)]
    for i in range(16):
        y_data = candidates[i][x_start_point - 1:]
        # Fit a curve to data
        popt, pcov = curve_fit(func, x_data, y_data)
        plt.plot(x_data, func(x_data, *popt),\
                color=colors[i],
                label='$\lambda = %d, a=%5.3f, b=%5.3f, c=%5.3f$' % (i + 1, *popt), linewidth=0.6)
        m = 2**8 - (i + 1)
        expect_number_of_queries = np.ceil((m*harmonic_number(m)))
        plt.plot([expect_number_of_queries]*2, [y_start_point, y_end_point],\
                    '--', color=colors[i], label='', linewidth=0.6)

    plt.legend(fontsize='xx-small', ncol=2, loc='best')
    plt.title("$y = a \cdot e^{-b \cdot N} + c$")

    x_tick_step = 215
    y_tick_step = 16
    plt.xticks(list(range(0, max_number_of_queries, x_tick_step)))
    plt.yticks([1] + list(range(16, 260, y_tick_step)))
    plt.grid(True)
    plt.xlabel('$N$: Number of known ciphertexts')
    plt.ylabel("$y$")
    folder_name = "Figures"
    file_name = "overview_fit_on_non_observed_values.svg"
    file_dir = os.path.join(folder_name, file_name)
    plt.savefig(file_dir, format='svg', dpi=1200)
    return plt

def fit_to_exp2():
    """
    Fit an exponential curve to derived data - close up
    """
    
    candidates = read_or_gen_data()

    lam = 1
    m = 2**8 - lam
    expected_number_of_queries = int(np.ceil((m*harmonic_number(m))))
    max_number_of_queries = expected_number_of_queries + 800

    x_start_point = 1
    x_end_point = max_number_of_queries
    x_data = np.arange(x_start_point, x_end_point)

    y_start_point = 0
    y_end_point = 260

    cmap = plt.get_cmap('hsv')
    colors = [cmap(i) for i in np.linspace(0, 1, 17)]
    for i in range(16):
        y_data = candidates[i][x_start_point - 1:]
        # Fit a curve to data
        popt, pcov = curve_fit(func, x_data, y_data)
        plt.plot(x_data, func(x_data, *popt),\
                color=colors[i],
                label='$\lambda = %d, a=%5.3f, b=%5.3f, c=%5.3f$' % (i + 1, *popt), linewidth=0.9)
        # Draw a vertical line to show the expected number of queries based on our estimation
        m = 2**8 - (i + 1)
        expect_number_of_queries = np.ceil((m*harmonic_number(m)))
        plt.plot([expect_number_of_queries]*2, [y_start_point, y_end_point],\
                    '--', color=colors[i], label='', linewidth=0.6)

    plt.legend(fontsize='xx-small', ncol=1, loc='best')
    plt.title("$y = a \cdot e^{-b \cdot N} + c$")

    x_tick_step = 150
    y_tick_step = 1
    plt.xticks(list(range(700, 2000, x_tick_step)))
    plt.yticks([1] + list(range(1, 20, y_tick_step)))
    plt.xlim(600, 2000)
    plt.ylim(0, 20)
    plt.grid(True)
    plt.xlabel('$N$: Number of known ciphertexts')
    plt.ylabel('$y$')
    folder_name = "Figures"
    file_name = "close_up_fit_on_non_obsereved_values.svg"
    file_dir = os.path.join(folder_name, file_name)
    plt.savefig(file_dir, format='svg', dpi=1200)
    return plt

if __name__ == "__main__":    
    plot_diagram1()
    plot_diagram2()
    fit_to_exp1()
    fit_to_exp2()