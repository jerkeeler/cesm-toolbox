import seaborn as sns
import matplotlib.pyplot as plt


def default_styles():
    sns.set_context("notebook")
    plt.rcParams["axes.grid.which"] = "major"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["axes.labelsize"] = 15
    # plt.rcParams["axes.linewidth"] = 1.5
    # plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["axes.titlepad"] = 10
    plt.rcParams["axes.titlesize"] = 20
    # plt.rcParams["figure.figsize"] = (10, 8)
    plt.rcParams["font.size"] = 15
    plt.rcParams["grid.alpha"] = 0.5
    plt.rcParams["grid.color"] = "#666666"
    plt.rcParams["grid.linestyle"] = "-"
    plt.rcParams["legend.fontsize"] = 13
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["xtick.major.size"] = 5
    plt.rcParams["xtick.major.width"] = 1.5
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["ytick.labelsize"] = 16
    plt.rcParams["ytick.major.size"] = 5
    plt.rcParams["ytick.major.width"] = 1.5
    # plt.rc("mathtext", fontset="stix")
    # plt.rc("font", family="STIXGeneral")
    print("Default plot styles have been set!")
