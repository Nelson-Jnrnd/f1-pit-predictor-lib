from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

def plot_remove_duplicate_legends(ax, **kwargs):
    """
    Remove duplicate legend labels

    This is a hack to remove duplicate legend labels when plotting with
    pandas. It is not a general solution, but works for many cases.

    Parameters
    ----------
    ax : matplotlib Axes
        The axes to work on
    **kwargs : dict
        Keyword arguments to be passed to ax.legend

    Returns
    -------
    ax : matplotlib Axes
        The axes with duplicate legend labels removed
    """
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), **kwargs)
    return ax

def plot_race_predictions(x, y, proba, ax, color='cyan', marker='X', label='Predicted pitstop', title='Predicted pitstops'):
    """
    Plot pitstop predictions for series of laps

    Parameters
    ----------
    x : pandas DataFrame
        A dataframe containing the following columns:
        - LapNumber
        - TotalLaps
    y : array-like
        The ground truth pitstop labels
    proba : array-like
        The predicted probability of pitting
    ax : matplotlib Axes
        The axes to plot on
    color : str, default 'cyan'
        The color of the plotted points
    marker : str, default 'X'
        The marker of the plotted points
    label : str, default 'Predicted pitstop'
        The label of the plotted points
    title : str, default 'Predicted pitstops'
        The title of the plot

    Returns
    -------
    ax : matplotlib Axes
        The axes with the plotted points
    """
    ax.hlines(y=0.5, xmin=x['LapNumber'].min(), xmax=x['TotalLaps'].max(), linewidth=1, colors='black', linestyles='dotted', zorder=0)
    indexes = np.where(y == 1)
    for idx in indexes:
        ax.vlines(x=idx+2, ymin=0, ymax=1, linewidth=1, colors='red', linestyles='dashed', zorder=0)
    sns.scatterplot(x=x['LapNumber'], y=proba, marker=marker, legend=False, ax=ax, s=50, label=label, color=color)
    sns.lineplot(x=x['LapNumber'], y=proba, linewidth=1, linestyle='dashed', zorder=0, ax=ax, color=color)
    ax.set_xlabel('Lap Number')
    ax.set_xticks(np.arange(0, x['TotalLaps'].max(), 5))
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability of pitting')
    ax.set_xlim(x['LapNumber'].min(), x['TotalLaps'].max())    

    ax.set_title(title)
    return plot_remove_duplicate_legends(ax, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0.)

def plot_stoppages(laps, ax):
    """
    Plot stoppages on a lap chart

    Parameters
    ----------
    laps : pandas DataFrame
        A dataframe containing the following columns:
        - LapNumber
        - Red
        - SC
        - VSC
        - SC_ending
        - Yellow
    ax : matplotlib Axes
        The axes to plot on

    Returns
    -------
    ax : matplotlib Axes
        The axes with the plotted stoppages
    """
    for idx, lap in laps.iterrows():
        if lap['Red']:
            ax.axvspan(idx, idx+1, facecolor='red', alpha=0.3, zorder=0, label='Red Flag')
        elif lap['SC']:
            ax.axvspan(idx, idx+1, facecolor='yellow', alpha=0.3, zorder=0, label='Safety Car', hatch='\\', edgecolor='black')
        elif lap['VSC']:
            ax.axvspan(idx, idx+1, facecolor='yellow', alpha=0.3, zorder=0, label='Virtual Safety Car', hatch='-', edgecolor='black')
        elif lap['SC_ending']:
            ax.axvspan(idx, idx+1, facecolor='yellow', alpha=0.3, zorder=0, label='Safety Car Ending', hatch='\\', edgecolor='green')
        elif lap['Yellow']:
            ax.axvspan(idx, idx+1, facecolor='yellow', alpha=0.3, zorder=0, label='Yellow Flag')

    return plot_remove_duplicate_legends(ax, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0.)

def plot_tyres(laps, ax):
    """
    Plot tyre compounds on a lap chart

    Parameters
    ----------
    laps : pandas DataFrame
        A dataframe containing the following columns:
        - LapNumber
        - Compound_HARD
        - Compound_MEDIUM
        - Compound_SOFT
    ax : matplotlib Axes
        The axes to plot on

    Returns
    -------
    ax : matplotlib Axes
        The axes with the plotted tyre compounds
    """
    for idx, lap in laps.iterrows():
        if lap['Compound_SOFT']:
            ax.axvspan(idx-1, idx, facecolor='red', alpha=1, zorder=-1, label='Soft', ymin=0.99)
            #ax.axhline(y=0.99, xmin=idx, xmax=idx+1, linewidth=1, color='red', zorder=0)
        elif lap['Compound_MEDIUM']:
            ax.axvspan(idx-1, idx, facecolor='yellow', alpha=1, zorder=-1, label='Medium', ymin=0.99)
            #ax.axhline(y=0.99, xmin=idx, xmax=idx+1, linewidth=1, color='yellow', zorder=0)
        elif lap['Compound_HARD']:
            ax.axvspan(idx-1, idx, facecolor='grey', alpha=1, zorder=-1, label='Hard', ymin=0.99)
            #ax.axhline(y=0.99, xmin=idx, xmax=idx+1, linewidth=1, color='grey', zorder=0)
        elif lap['Compound_INTERMEDIATE']:
            ax.axvspan(idx-1, idx, facecolor='green', alpha=1, zorder=-1, label='Intermediate', ymin=0.9)
            #ax.axhline(y=0.99, xmin=idx, xmax=idx+1, linewidth=1, color='green', zorder=0)
        elif lap['Compound_WET']:
            ax.axvspan(idx-1, idx, facecolor='blue', alpha=1, zorder=-1, label='Wet', ymin=0.99)
            #ax.axhline(y=0.99, xmin=idx, xmax=idx+1, linewidth=1, color='blue', zorder=0)

    return plot_remove_duplicate_legends(ax, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0.)