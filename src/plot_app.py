#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot functions for visualisation
"""
from pandas.core.frame import DataFrame
import plotly.graph_objects as go


def plot_steps(exp_df, data_column, layout=None):
    """
    Plot BO convergence trace

    Parameters
    ----------
    exp_df: DataFrame
        Table of all experimental data
    data_column: str
        The name of y value column in exp_df
    layout: dict or None
        Plot layout

    Returns
    -------
    fig: go.Figure
    """
    def get_maximum_value(df, values_column):
        maximum_list = []
        max_value = 0
        for i in range(max(df.step) + 1):
            step_max = df[df.step == i].loc[:, values_column].values.max()
            if step_max > max_value:
                max_value = step_max
            maximum_list.append(max_value)
        return maximum_list

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=exp_df.loc[:, 'step'],
        y=exp_df.loc[:, 'yield'] * 100,
        text=exp_df.loc[:, 'name'],
        name='BO experiment',
        mode='markers',
        marker=dict(
            symbol='circle',
            color='blue'
        )
    ))
    fig.add_trace(go.Scatter(
        x=[i for i in range(max(exp_df.step) + 1)],
        y=get_maximum_value(exp_df, values_column=data_column) * 100,
        name='Maximum yield',
        mode='lines',
        marker=dict(
            symbol='circle',
            color='blue'
        )
    ))
    if layout is None:
        fig.update_layout(
            height=600,
            width=800,
            title='BO search vs random',
            yaxis_title='Yield/%',
            xaxis_title='Number of steps'
        )
    else:
        fig.update_layout(layout)
    return fig


def plot_generator(df, exp_id, suggested_id, title):
    """
    Generate a 2D plot for BO steps overview

    Parameters
    ----------
    df: DataFrame
        The origin DataFrame
    exp_id: list
        The experimental samples ID in the DataFrame
    suggested_id: list
        The suggested samples ID in the DataFrame
    title: str
        The title of this plot
    Returns
    -------
    go.Figure
    """
    # Unselected points
    design_id = df.index.tolist()
    for i in exp_id:
        design_id.remove(i)
    # Samples are divided into 3 traces
    designed_mol = go.Scatter(
        x=df.loc[design_id, 'pos_0'],
        y=df.loc[design_id, 'pos_1'],
        text=df.loc[design_id, 'text'],
        mode='markers',
        name='designed_mol',
        marker=dict(
            size=df.loc[design_id, 'mean'] * 20 + 9,
            symbol='circle',
            line=dict(
                color='#8c8c8c'
            ),
            color=df.loc[design_id, 'std'],
            colorscale='RdBu',
            colorbar=dict(
                title='Standard deviation',
                x=1.02, y=0.42, len=0.85, yanchor='middle'
            ),
            cmin=0,
            cmax=0.2,
            reversescale=True,
            showscale=True
        )
    )
    exp_mol = go.Scatter(
        x=df.loc[exp_id, 'pos_0'],
        y=df.loc[exp_id, 'pos_1'],
        text=df.loc[exp_id, 'text'],
        opacity=1,
        mode='markers',
        name='experiment_mol',
        marker=dict(
            size=df.loc[exp_id, 'mean'] * 20 + 9,
            line=dict(
                color='#8c8c8c'
            ),
            symbol='square',
            color='#ff9500',
            showscale=False
        )
    )
    suggested_mol = go.Scatter(
        x=df.loc[suggested_id, 'pos_0'],
        y=df.loc[suggested_id, 'pos_1'],
        text=df.loc[suggested_id, 'text'],
        opacity=1,
        mode='markers',
        name='suggested_mol',
        marker=dict(
            size=df.loc[suggested_id, 'mean'] * 20 + 9,
            line=dict(
                color='#8c8c8c'
            ),
            symbol='x',
            color='#5b00ed',
            showscale=False
        )
    )
    # Hidden all axis
    axis_template = dict(
        showgrid=False,
        zeroline=False,
        showline=False,
        showticklabels=False,
    )
    layout = go.Layout(
        height=900,
        width=1200,
        title=dict(text=title, x=0.5),
        hovermode='closest',
        xaxis=axis_template,
        yaxis=axis_template,
        showlegend=True
    )
    # The plot function
    fig = go.Figure(data=[designed_mol, exp_mol, suggested_mol], layout=layout)
    fig.update_traces()
    # chart_studio.plotly.iplot(fig, filename=name)
    print('Done.')
    return fig
