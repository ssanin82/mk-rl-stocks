from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dash import Dash, html, dcc


def create_figure(prices, actions, portfolio_values, metrics):
    # Find buy and sell points
    buy_indices = [i for i, a in enumerate(actions) if a == 1]
    sell_indices = [i for i, a in enumerate(actions) if a == 2]
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"colspan": 2}, None],
            [{"colspan": 2}, None],
            [{"type": "indicator"}, {"type": "indicator"}]
        ],
        subplot_titles=("Stock Price with Buy/Sell Signals", "Portfolio Value Over Time"),
        vertical_spacing=0.12,
        row_heights=[0.4, 0.4, 0.2]
    )
    
    # Price chart
    fig.add_trace(
        go.Scatter(x=list(range(len(prices))), y=prices, name="Stock Price",
                  line=dict(color='#2196F3', width=2), fill='tozeroy',
                  fillcolor='rgba(33, 150, 243, 0.1)'),
        row=1, col=1
    )
    
    # Buy signals
    fig.add_trace(
        go.Scatter(x=buy_indices, y=[prices[i] for i in buy_indices],
                  mode='markers', name='Buy Signal',
                  marker=dict(color='#4CAF50', size=12, symbol='triangle-up')),
        row=1, col=1
    )
    
    # Sell signals
    fig.add_trace(
        go.Scatter(x=sell_indices, y=[prices[i] for i in sell_indices],
                  mode='markers', name='Sell Signal',
                  marker=dict(color='#f44336', size=12, symbol='triangle-down')),
        row=1, col=1
    )
    
    # Portfolio value chart
    fig.add_trace(
        go.Scatter(x=list(range(len(portfolio_values))), y=portfolio_values,
                  name="Portfolio Value", line=dict(color='#4CAF50', width=2),
                  fill='tozeroy', fillcolor='rgba(76, 175, 80, 0.1)'),
        row=2, col=1
    )
    
    # Initial capital line
    fig.add_trace(
        go.Scatter(x=list(range(len(portfolio_values))),
                  y=[metrics['initial_capital']] * len(portfolio_values),
                  name="Initial Capital", line=dict(color='#FF9800', width=2, dash='dash')),
        row=2, col=1
    )
    
    # Metrics indicators
    pnl_color = '#4CAF50' if metrics['total_pnl'] >= 0 else '#f44336'
    
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=metrics['final_capital'],
            title={"text": "Final Capital"},
            delta={'reference': metrics['initial_capital'], 'relative': False},
            number={'prefix': "$", 'valueformat': ".2f"},
            domain={'x': [0, 1], 'y': [0, 1]}
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=metrics['total_return'],
            title={"text": "Total Return"},
            number={'suffix': "%", 'valueformat': ".2f", 'font': {'color': pnl_color}},
            domain={'x': [0, 1], 'y': [0, 1]}
        ),
        row=3, col=2
    )
    
    # Update layout with dark theme
    fig.update_layout(
        template='plotly_dark',
        height=1000,
        showlegend=True,
        title={
            'text': '📈 Reinforcement Learning Trading Strategy Results',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#4CAF50'}
        },
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#2a2a2a',
        font={'color': '#e0e0e0'}
    )
    
    return fig


def create_dash_app(prices, actions, portfolio_values, metrics):
    app = Dash(__name__)
    pnl_color = '#4CAF50' if metrics['total_pnl'] >= 0 else '#f44336'
    app.layout = html.Div(
        style={'backgroundColor': '#1a1a1a', 'padding': '20px', 'fontFamily': 'Segoe UI, sans-serif'},
        children=[
            html.Div(
                style={'maxWidth': '1400px', 'margin': '0 auto'},
                children=[
                    # Metrics grid
                    html.Div(
                        style={
                            'background': '#2a2a2a',
                            'padding': '20px',
                            'borderRadius': '8px',
                            'margin': '20px 0'
                        },
                        children=[
                            html.H2(
                                'Performance Metrics',
                                style={'color': '#4CAF50', 'textAlign': 'center'}
                            ),
                            html.Div(
                                style={
                                    'display': 'grid',
                                    'gridTemplateColumns': 'repeat(3, 1fr)',
                                    'gap': '20px',
                                    'marginTop': '20px'
                                },
                                children=[
                                    html.Div([
                                        html.Div('Initial Capital', style={'color': '#888', 'fontSize': '14px', 'textAlign': 'center'}),
                                        html.Div(f"${metrics['initial_capital']:.2f}", style={'color': '#e0e0e0', 'fontSize': '24px', 'fontWeight': 'bold', 'textAlign': 'center'})
                                    ]),
                                    html.Div([
                                        html.Div('Final Capital', style={'color': '#888', 'fontSize': '14px', 'textAlign': 'center'}),
                                        html.Div(f"${metrics['final_capital']:.2f}", style={'color': pnl_color, 'fontSize': '24px', 'fontWeight': 'bold', 'textAlign': 'center'})
                                    ]),
                                    html.Div([
                                        html.Div('Total P&L', style={'color': '#888', 'fontSize': '14px', 'textAlign': 'center'}),
                                        html.Div(f"${metrics['total_pnl']:.2f}", style={'color': pnl_color, 'fontSize': '24px', 'fontWeight': 'bold', 'textAlign': 'center'})
                                    ]),
                                    html.Div([
                                        html.Div('Max Drawdown', style={'color': '#888', 'fontSize': '14px', 'textAlign': 'center'}),
                                        html.Div(f"{metrics['max_drawdown']:.2f}%", style={'color': '#f44336', 'fontSize': '24px', 'fontWeight': 'bold', 'textAlign': 'center'})
                                    ]),
                                    html.Div([
                                        html.Div('Volatility', style={'color': '#888', 'fontSize': '14px', 'textAlign': 'center'}),
                                        html.Div(f"{metrics['volatility']:.2f}%", style={'color': '#e0e0e0', 'fontSize': '24px', 'fontWeight': 'bold', 'textAlign': 'center'})
                                    ]),
                                    html.Div([
                                        html.Div('Total Return', style={'color': '#888', 'fontSize': '14px', 'textAlign': 'center'}),
                                        html.Div(f"{metrics['total_return']:.2f}%", style={'color': pnl_color, 'fontSize': '24px', 'fontWeight': 'bold', 'textAlign': 'center'})
                                    ])
                                ]
                            )
                        ]
                    ),
                    # Chart
                    dcc.Graph(
                        figure=create_figure(prices, actions, portfolio_values, metrics),
                        config={'displayModeBar': True}
                    )
                ]
            )
        ]
    )
    return app
