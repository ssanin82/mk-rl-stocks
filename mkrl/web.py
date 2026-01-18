from plotly.subplots import make_subplots
import plotly.graph_objects as go


def create_figure(prices, actions, portfolio_values, metrics):
    # Ensure prices and actions are aligned - use minimum length
    min_len = min(len(prices), len(actions))
    prices = prices[:min_len]
    actions = actions[:min_len]
    
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
    
    # Price chart - only plot up to actions length
    fig.add_trace(
        go.Scatter(x=list(range(len(prices))), y=prices, name="Stock Price",
                  line=dict(color='#2196F3', width=2), fill='tozeroy',
                  fillcolor='rgba(33, 150, 243, 0.1)'),
        row=1, col=1
    )
    
    # Buy signals - only show if there are any
    if buy_indices:
        fig.add_trace(
            go.Scatter(x=buy_indices, y=[prices[i] for i in buy_indices],
                      mode='markers', name='Buy Signal',
                      marker=dict(color='#4CAF50', size=12, symbol='triangle-up')),
            row=1, col=1
        )
    
    # Sell signals - only show if there are any
    if sell_indices:
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


def create_static_html(prices, actions, portfolio_values, metrics, output_file='trading_results.html'):
    """Create a static HTML file with the trading results visualization."""
    fig = create_figure(prices, actions, portfolio_values, metrics)
    pnl_color = '#4CAF50' if metrics['total_pnl'] >= 0 else '#f44336'
    
    # Create HTML content
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>📈 RL Trading Strategy Results</title>
    <style>
        body {{
            font-family: 'Segoe UI', sans-serif;
            background-color: #1a1a1a;
            color: #e0e0e0;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .metrics-grid {{
            background: #2a2a2a;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .metrics-title {{
            color: #4CAF50;
            text-align: center;
            margin-bottom: 20px;
        }}
        .metrics-row {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-top: 20px;
        }}
        .metric-item {{
            text-align: center;
        }}
        .metric-label {{
            color: #888;
            font-size: 14px;
            margin-bottom: 8px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
        }}
        .chart-container {{
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="metrics-grid">
            <h2 class="metrics-title">Performance Metrics</h2>
            <div class="metrics-row">
                <div class="metric-item">
                    <div class="metric-label">Initial Capital</div>
                    <div class="metric-value" style="color: #e0e0e0;">${metrics['initial_capital']:.2f}</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Final Capital</div>
                    <div class="metric-value" style="color: {pnl_color};">${metrics['final_capital']:.2f}</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Total P&L</div>
                    <div class="metric-value" style="color: {pnl_color};">${metrics['total_pnl']:.2f}</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Max Drawdown</div>
                    <div class="metric-value" style="color: #f44336;">{metrics['max_drawdown']:.2f}%</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Volatility</div>
                    <div class="metric-value" style="color: #e0e0e0;">{metrics['volatility']:.2f}%</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Total Return</div>
                    <div class="metric-value" style="color: {pnl_color};">{metrics['total_return']:.2f}%</div>
                </div>
            </div>
        </div>
        <div class="chart-container">
            {fig.to_html(include_plotlyjs='cdn', div_id='plotly-chart', full_html=False)}
        </div>
    </div>
</body>
</html>'''
    
    # Write HTML to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return output_file
