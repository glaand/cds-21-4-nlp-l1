import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_probs(df): 
    # Create the subplots
    fig = make_subplots(rows=3, 
                        cols=2, 
                        specs=[[{'type': 'bar'}, {'type': 'bar'}], 
                               [{'colspan': 2,'type': 'table'}, None],
                               [{'colspan': 2,'type': 'table'}, None]],
                        subplot_titles=("Sentiment-Analyse DE", "Sentiment-Analyse EN", "Deutsch-Tabelle", "Englisch-Tabelle"))

    # Create the traces for each sentiment category and language
    x_axis_german = ['SentimentModell', 'TextBlob', 'PolyGlot', 'Spacy SentiWS', 'Chat GPT']
    x_axis_english = ['SentimentIntensity', 'TextBlob', 'Afinn', 'Flair: TextClassifier', 'Chat GPT']
    traces_de = [    
        go.Bar(x=x_axis_german, y=df['de_pos'], name='DE Positiv', marker=dict(color='blue'), text=['pos: {:.2f}'.format(v) for v in df['de_pos']], textposition='inside', legendgroup='1'),
        go.Bar(x=x_axis_german, y=df['de_neg'], name='DE Negativ', marker=dict(color='red'), text=['neg: {:.2f}'.format(v) for v in df['de_neg']], textposition='inside', legendgroup='2'),
        go.Bar(x=x_axis_german, y=df['de_neut'], name='DE Neutral', marker=dict(color='gray'), text=['neut: {:.2f}'.format(v) for v in df['de_neut']], textposition='inside', legendgroup='3')
    ]
    traces_en = [    
        go.Bar(x=x_axis_english, y=df['en_pos'], name='EN Positiv', marker=dict(color='blue'), text=['pos: {:.2f}'.format(v) for v in df['en_pos']], textposition='inside', legendgroup='1'),
        go.Bar(x=x_axis_english, y=df['en_neg'], name='EN Negativ', marker=dict(color='red'), text=['neg: {:.2f}'.format(v) for v in df['en_neg']], textposition='inside', legendgroup='2'),
        go.Bar(x=x_axis_english, y=df['en_neut'], name='EN Neutral', marker=dict(color='gray'), text=['neut: {:.2f}'.format(v) for v in df['en_neut']], textposition='inside', legendgroup='3')
    ]

    # Add the traces to the subplots
    for trace in traces_de:
        fig.add_trace(trace, row=1, col=1)
    for trace in traces_en:
        fig.add_trace(trace, row=1, col=2)

    # Create the table trace
    table_trace = go.Table(
        header=dict(
            values=['', 'Positiv', 'Neutral', 'Negativ'],
            font=dict(size=14),
            align='left'
        ),
        cells=dict(
            values=[
                x_axis_german,
                df['de_pos'], df['de_neut'], df['de_neg']
            ],
            font=dict(size=14),
            align='left'
        )
    )
    fig.add_trace(table_trace, row=2, col=1)
    table_trace = go.Table(
        header=dict(
            values=['', 'Positiv', 'Neutral', 'Negativ'],
            font=dict(size=14),
            align='left'
        ),
        cells=dict(
            values=[
                x_axis_english,
                df['en_pos'], df['en_neut'], df['en_neg']
            ],
            font=dict(size=14),
            align='left'
        )
    )
    fig.add_trace(table_trace, row=3, col=1)

    # Update the layout
    fig.update_layout(
        title='Sentiment-Analyse Resultate',
        font=dict(size=16),
        height=800,
        yaxis1_range=[0,1],
        legend=dict(
            title='Sentiment',
            traceorder="normal",
            itemsizing='constant',
            itemclick='toggleothers',
            font=dict(
                size=14
            )
        )
    )
    return fig