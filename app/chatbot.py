import streamlit as st
from pipeline import Pipeline
import pandas as pd
from plot_probabilities import plot_probs

pipe = Pipeline()

# Set title and subtitle
st.title('Luune-Detektor')
st.text('Gib en satz ih und mir seged dir ob er positiv, negativ oder neutral isch.')
st.caption('Erstellt vo: Ajshe Fetai, André Glatzl, Alexandru Schneider')

# Create chatbot
user_input = st.text_input('Gib din Satz ih. Mir analysierend en.')
button = st.button('Sendä')
if button:
    result = pipe.run([user_input])
    print(result)
    # get first row of dataframe
    row = result.iloc[0]
    smileys = []
    for index, value in row.items():
        if index == 'pos' and value == 1:
            smileys.append(':smile:')
        elif index == 'neg' and value == 1:
            smileys.append(':pensive:')
        elif index == 'neut' and value == 1:
            smileys.append(':neutral_face:')
    if len(smileys) == 0:
        smileys.append(':question:')
    st.write(''.join(smileys))
    
    # TODO: CHANGE data to result reveived from pipeline @Andre
    data = {
        'model_nr': [1, 2, 3, 4, 5],
        'de_pos': [0.9, 0.8, 0.7, 0.6, 0.5],
        'de_neg': [0.05, 0.1, 0.15, 0.2, 0.25],
        'de_neut': [0.05, 0.1, 0.15, 0.2, 0.25],
        'en_pos': [0.87, 0.91, 0.95, 0.94, 0.98],
        'en_neg': [0.03, 0.02, 0.01, 0.02, 0.01],
        'en_neut': [0.1, 0.07, 0.04, 0.04, 0.01]
    }
    df = pd.DataFrame(data)
    df.set_index('model_nr', inplace=True)
    fig = plot_probs(df)
    st.write(fig)