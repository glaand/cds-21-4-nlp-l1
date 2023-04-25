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
    experiment, results = pipe.run([user_input])
    # get first row of dataframe
    row = experiment.iloc[0]
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
    #st.write(''.join(smileys))
    
    # TODO: CHANGE data to result reveived from pipeline @Andre
    data = {
        'model_nr': [1, 2, 3, 4, 5],
        'de_pos': [0, 0, 0, 0, 7],
        'de_neg': [0, 0, 0, 0, 7],
        'de_neut': [0, 0, 0, 0, 7],
        'en_pos': [0, 0, 0, 0, 7],
        'en_neg': [0, 0, 0, 0, 7],
        'en_neut': [0, 0, 0, 0, 7],
    }

    for i in range(len(results)):
        model_language = 'en'
        if i >= 5:
            model_language = 'de'
        cur_df = results[i]

        # get first row of dataframe
        rel_i = i
        if model_language == 'de':
            rel_i = i - 5
        row = cur_df.iloc[0]
        data[f'{model_language}_pos'][rel_i] = row['pos']
        data[f'{model_language}_neg'][rel_i] = row['neg']
        data[f'{model_language}_neut'][rel_i] = row['neut']


    df = pd.DataFrame(data)
    df.set_index('model_nr', inplace=True)
    fig = plot_probs(df)
    st.write(fig)