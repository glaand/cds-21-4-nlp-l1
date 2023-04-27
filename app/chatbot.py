import os
import streamlit as st
from pipeline import Pipeline
import pandas as pd
from plot_probabilities import plot_probs

pipe = Pipeline()

# Set title and subtitle
st.title('Luune-Detektor')
st.caption('Erstellt vo: Ajshe Fetai, André Glatzl, Alexandru Schneider')
chatgpt_key = st.text_input('Gib din ChatGPT Key ih:')
os.environ["OPENAI_API_KEY"] = chatgpt_key

# Create chatbot
prev_qry = ""
user_input = st.text_input('Gib en Satz ih, wod analysiert ha wotsch:')
button = st.button('Sendä')
if button or (prev_qry != user_input):
    prev_qry = user_input
    experiment, results = pipe.run([user_input])
    smileys = []
    smiley_df = experiment.sum(axis=0)
    pos = smiley_df.values[0]
    neg = smiley_df.values[1]
    neut = smiley_df.values[2]
    if pos > neg and pos > neut:
        smileys.append(':smile:')
    elif neg > pos and neg > neut:
        smileys.append(':cry:')
    elif (neut > pos and neut > neg) or pos == neg:
        smileys.append(':neutral_face:')
    elif pos == neut:
        smileys.append(':slightly_smiling_face:')
    elif neg == neut:
        smileys.append(':pensive:')
    else:
        smileys.append(':question:')
    #st.write(f"Modelle: \nPositiv: {pos}, Negativ: {neg}, Neutral: {neut}")
    st.markdown(f"<font size='20'>{''.join(smileys)}</font>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["X", "Plots"])
    
    with tab1:
        st.write('')
        
    with tab2:
        # Start Plots
        data = {
            'model_nr': [1, 2, 3, 4, 5],
            'de_pos': [0, 0, 0, 0, 0],
            'de_neg': [0, 0, 0, 0, 0],
            'de_neut': [0, 0, 0, 0, 0],
            'en_pos': [0, 0, 0, 0, 0],
            'en_neg': [0, 0, 0, 0, 0],
            'en_neut': [0, 0, 0, 0, 0],
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