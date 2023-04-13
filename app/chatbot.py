import streamlit as st
from streamlit_chat import message as st_message
from pipeline import Pipeline
pipe = Pipeline()

# Set title and subtitle
st.title('Luune-Detektor')
st.text('Gib en satz ih und mir segend dir ob er positiv, negativ oder neutral isch.')
st.caption('Erstellt vo: Ajshe Fetai, André Glatzl, Alexandru Schneider')

# Create chatbot
user_input = st.text_input('Wiä fühlsch du dich?')
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
