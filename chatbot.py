import streamlit as st
from streamlit_chat import message as st_message
from pipeline import Pipeline
pipe = Pipeline()

# Set title and subtitle
st.title('Luune-Detektor')
st.text('Das isch en Chatbot wo dir hilft, dini Gfühl zerkenne.')
st.caption('Erstellt vo: Ajshe Fetai, André Glatzl, Alexandru Schneider')

# Create chatbot
user_input = st.text_input('Wiä fühlsch du dich?')
button = st.button('Sendä')
if button:
    result = pipe.run([user_input])
    # get first row of dataframe
    row = result.iloc[0]
    if row['pos'] == 1:
        st.write(':smile:')
    elif row['neg'] == 1:
        st.write(':pensive:')
    elif row['neut'] == 1:
        st.write(':neutral_face:')
    else:
        st.write(':question:')
