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
    scoring, label = pipe.run(user_input)
    if label == 'pos':
        st.write(':smile:')
    elif label == 'neg':
        st.write(':pensive:')
    else:
        st.write(':neutral_face:')

