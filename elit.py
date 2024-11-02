import streamlit as st
from etal import *
from egpt import *


class elit:
    def __init__(self):
        st.set_page_config(page_title='Legacy - Mental Health', layout='centered')
        self.displayHeader()
        self.modelChoice = st.radio('Choose a Model', ['etal', 'egpt'])
        if self.modelChoice == 'etal':
            self.displayEtalPanel()
        elif self.modelChoice == 'egpt':
            self.displayEgptPanel()
    
    def displayHeader(self):
        st.title('Legacy - Mental Health')
        st.markdown('[Open Google Colab Notebook for Analysis](https://colab.research.google.com/drive/1UVrgohHSifjsw2OVP8j8EfDs_qeTOkCn?usp=sharing)')

    def displayEtalPanel(self):
        st.subheader('etal Model - Usage & Response')
        inputText = st.text_area('Enter Context for etal', placeholder='Type the context here...')
        if st.button('Get Response from etal'):
            model = etal()
            response = model.predict(inputText)
            st.write('Response:', response)

    def displayEgptPanel(self):
        st.subheader('egpt Model - Usage & Response')
        inputText = st.text_area('Enter Context for egpt', placeholder='Type the context here...')
        if st.button('Get Response from egpt'):
            apiKey = st.secrets['openai_api_key']
            model = egpt(apiKey)
            response = model.respond(inputText)
            st.write('Response:', response)
