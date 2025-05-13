import streamlit as st
import pandas as pd
import whatsAppFunctions as waf
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import whatsAppFunctions as waf
import tempfile
import plotly.express as px

st.set_page_config(
    page_title="WA Groupchat Analysis",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/thumb/6/6b/WhatsApp.svg/2044px-WhatsApp.svg.png",  
    layout="wide"
)
st.markdown(
"""
<h1 style='text-align: center;'>WhatsApp Groupchat Analysis</h1>
""",
unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://i.redd.it/ts7vuoswhwf41.jpg");
        background-size: auto;
        background-repeat: repeat;
        background-position: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col2:
    chatFile = st.file_uploader("", type=["txt"])

if chatFile:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
        tmp_file.write(chatFile.read())
        tmp_path = tmp_file.name

    df, message = waf.importAndClean(tmp_path)

    col1, col2 = st.columns(2)

    with col1:
        messageCount = waf.messageCountTable(df)
        topTalker, topMessages = messageCount.index[0], messageCount['message'][0]

        st.title("🗣️ Top Contributors")
        fig = waf.graphTalkers(df)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**The most talkative is `{topTalker}` with `{topMessages}` messages sent to the group.**")

    with col2:
        st.title("🔤 Avg. Words per Day per Member")
        st.dataframe(waf.wordsPerDay(df))



    st.markdown(
        """
        <h1 style='text-align: center; color: white;'>Most Used Words</h1>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        allwords = waf.wordRank(df)
        allwords_df = allwords.reset_index()
        allwords_df.columns = ['Word', 'Count']
        st.dataframe(allwords_df)

    with col2:
        fig = waf.wordcloudSeries(waf.wordRank(df))
        st.pyplot(fig)

    st.title("🧠 Message Sentiment Analysis")

    sentiment = waf.sentimentAnalyse(df)
    sentimentChart = waf.interactive_compound_chart(sentiment)

    st.plotly_chart(sentimentChart, use_container_width=True)

else:


    st.subheader("Instructions:")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.write("1. Open the WhatsApp group chat and tap on the group name at the top.")
        st.image('images/Screenshot1.png')
    
    with col2:
        st.write("2. Scroll down and select 'Export Chat' from the menu")
        st.image('images/Screenshot2.png')

    with col3:
        st.write("3. Choose 'Without Media' to allow for correct processing.")
        st.image('images/Screenshot3.png')

    with col4:
        st.write("4. Go to your Downloads folder and find the compressed chat file.")
        st.image('images/Screenshot4.png')

    with col5:
        st.write("5. Unzip and upload the `.txt` file using the button above.")
        st.image('images/Screenshot5.png')





