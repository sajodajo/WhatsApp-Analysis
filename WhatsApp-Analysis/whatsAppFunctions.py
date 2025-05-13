import re
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import plotly.express as px

def clean_line(line):
    return re.sub(r'[\u200e\u200f\u202a-\u202e]', '', line).strip()

def parse_whatsapp_chat(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        chat_data = file.readlines()

    messages = []
    
    for line in chat_data:
        line = clean_line(line)
        match = re.match(r'\[(\d{1,2}/\d{1,2}/\d{2}), (\d{2}:\d{2}:\d{2})\] (.+?): (.+)', line)
        if match:
            timestamp_str = f"{match.group(1)}, {match.group(2)}"
            timestamp = datetime.strptime(timestamp_str, '%d/%m/%y, %H:%M:%S')
            sender = match.group(3)
            message = match.group(4)
            messages.append([timestamp, sender, message])
    
    df = pd.DataFrame(messages, columns=["datetime", "sender", "message"])
    df = df.set_index("datetime").sort_index()
    return df

def totalMessages(df):
    return len(df)

def timeRange(df):
    minDate = df.index.min().date().strftime('%B %d, %Y')
    maxDate = df.index.max().date().strftime('%B %d, %Y')
    return f'{minDate} to {maxDate}'

def importAndClean(file):
    df = parse_whatsapp_chat(file)
    df = nameClean(df)
    df = otherTypes(df)
    tm = totalMessages(df)
    tr = timeRange(df)
    message = f'{tm} messages returned from the time period {tr}'
    return df, message

def otherTypes(df):
    df = df[~df['message'].isin(['image omitted', 'GIF omitted', 'video omitted', 'sticker omitted'])]
    df = df[~df['message'].str.startswith(('https://www.', 'www.'))]
    return df

def nameClean(df):
    df = df[df['sender'] != 'You']
    df = df[~df['sender'].str.contains('\+', na=False)] 

    return df

def messageCountTable(df):
    return df.groupby('sender').count().sort_values(by='message', ascending=False).head(20)

def topSenders(df):
    return df['sender'].value_counts().head(20)


def graphTalkers(df):
    talkers = messageCountTable(df)
    
    fig, ax = plt.subplots()

    gradient_colors = plt.cm.coolwarm(np.linspace(0, 1, len(talkers)))
    colors = [whatsapp_green] + list(gradient_colors[1:])

    talkers.plot(kind='bar', color=colors, ax=ax, label='No. of Messages')

    ax.set_xlabel("Author")
    ax.set_ylabel("Values")
    ax.set_title(f"Top Yappers as of {datetime.now().strftime('%B %d, %Y')}")
    ax.legend(['No. of Messages'])

    return fig



def graphTalkers(df):
    whatsapp_green = "#25D366"
    top_senders = df['sender'].value_counts().head(20).reset_index()
    top_senders.columns = ['Sender', 'Message Count']
    fig = px.bar(top_senders, x='Sender', y='Message Count', title='Top Contributors')

    fig.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',  
    plot_bgcolor='rgba(0,0,0,0)' 
    )
    return fig

def wordCheck(df, inputWord):
    df[inputWord] = df['message'].str.contains(inputWord, case=False, na=False)
    wordCheck = df.groupby('sender')[inputWord].sum().sort_values(ascending=False)
    return wordCheck


def allWords(df):
  df['words'] = df['message'].str.split()

  allWords = []

  for message in df['words']:
    for word in message:
      allWords.append(word)

  return allWords


def allWordsBySender(df):
    stopWords = set(stopwords.words('english'))
    customStops = {'omitted', 'image', 'message', 'sticker', 'edited', '<this message was edited>'}
    allStops = stopWords.union(customStops)

    df['words'] = df['message'].str.lower().str.split()
    sender_words = {}

    for _, row in df.iterrows():
        sender = row['sender']
        words = [word for word in row['words'] if word not in allStops]
        if sender not in sender_words:
            sender_words[sender] = []
        sender_words[sender].extend(words)

    return sender_words


def wordRank(df):
    df['message'] = df['message'].str.lower().apply(lambda x: re.sub(r'[\*\[\]\(\)\{\}\'\"“”‘’.,!?]', '', x))
    df['words'] = df['message'].str.split()

    allWords = [word for message in df['words'] for word in message]

    wordSeries = pd.Series(allWords)

    stopWords = set(stopwords.words('english'))

    customStops = {'omitted', 'image', 'message', 'sticker', 'edited','<this','edited>'}
    allStops = stopWords.union(customStops)

    wordSeries = wordSeries[~wordSeries.isin(allStops)]

    commonWords = wordSeries.value_counts()

    return commonWords

def checkTopWords(df,wordCheckList):
    commonWords = wordRank(df)
    topWords = commonWords.loc[wordCheckList]
    return topWords.sort_values(ascending=False)

def wordRatioCheck(df, inputWord):
    return (wordCheck(df,inputWord) / topSenders(df)).sort_values(ascending=False) *100

def wordsPerDay(df):
    NQBdaysold = ((datetime.today().date() - df.index.min().date()).days)
    return round(df['sender'].value_counts()/NQBdaysold,2)

def sentimentAnalyse(df):
    sia = SentimentIntensityAnalyzer()

    df['message'] = df['message'].astype(str)

    sentiment_df = df['message'].apply(sia.polarity_scores).apply(pd.Series)
    df = pd.concat([df, sentiment_df], axis=1)

    aggregated = df.groupby('sender')[['neg', 'neu', 'pos', 'compound']].mean()
    aggregated = aggregated.round(3)

    return aggregated.sort_values('compound', ascending=False)

def interactive_compound_chart(df):
    fig = px.bar(
        df.reset_index().sort_values('compound'),
        x='compound',
        y='sender',
        orientation='h',
        color='compound',
        color_continuous_scale='RdYlGn',
        range_color=[df['compound'].min(), df['compound'].max()],
        labels={'compound': 'Compound Score', 'sender': 'Sender'},
        height=600
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    fig.update_layout(xaxis_tickformat=".2f",showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)' )
    fig.update_coloraxes(showscale=False)
    return fig

def wordcloudSeries(word_series):
    freqs = word_series.to_dict()

    wordcloud = WordCloud(
        width=600,
        height=400,
        background_color=None, 
        mode='RGBA',           
        colormap='Greens_r',
        max_words=20
    ).generate_from_frequencies(freqs)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor='none')  # transparent figure
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_facecolor('none')  # transparent axes
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    return fig