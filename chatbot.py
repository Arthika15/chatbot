# Prepare a chatbot with NLTK Toolkit
# chatbot.txt is the corpus file
# importing libraries

import nltk as nt
import string as st
import random as rd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#importing and reading the corpus

file=open('chatbot.txt','r',errors='ignore')
doc=file.read()
doc=doc.lower()          #convert text to lowercase
nt.download('punkt')            #punkt tokentizer divides a text into list of sentences
nt.download('wordnet')          #wordnet dictionary to find meaning of words,synonmys and antonmys
nt.download('omw-1.4')
sent_tokens=nt.sent_tokenize(doc)
word_tokens=nt.word_tokenize(doc)

# Text preprocesing with lemmatization
#Lemmatization links the words with similar meanings to one word

lemmer=nt.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict=dict((ord(punct),None) for punct in st.punctuation)
def LemNormalize(text):
    return LemTokens(nt.word_tokenize(text.lower().translate(remove_punct_dict)))
greeting_inputs=["hello","hi","greetings","sup","what's up","hey"]
greeting_response=["hi","hello","hi there","welcome!!"]

# Defining a greeting function
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in greeting_inputs:
            return rd.choice(greeting_response)

# Response generation
def response(user_response):
    bot_response=''
    TfidfVec=TfidfVectorizer(tokenizer=LemNormalize,stop_words='english')
    tfidf=TfidfVec.fit_transform(sent_tokens)
    vals=cosine_similarity(tfidf[-1],tfidf)
    idx=vals.argsort()[0][-2]
    flat=vals.flatten()
    flat.sort()
    req_tfidf=flat[-2]
    if(req_tfidf==0):
        bot_response=bot_response+"Iam sorry!I don't understand you"
        return bot_response
    else:
        bot_response=bot_response+sent_tokens[idx]
        return bot_response

# defining start and end protocols
flag = True
print("BOT:Myself chatbot.let's have a conversation.Also,if you want to exit any time just type bye")
while (flag == True):
    user_response = input()
    user_response = user_response.lower()
    if (user_response != "bye"):
        if (user_response == "thanks" or user_response == "thankyou"):
            flag = False
            print("BOT:You are Welcome")
        else:
            if (greeting(user_response) != None):
                print("BOT:" + greeting(user_response))
            else:
                sent_tokens.append(user_response)
                word_tokens = word_tokens + nt.word_tokenize(user_response)
                final_words = list(set(word_tokens))
                print("BOT:", end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag = False
        print("BOT:Good bye!!!Have a nice day")
