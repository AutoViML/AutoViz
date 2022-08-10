"""
Copyright 2020 Google LLC. This software is provided as-is, without warranty or
representation for any use or purpose. Your use of it is subject to your
agreement with Google.
"""
# --------------------------------------------------------------------------------
import pandas as pd, numpy as np
import pdb
import nltk
pd.set_option('display.max_colwidth',5000)
# --------------------------------------------------------------------------------
import nltk
#nltk.download('popular')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from operator import itemgetter
import copy
from nltk import word_tokenize, pos_tag
from collections import Counter
from nltk.stem.wordnet import WordNetLemmatizer

#Contraction map
c_dict = {
  "ain't": "am not",
  "aren't": "are not",
  "cant":"cannot",
  "can't": "cannot",
  "can't've": "cannot have",
  "'cause": "because",
  "b": "be",
  "bc": "because",
  "becos":"because",
  "bs": "Expletive",
  "cause": "because",
  "could've": "could have",
  "couldn't": "could not",
  "couldn't've": "could not have",
  "corp": "corporation",
  "cud":"could",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "execs": "executives",
  "fck": "fuck",
  "fcking": "fucking",
  "gon na": "going to",
  "hadn't": "had not",
  "hadn't've": "had not have",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he would",
  "he'd've": "he would have",
  "he'll": "he will",
  "he'll've": "he will have",
  "he's": "he is",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how is",
  "im": "i am",
  "iam": "i am",
  "i'd": "I would",
  "i'd've": "I would have",
  "i'll": "I will",
  "i'll've": "I will have",
  "i'm": "I am",
  "i've": "I have",
  "isn't": "is not",
  "it'd": "it had",
  "it'd've": "it would have",
  "it'll": "it will",
  "it'll've": "it will have",
  "it's": "it is",
  "let's": "let us",
  "ma'am": "madam",
  "mayn't": "may not",
  "mgr": "manager",
  "might've": "might have",
  "mightn't": "might not",
  "mightn't've": "might not have",
  "must've": "must have",
  "mustn't": "must not",
  "mustn't've": "must not have",
  "needn't": "need not",
  "needn't've": "need not have",
  "o'clock": "of the clock",
  "ofc": "office",
  "oughtn't": "ought not",
  "oughtn't've": "ought not have",
  "pics": "pictures",
  "shan't": "shall not",
  "sha'n't": "shall not",
  "shan't've": "shall not have",
  "she'd": "she would",
  "she'd've": "she would have",
  "she'll": "she will",
  "she'll've": "she will have",
  "she's": "she is",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "so's": "so is",
  "svc":"service",
  "that'd": "that would",
  "that'd've": "that would have",
  "that's": "that is",
  "there'd": "there had",
  "there'd've": "there would have",
  "there's": "there is",
  "they'd": "they would",
  "they'd've": "they would have",
  "they'll": "they will",
  "they'll've": "they will have",
  "they're": "they are",
  "they've": "they have",
  "tho":"though",
  "to've": "to have",
  "wan na": "want to",
  "wasn't": "was not",
  "we'd": "we had",
  "we'd've": "we would have",
  "we'll": "we will",
  "we'll've": "we will have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what'll": "what will",
  "what'll've": "what will have",
  "what're": "what are",
  "what's": "what is",
  "what've": "what have",
  "when's": "when is",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where is",
  "where've": "where have",
  "who'll": "who will",
  "who'll've": "who will have",
  "who's": "who is",
  "who've": "who have",
  "why's": "why is",
  "why've": "why have",
  "will've": "will have",
  "won't": "will not",
  "won't've": "will not have",
  "would've": "would have",
  "wouldn't": "would not",
  "wouldn't've": "would not have",
  "y'all": "you all",
  "y'alls": "you alls",
  "y'all'd": "you all would",
  "y'all'd've": "you all would have",
  "y'all're": "you all are",
  "y'all've": "you all have",
  "you'd": "you had",
  "you'd've": "you would have",
  "you'll": "you you will",
  "you'll've": "you you will have",
  "you're": "you are",
  "you've": "you have"
}

##################################################################################
def left_subtract(l1,l2):
    lst = []
    for i in l1:
        if i not in l2:
            lst.append(i)
    return lst
################################################################################
def return_stop_words():
    from nltk.corpus import stopwords
    STOP_WORDS = ['it', "this", "that", "to", 'its', 'am', 'is', 'are', 'was', 'were', 'a',
                'an', 'the', 'and', 'or', 'of', 'at', 'by', 'for', 'with', 'about', 'between',
                 'into','above', 'below', 'from', 'up', 'down', 'in', 'out', 'on', 'over','will','shall','could',
                  'under', 'again', 'further', 'then', 'once', 'all', 'any', 'both', 'each','would',
                   'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so',
                    'than', 'too', 'very', 's', 't', 'can', 'just', 'd', 'll', 'm', 'o', 're',
                    've', 'y', 'ain', 'ma','them','themselves','they','he','she','ex','become','their']
    add_words = ["s", "m",'you', 'not',  'get', 'no', 'via', 'one', 'still', 'us', 'u','hey','hi','oh','jeez',
                'the', 'a', 'in', 'to', 'of', 'i', 'and', 'is', 'for', 'on', 'it', 'got','aww','awww',
                'not', 'my', 'that', 'by', 'with', 'are', 'at', 'this', 'from', 'be', 'have', 'was',
                '', ' ', 'say', 's', 'u', 'ap', 'afp', '...', 'n', '\\']
    #stopWords = text.ENGLISH_STOP_WORDS.union(add_words)
    stop_words = list(set(STOP_WORDS+add_words))
    excl =['will',"i'll",'shall',"you'll",'may',"don't","hadn't","hasn't","haven't",
           "don't","isn't",'if',"mightn't","mustn'","mightn't",'mightn',"needn't",
           'needn',"needn't",'no','not','shan',"shan't",'shouldn',"shouldn't","wasn't",
          'wasn','weren',"weren't",'won',"won't",'wouldn',"wouldn't","you'd",
          "you'd","you'll","you're",'yourself','yourselves']
    stopWords = left_subtract(stop_words,excl)
    return sorted(stopWords)
##################################################################################
def expandContractions(text):
    """
    Takes in a sentence, splits it into list of strings and returns sentence back with
    items sibstituted by expanded abbreviations or abbreviated words that are expanded.
    """
    text_list = text.split(" ")
    return " ".join([c_dict.get(item, item) for item in text_list])
#
# remove entire URL
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

# Remove just HTML markup language
def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Convert Emojis to Text
import emoji
def convert_emojis(text):
    try:
        return emoji.demojize(text)
    except:
        return "Errorintext"

import string
def remove_punct(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)

# Clean even further removing non-printable text
# Thanks to https://www.kaggle.com/rftexas/text-only-kfold-bert
import string
def remove_stopwords(tweet):
    """Removes STOP_WORDS characters"""
    stop_words = return_stop_words()
    tweet = tweet.lower()
    tweet = ' '.join([x for x in tweet.split(" ") if x not in stop_words])
    tweet = ''.join([x for x in tweet if x in string.printable])
    return tweet

# define a function that accepts text and returns a list of lemmas
def split_into_lemmas(text):
    words = TextBlob(text).words
    text = ' '.join([word.lemmatize() for word in words])
    return text

# Expand Slangs
# Thanks to https://www.kaggle.com/rftexas/text-only-kfold-bert
slangs = {
    "IG" : "Instagram",
    "FB": "Facebook",
    "MOFO" : "Expletive",
    "OMG" : "Oh my God",
    "ROFL" : "roll on the floor laughing",
    "ROFLOL" : "roll on the floor laughing out loud",
    "ROTFLMAO" : "roll on the floor laughing my ass off",
    "FCK": "Expletive",
    "LMAO": "Laugh my Ass off",
    "LOL" : "laugh out loud",
}

abbreviations = {
    "$" : " dollar ",
    "€" : " euro ",
    "4ao" : "for adults only",
    "a.m" : "before midday",
    "a3" : "anytime anywhere anyplace",
    "aamof" : "as a matter of fact",
    "acct" : "account",
    "adih" : "another day in hell",
    "afaic" : "as far as i am concerned",
    "afaict" : "as far as i can tell",
    "afaik" : "as far as i know",
    "afair" : "as far as i remember",
    "afk" : "away from keyboard",
    "app" : "application",
    "approx" : "approximately",
    "apps" : "applications",
    "asap" : "as soon as possible",
    "asl" : "age, sex, location",
    "atk" : "at the keyboard",
    "ave." : "avenue",
    "aymm" : "are you my mother",
    "ayor" : "at your own risk",
    "b&b" : "bed and breakfast",
    "b+b" : "bed and breakfast",
    "b.c" : "before christ",
    "b2b" : "business to business",
    "b2c" : "business to customer",
    "b4" : "before",
    "b4n" : "bye for now",
    "b@u" : "back at you",
    "bae" : "before anyone else",
    "bak" : "back at keyboard",
    "bbbg" : "bye bye be good",
    "bbc" : "british broadcasting corporation",
    "bbias" : "be back in a second",
    "bbl" : "be back later",
    "bbs" : "be back soon",
    "be4" : "before",
    "bfn" : "bye for now",
    "blvd" : "boulevard",
    "bout" : "about",
    "brb" : "be right back",
    "bros" : "brothers",
    "brt" : "be right there",
    "bsaaw" : "big smile and a wink",
    "btch": "bitch",
    "btw" : "by the way",
    "btfd": "buy the Expletive dip",
    "bwl" : "bursting with laughter",
    "c/o" : "care of",
    "cet" : "central european time",
    "cf" : "compare",
    "cia" : "central intelligence agency",
    "csl" : "can not stop laughing",
    "cu" : "see you",
    "cul8r" : "see you later",
    "cv" : "curriculum vitae",
    "cwot" : "complete waste of time",
    "cya" : "see you",
    "cyt" : "see you tomorrow",
    "dae" : "does anyone else",
    "dbmib" : "do not bother me i am busy",
    "diy" : "do it yourself",
    "dm" : "direct message",
    "dwh" : "during work hours",
    "e123" : "easy as one two three",
    "eet" : "eastern european time",
    "eg" : "example",
    "embm" : "early morning business meeting",
    "encl" : "enclosed",
    "encl." : "enclosed",
    "etc" : "and so on",
    "faq" : "frequently asked questions",
    "fawc" : "for anyone who cares",
    "fb" : "facebook",
    "fc" : "fingers crossed",
    "fig" : "figure",
    "fimh" : "forever in my heart",
    "ft." : "feet",
    "ft" : "featuring",
    "ftl" : "for the loss",
    "ftw" : "for the win",
    "fwiw" : "for what it is worth",
    "fyi" : "for your information",
    "g9" : "genius",
    "gahoy" : "get a hold of yourself",
    "gal" : "get a life",
    "gcse" : "general certificate of secondary education",
    "gfn" : "gone for now",
    "gg" : "good game",
    "gl" : "good luck",
    "glhf" : "good luck have fun",
    "gmt" : "greenwich mean time",
    "gmta" : "great minds think alike",
    "gn" : "good night",
    "g.o.a.t" : "greatest of all time",
    "goat" : "greatest of all time",
    "goi" : "get over it",
    "gps" : "global positioning system",
    "gr8" : "great",
    "gratz" : "congratulations",
    "gyal" : "girl",
    "h&c" : "hot and cold",
    "hp" : "horsepower",
    "hr" : "hour",
    "hrh" : "his royal highness",
    "ht" : "height",
    "ibrb" : "i will be right back",
    "ic" : "i see",
    "icq" : "i seek you",
    "icymi" : "in case you missed it",
    "idc" : "i do not care",
    "idgadf" : "i do not give a damn Expletive",
    "idgaf" : "i do not give a Expletive",
    "idk" : "i do not know",
    "ie" : "that is",
    "i.e" : "that is",
    "ifyp" : "i feel your pain",
    "iirc" : "if i remember correctly",
    "ilu" : "i love you",
    "ily" : "i love you",
    "imho" : "in my humble opinion",
    "imo" : "in my opinion",
    "imu" : "i miss you",
    "iow" : "in other words",
    "irl" : "in real life",
    "j4f" : "just for fun",
    "jic" : "just in case",
    "jk" : "just kidding",
    "jsyk" : "just so you know",
    "l8r" : "later",
    "lb" : "pound",
    "lbs" : "pounds",
    "ldr" : "long distance relationship",
    "lmao" : "laugh my ass off",
    "lmfao" : "laugh my Expletive ass off",
    "lol" : "laugh out loud",
    "ltd" : "limited",
    "ltns" : "long time no see",
    "m8" : "mate",
    "mf" : "Expletive",
    "mfing": "Expletive",
    "mfs" : "Expletive",
    "mfw" : "my face when",
    "mofo" : "Expletive",
    "mph" : "miles per hour",
    "mr" : "mister",
    "mrw" : "my reaction when",
    "ms" : "miss",
    "mte" : "my thoughts exactly",
    "nagi" : "not a good idea",
    "nbc" : "national broadcasting company",
    "nbd" : "not big deal",
    "nfs" : "not for sale",
    "ngl" : "not going to lie",
    "nhs" : "national health service",
    "nrn" : "no reply necessary",
    "nsfl" : "not safe for life",
    "nsfw" : "not safe for work",
    "nth" : "nice to have",
    "nvr" : "never",
    "nyc" : "new york city",
    "oc" : "original content",
    "og" : "original",
    "ohp" : "overhead projector",
    "oic" : "oh i see",
    "omdb" : "over my dead body",
    "omg" : "oh my god",
    "omw" : "on my way",
    "p.a" : "per annum",
    "p.m" : "after midday",
    "pm" : "prime minister",
    "poc" : "people of color",
    "pov" : "point of view",
    "pp" : "pages",
    "ppl" : "people",
    "prw" : "parents are watching",
    "ps" : "postscript",
    "pt" : "point",
    "ptb" : "please text back",
    "pto" : "please turn over",
    "qpsa" : "what happens", #"que pasa",
    "ratchet" : "rude",
    "rbtl" : "read between the lines",
    "rlrt" : "real life retweet",
    "rofl" : "rolling on the floor laughing",
    "roflol" : "rolling on the floor laughing out loud",
    "rotflmao" : "rolling on the floor laughing my ass off",
    "rt" : "retweet",
    "ruok" : "are you ok",
    "sfw" : "safe for work",
    "sk8" : "skate",
    "smh" : "shake my head",
    "sq" : "square",
    "srsly" : "seriously",
    "ssdd" : "same stuff different day",
    "tbh" : "to be honest",
    "tbs" : "tablespooful",
    "tbsp" : "tablespooful",
    "tfw" : "that feeling when",
    "thks" : "thank you",
    "tho" : "though",
    "thx" : "thank you",
    "tia" : "thanks in advance",
    "til" : "today i learned",
    "tl;dr" : "too long i did not read",
    "tldr" : "too long i did not read",
    "tmb" : "tweet me back",
    "tntl" : "trying not to laugh",
    "ttyl" : "talk to you later",
    "u" : "you",
    "u2" : "you too",
    "u4e" : "yours for ever",
    "utc" : "coordinated universal time",
    "w/" : "with",
    "w/o" : "without",
    "w8" : "wait",
    "wassup" : "what is up",
    "wb" : "welcome back",
    "wtf" : "what the Expletive",
    "WTF" : "what the Expletive",
    "wtg" : "way to go",
    "wtpa" : "where the party at",
    "wuf" : "where are you from",
    "wuzup" : "what is up",
    "wywh" : "wish you were here",
    "yd" : "yard",
    "ygtr" : "you got that right",
    "ynk" : "you never know",
    "zzz" : "sleeping bored and tired"
}

# Thanks to https://www.kaggle.com/rftexas/text-only-kfold-bert
from nltk.tokenize import word_tokenize

# Thanks to https://www.kaggle.com/rftexas/text-only-kfold-bert
def expandAbbreviations(sentence):
    text = sentence.split(" ")
    return " ".join([abbreviations.get(item, item) for item in text])

# Thanks to https://www.kaggle.com/rftexas/text-only-kfold-bert
def expandSlangs(sentence):
    text = sentence.split(" ")
    return " ".join([slangs.get(item, item) for item in text])

def join_words(text):
    return " ".join(text)

def remove_punctuations(text):
    try:
        remove_puncs = re.sub(r'[?|!|~|@|$|%|^|&|#]', r'', text)
    except:
        return "error"
    return re.sub(r'[.|,\'|,|)|(|\|/|+|-|{|}|:|]', r' ', remove_puncs)

import re
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Remove punctuation marks
import string
def remove_punct(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)

#### This counts emojis in a sentence which is very helpful to gauge sentiment
def count_emojis(sentence):
    import regex
    import emoji
    emoji_counter = 0
    data = regex.findall(r'\X', sentence)
    for word in data:
        if any(char in emoji.UNICODE_EMOJI for char in word):
            emoji_counter += 1
    return emoji_counter
################################################################################
import re
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from textblob import TextBlob, Word
from itertools import chain

replace_spaces = re.compile('[/(){}\[\]\|@,;]')
remove_special_chars = re.compile('[^0-9a-z #+_]')
STOPWORDS = return_stop_words()
remove_ip_addr = re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b')

def clean_steps(text):
    text = text.replace('\n', ' ').lower()#
    text = remove_ip_addr.sub('', text)
    text = replace_spaces.sub(' ',text)
    text = remove_special_chars.sub('',text)
    text = ' '.join([w for w in text.split() if not w in STOPWORDS])
    return text

def clean_text(x):
    """
    ###############################################################################
    ## This cleans text string. Use it only as a Series.map(clean_text) function  #
    ###############################################################################
    Input must be one text string only. Don't send arrays or dataframes.
    Clean steps cleans one tweet at a time using following steps:
    1. removes URL
    2. Removes a very small list of stop words - about 65
    """
    x = expandSlangs(x) ### do this before lowering case since case is important for sentiment
    x = expandAbbreviations(x) ### this is before lowering case since case is important in sentiment
    x = expandContractions(x)  ### this is after lowering case - just to double check
    x = remove_stopwords(x) ## this works well to remove a small number of stop words
    x = remove_punctuations(x) # this works well to remove punctuations and add spaces correctly
    x = split_into_lemmas(x) ## this lemmatizes text and gets it ready for wordclouds ###
    return x

def draw_wordcloud_from_dataframe(dataframe, column, chart_format, 
                                depVar, mk_dir, verbose=0):
    """
    This handy function draws a dataframe column using Wordcloud library and nltk.
    """
    imgdata_list = []
    
    ### Remember that fillna only works at dataframe level! ##
    X_train = dataframe[[column]].fillna("missing")
    ### Map function only works on Series, so you should use this ###
    X_train = X_train[column].map(clean_steps)
    ### next time, you get back a series, so just use it as is ###
    X_train = X_train.map(clean_text)
    
    # Dictionary of all words from train corpus with their counts.

    ### Fantastic way to count words using one line of code #############
    ###  Thanks to : https://stackoverflow.com/questions/35857519/efficiently-count-word-frequencies-in-python
    words_counts = Counter(chain.from_iterable(map(str.split, X_train)))
    vocab_size = 50000
    top_words = sorted(words_counts, key=words_counts.get, reverse=True)[:vocab_size]
    text_join = ' '.join(top_words)

    #picture_mask = plt.imread('test.png')

    wordcloud1 = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='white',
                          width=1800,
                          height=1400,
                          #mask=picture_mask
                ).generate(text_join)
    return wordcloud1
################################################################################
# Removes duplicates from a list to return unique values - USED ONLYONCE
def find_remove_duplicates(values):
    output = []
    seen = set()
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output

def draw_word_clouds(dft, each_string_var, chart_format, plotname, 
                        dep, problem_type, classes, mk_dir, verbose=0):
    dft = dft[:]
    width_size = 20
    height_size = 10
    image_count = 0
    imgdata_list = []
    
    if problem_type == 'Regression' or problem_type == 'Clustering':
        ########## This is for Regression and Clustering problems only #####
        num_plots = 1
        fig = plt.figure(figsize=(min(num_plots*width_size,20),min(num_plots*height_size,20)))
        cols = 2
        rows = int(num_plots/cols + 0.5)
        plotc = 1
        while plotc <= num_plots:
            plt.subplot(rows, cols, plotc)
            ax1 = plt.gca()
            wc1 = draw_wordcloud_from_dataframe(dft, each_string_var, chart_format,
                    dep, mk_dir, verbose)
            plotc += 1
            ax1.axis("off")
            ax1.imshow(wc1)
            ax1.set_title('Wordcloud for %s' %each_string_var)
        image_count = 0
        if verbose == 2:
            imgdata_list.append(save_image_data(fig, chart_format,
                                plotname, depVar, mk_dir))
            image_count += 1
        if verbose <= 1:
            plt.show();
    else:
        ########## This is for Classification problems only ###########
        num_plots = len(classes)
        target_vars = dft[dep].unique()
        fig = plt.figure(figsize=(min(num_plots*width_size,20),min(num_plots*height_size,20)))
        cols = 2
        rows = int(num_plots/cols + 0.5)
        plotc = 1
        while plotc <= num_plots:
            plt.subplot(rows, cols, plotc)
            ax1 = plt.gca()
            ax1.axis("off")
            dft_target = dft.loc[(dft[dep] == target_vars[plotc-1])][each_string_var]
            if isinstance(dft_target,pd.Series):
                wc1 = draw_wordcloud_from_dataframe(pd.DataFrame(dft_target), each_string_var, chart_format,
                        dep, mk_dir, verbose)
            else:
                wc1 = draw_wordcloud_from_dataframe(dft_target, each_string_var, chart_format,
                        dep, mk_dir, verbose)
            ax1.imshow(wc1)
            ax1.set_title('Wordcloud for %s, target=%s' %(each_string_var, target_vars[plotc-1]), fontsize=20)
            plotc += 1
        fig.tight_layout();
        ### This is where you save the fig or show the fig ######
        image_count = 0
        if verbose == 2:
            imgdata_list.append(save_image_data(fig, chart_format,
                                plotname, depVar, mk_dir))
            image_count += 1
        if verbose <= 1:
            plt.show();
    ####### End of Word Clouds #############################

