import requests
import pandas as pd
from pandas import DataFrame
import numpy as np
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

'''
Spam detector using Logitic Regression
By :YOUSSEF AIDANI
Year: 2019


'''

# Import dataset in its raw format from the txt files
hosts = pd.read_csv('hostnames.txt', delimiter=' ')
raw_assess = pd.read_csv('raw-assessments.txt', delimiter=' ')
labels = pd.read_csv('labels.txt', delimiter=' ')
# request = None # holds a refrence to the request session
request_session = None  # holds a reference to the request session
soup = None  # temporay var, holds the currently processed soup object / page
final_frame = DataFrame()
frames = None
similarity_vector = None
'''
    Raw ata preprocessing

'''


# Not all the links in the hostname list labeled so we reduce the list of links
# to the ones that were labeled to avoid lengthy preprocessing
def reduce_link_list():
    global hosts
    global labels
    global frames
    labels = labels[labels.spamicity != 'undecided']
    labels['spamicity'] = labels['spamicity'].map({'spam': 1, 'nonspam': 0})
    frames = [hosts, labels]
    res = pd.merge(hosts, labels, left_on=['id'], right_on=['id'], how='inner')
    hosts = res.iloc[:, 0:2]

    print('Link list reduced')


# res.to_excel("last_res.xlsx")
# res = pd.read_excel("hosts.xlsx")


# This function gets the raw dataset and extacts the links with respose code 200
def process_links():
    global labels
    global raw_assess
    x = 0  # counter value set to 0
    while x < len(hosts):
        print('Entry ' + str(x) + ' under focus: ' + hosts.iloc[x][1])
        print('Value of x after update is ' + str(x))
        try:
            requests.get('http://' + str(hosts.iloc[x][1]))
            print('Link accepted for index ' + str(x) + ' and link ' + str(hosts.iloc[x][1]))
            x += 1
            print('-------------\n')
        except Exception as e:
            print('Entry ' + str(x) + ' ' + hosts.iloc[x][1] + ' deleted!')
            host_id = hosts.iloc[x][0]
            labels = labels[labels.id != host_id]
            raw_assess = raw_assess[raw_assess.id != host_id]
            hosts.drop(hosts.index[x], inplace=True)
            set_size = len(hosts)
            # print(x)
            print('---------------\n')
            print(x)

    # This function persists the clearned datafrme object to a csv to avoid repeating lengthy link processing


def persist_clean_set():
    hosts.to_csv('wokring_links_hosts.csv', sep=' ')
    raw_assess.to_csv('working_links_assess.csv', sep=' ')
    labels.to_csv('working_links_labels.csv', sep=' ')

    # we preserve the variables to which we load our dataset, There is no need to redifine them


def get_clean_dataset():
    hosts = pd.read_csv('wokring_links_hosts.csv', delimiter=' ')
    raw_assess = pd.read_csv('working_links_assess.csv', delimiter=' ')
    labels = pd.read_csv('working_links_labels.csv', delimiter=' ')

    # This function extracts the page title for each link


def preprocessing_unit():
    reduce_link_list()
    process_links()
    persist_clean_set()
    get_clean_dataset()

    #############################################################################################


def set_up_request():
    # we set up a soup request session
    global request_session
    request_session = requests.session();
    page = request_session.get("https://google.com")
    if (page.ok):
        print('The Session is successfully set!')


def get_page_title():
    # print('Title: ------>  '+soup.find('title').get_text())
    if (soup.find('title') == None):
        return None
    else:
        return soup.find('title').get_text()


# This functin extracts the paragraphs existing on each page

def get_page_paragraphs():
    global soup
    parags = soup.find_all('p')
    # print(parags[1])
    concat = 'Value = \n'
    for x in range(0, len(parags)):
        tex = parags[x].get_text().strip()
        # print(str(x)+' : '+tex)
        if (len(tex) != 0):
            concat += tex
    return concat


def create_final_data_frame():
    global final_frame
    global request_session
    global request
    global soup
    x = 0
    current_link = ''
    current_title = ''
    currenet_paragraph_concatenation = ''
    set_up_request()
    while (x < len(hosts)):
        current_link = 'http://' + hosts.iloc[x][1]
        print('Soup from: ' + current_link)
        try:
            request = request_session.get(current_link)
            soup = BeautifulSoup(request.content, 'html.parser')
            current_title = get_page_title()
            print('current_title: ' + current_title)
            currenet_paragraph_concatenation = get_page_paragraphs()
        except Exception as e:
            print('Request Failed!')
            x += 1
            continue

        final_frame = final_frame.append(
            {'host_id': hosts.iloc[x][0], 'title': current_title, 'content': currenet_paragraph_concatenation,
             'average': labels.loc[labels.id == hosts.iloc[x][0]].iloc[0][2],
             'spamicity': labels.loc[labels.id == hosts.iloc[x][0]].iloc[0][1]}, ignore_index=True)
        x += 1
    print('Final Data Frame ready for processing')


# bag of words function that takes a title a paragraph as parameters
# logistic regression needed here   
def title_to_content_similarity():
    global similarity_vector
    similarity_vector = pd.DataFrame()
    title_content = final_frame[['title', 'content']]
    current_title = None
    current_content = None
    current_corpus = None
    current_matrix = None
    current_cos_silimarity = None
    tfidf_vectorizer = TfidfVectorizer()
    x = 0
    while x < len(title_content):
        current_title = str(title_content.iloc[x][0])
        current_content = str(title_content.iloc[x][1])
        current_corpus = [current_title, current_content]
        current_matrix = tfidf_vectorizer.fit_transform(current_corpus)
        current_cos_silimarity = cosine_similarity(current_matrix[0:1], current_matrix)
        current_cos_silimarity = current_cos_silimarity[0][1]
        similarity_vector = similarity_vector.append({'cos_sim': current_cos_silimarity}, ignore_index=True)
        print('title content with id ' + str(x) + ' have cos similiary of ' + str(current_cos_silimarity))
        x += 1


def processing_unit():
    global final_frame
    global similarity_vector
    title_to_content_similarity()
    x = x = final_frame[['average']]
    x = x.join(similarity_vector)
    y = final_frame['spamicity']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    for c in [0.01, 0.05, 0.25, 0.5, 1]:
        lr = LogisticRegression(C=c)
        lr
        lr.fit(x_train, y_train)
        print("Accuracy for C=%s: %s"
              % (c, accuracy_score(y_test, lr.predict(x_test))))


# run this fist
preprocessing_unit()
# This will show an error in case we dont run it first processing unit
create_final_data_frame()
# Then we make the preprocessing ::: machine learning
# uncomment the following line then execute processing unit in case you don't want to wait lengthy preprocessing
# final_frame = pd.read_csv('final_frame.csv',encoding = 'ISO-8859-1',delimiter=' ')
processing_unit()
