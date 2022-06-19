import ast
import pandas as pd
import numpy as np
import streamlit as st
import pickle
import re
import nltk 
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from random import choice

nltk.download('punkt')
nltk.download('stopwords')

if 'button1_pressed' not in st.session_state:
    st.session_state['button1_pressed']=False

if 'button2_pressed' not in st.session_state:
    st.session_state['button2_pressed']=False

if 'sort_attr' not in st.session_state:
    st.session_state['sort_attr']=None

if 'filter_by' not in st.session_state:
    st.session_state['filter_by']=None

@st.cache(allow_output_mutation=True,show_spinner=False)
def load_data():
    import pandas as pd
    import pickle
    similarities_path= 'similarities.pkl'
    with open(similarities_path, 'rb') as ref:
        similarities= pickle.load(ref)

    movie_tags_path= 'movie_tags.pkl'
    with open(movie_tags_path, 'rb') as ref:
        movie_tags= pickle.load(ref)
    movie_tags.year= pd.to_numeric(movie_tags.year, errors='coerce')
    return similarities, movie_tags

def has_any(x, values):
    return any([i in x for i in values])

def tokenize_and_lemmatize(text):
    lem_text= [WordNetLemmatizer().lemmatize(word) for word in word_tokenize(text) if word not in stopwords.words('english')]
    return ' '.join(lem_text)

def preprocess(text):
    text= re.sub('[^\w]',' ',text.lower())
    return tokenize_and_lemmatize(text)

def return_movie_id(text):
    for key,val in movie_dicts.items():
        if val==text:
            return key
        else:
            pass

def fetchimages(movie_id):
    import requests
    noimg='https://media.istockphoto.com/vectors/no-image-available-icon-vector-id1216251206?b=1&k=20&m=1216251206&s=170667a&w=0&h=z0hxu_BaI_tuMjMneE_APbnx_-R2KGPXgDjdwLw5W7o='
    path= 'https://api.themoviedb.org/3/movie/{}?api_key=90d93474949b3448283470d75cb41f56&language=en-US'.format(movie_id)
    try:
        data= requests.get(path).json()
        poster_path= 'https://image.tmdb.org/t/p/w500'+ data.get('poster_path')
        return poster_path
    except:
        return noimg

def find_similar_movies(movie_name, size=40, sorter=None, filter_by=None):
    similar_movie_ids= similarities[return_movie_id(movie_name)].sort_values(ascending=False).iloc[1:size+1].index
    if filter_by is not None:
        if sorter is not None:
            similar_movie_ids=movie_tags[(movie_tags.index.isin(similar_movie_ids)) & (movie_tags.genres.apply(lambda x: has_any(x, filter_by)))]\
                                .sort_values(by='popularity', ascending=False)\
                                .sort_values(by=sorter, ascending=False)\
                                .index.to_list()
        else:
            similar_movie_ids=movie_tags[(movie_tags.index.isin(similar_movie_ids)) & (movie_tags.genres.apply(lambda x: has_any(x, filter_by)))]\
                                .sort_values(by='popularity', ascending=False)\
                                .index.to_list()
    elif (filter_by is None) | (filter_by == 'All') :
        if sorter is not None:
            similar_movie_ids=movie_tags[(movie_tags.index.isin(similar_movie_ids))]\
                                .sort_values(by='popularity', ascending=False)\
                                .sort_values(by=sorter, ascending=False)\
                                .index.to_list()
        else:
            similar_movie_ids=movie_tags[(movie_tags.index.isin(similar_movie_ids))]\
                                .sort_values(by='popularity', ascending=False)\
                                .index.to_list()
    return [movie_dicts.get(int(id)) for id in similar_movie_ids]

def search_for_movie(keyword):
    keyword= preprocess(keyword)
    resultids= movie_tags[movie_tags.keywords.str.contains(keyword)]\
            .sort_values(by=['popularity','year'], ascending=False)\
            .index.to_list()
    return [movie_dicts.get(id) for id in resultids]

def display_movies_and_img(search_result_iter):
    for _ in range(10):
        with st.container() as ctr:
            columns= st.columns(4)
            for i, col in enumerate(columns):
                try:
                    mov= next(search_result_iter)
                    img= fetchimages(return_movie_id(mov))
                    col.image(img)
                    with col.expander(mov):
                        try:
                            write_details(return_movie_id(mov))
                        except:
                            pass
                except StopIteration:
                    col.empty()
                    pass

def find_movies(search_key,sorter=None, filter_by=None):
    search_results= search_for_movie(search_key)
    if (len(search_results) == 0):
        st.subheader("Sorry, we didn't find anything for '{}'. Please, try again.".format(search_key))
    elif search_key=='':
            pass
    else:
        similar_search= find_similar_movies(search_results[0], sorter=sorter, filter_by=filter_by)
        search_results.extend(similar_search)
        search_results= sorted(set(search_results), key=search_results.index)
        search_result_iter= iter(search_results)
        st.subheader("Showing results for '{}'".format(search_key))
        display_movies_and_img(search_result_iter)

def find_all_movies(sorter=None, filter_by= None):
    if filter_by is not None:
        if sorter is not None:
            all_movie_ids= movie_tags[movie_tags.genres.apply(lambda x: has_any(x, filter_by))]\
                                    .sort_values(by=['popularity','revenue','year'], ascending=False)\
                                    .sort_values(by=sorter, ascending=False).index
        else:
            all_movie_ids= movie_tags[movie_tags.genres.apply(lambda x: has_any(x, filter_by))]\
                                    .sort_values(by=['popularity','revenue','year'], ascending=False).index
    else:
        if sorter is not None:
            all_movie_ids= movie_tags.sort_values(by=['popularity','revenue','year'], ascending=False)\
                                     .sort_values(by=sorter, ascending=False).index
        else:
            all_movie_ids= movie_tags.sort_values(by=['popularity','revenue','year'], ascending=False).index
    search_results= [movie_dicts.get(id) for id in all_movie_ids]
    search_result_iter= iter(search_results)
    st.subheader("Showing all movies")
    display_movies_and_img(search_result_iter)

def display_top_picks(genres, label):
    top_pick_ids= movie_tags[movie_tags.genres.apply(lambda x: has_any(x, genres))]\
                        .sort_values(by= ['popularity','revenue'], ascending=False).index
    top_pick_ids= iter(top_pick_ids)

    with st.container():
        st.markdown(label)
        cols= st.columns(4)
    for col in cols:
        next_id= next(top_pick_ids)
        col.image(fetchimages(next_id))
        with col.expander(movie_dicts.get(next_id)):
            write_details(next_id)

def display_top_rated():
    genres= filters.get('All')
    top_pick_ids= movie_tags[movie_tags.genres.apply(lambda x: has_any(x, genres))]\
                        .sort_values(by= ['ratings','popularity'], ascending=False).index
    top_pick_ids= iter(top_pick_ids)

    with st.container():
        st.markdown('Top Rated by IMDB')
        cols= st.columns(4)
    for col in cols:
        next_id= next(top_pick_ids)
        col.image(fetchimages(next_id))
        with col.expander(movie_dicts.get(next_id)):
            write_details(next_id)

def fetch_mov_details(movie_id):
    import requests
    path= 'https://api.themoviedb.org/3/movie/{}?api_key=90d93474949b3448283470d75cb41f56&language=en-US'.format(movie_id)
    details= requests.get(path).json()
    casts= movie_tags.cast.str.split('|')[movie_id][:4]
    director= movie_tags.crew[movie_id]
    return {'Title': details.get('original_title'),
            'Year': int(details.get('release_date')[:4]),
            'Director': director ,
            'Overview': details.get('overview'),
            'Cast': casts
            }

def write_details(movie_id):
    mov_details= fetch_mov_details(movie_id)
    link= 'https://www.themoviedb.org/movie/{}'.format(movie_id)
    link = '[Go to website]({})'.format(link)
    st.markdown(link, unsafe_allow_html=True)
    st.text('Year: {}'.format(mov_details.get('Year')))
    st.text('Director: {}'.format(mov_details.get('Director')))
    st.text('Overview: {}'.format(mov_details.get('Overview')))
    st.text('Casts: {}'.format(','.join(mov_details.get('Cast'))))

similarities, movie_tags= load_data()

path= 'movie_dicts.pkl'
with open(path,'rb') as fp:
    movie_dicts= pickle.load(fp)

with open('./style.css') as css:
    html= "<style>{}</style>".format(css.read())

st.markdown(html,unsafe_allow_html=True)

with st.container():
    st.markdown('''<li style= 'color:red;line-height: 0.85;'>{}</li>
                   <li style= 'color: #a6a0b0; line-height: 0.85'>{}</li>'''.format('Movie','Recommender'), unsafe_allow_html=True)

cont1, cont2= st.container(), st.container()
for cont in [cont1, cont2]:
    with cont:
        st.empty()

with st.container():
    cols= st.columns(3)
    with cols[0]:
        text_input= st.text_input(label='  Search for a movie')
    cols[1].empty()

with st.container():
    cols= st.columns(7)
    btn=cols[3].button('Go')
    btn2= cols[0].button('Show all movies')
    cols[1].empty()
    cols[2].empty()
    cols[4].empty()
    cols[5].empty()
    cols[6].empty()

with st.container():
    cols= st.columns(6)
    for col in cols[:5]:
        col.empty()

if btn:
    st.session_state['button1_pressed']=True
if btn2:
    st.session_state['button2_pressed']=True


filters= {'All': movie_tags.genres.unique(), 'Suspense/Thrillers': ['Thriller','Mystery','Crime'], 'Drama':['Drama','Family'], 'Romance':['Romance'],
            'Action/Adventrure':['Action','Adventure'], 'Comedy':['Comedy'], 'Fantasy':'Fantasy',
            'Documentary':['Documentary'],'Horror':['Horror'], 'Science Fiction': ['ScienceFiction'], 'Animation': ['Animation'] }  
options= {'Latest': 'year','Highest grossing': 'revenue', 'Most popular':'popularity','Critically acclaimed':'ratings'}


# top picks for you


if st.session_state['button1_pressed']:  
    st.session_state['filter_by']= cols[4].selectbox('Refine your results',filters.keys())
    st.session_state['sort_attr']= cols[5].selectbox('Sort by', options.keys())
    find_movies(search_key= text_input, sorter=options.get(st.session_state['sort_attr']), 
                    filter_by=filters.get(st.session_state['filter_by']))

if st.session_state['button2_pressed']:  
    st.session_state['filter_by']= cols[4].selectbox('Refine your results',filters.keys())
    st.session_state['sort_attr']= cols[5].selectbox('Sort by', options.keys())
    find_all_movies(sorter=options.get(st.session_state['sort_attr']), 
                    filter_by=filters.get(st.session_state['filter_by']))

display_top_picks(filters.get('All'), 'Most popular')
display_top_picks(filters.get('Suspense/Thrillers'), 'Thrillers')
display_top_rated()
display_top_picks(filters.get('Romance'), 'Romantic movies')
display_top_picks(filters.get('Drama'), 'Dramas')
display_top_picks(filters.get('Horror'), 'Horror movies')
display_top_picks(filters.get('Comedy'), 'Comedy movies')





