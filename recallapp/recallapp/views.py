from flask import render_template
from flask import request, jsonify
from recallapp import app
#from sqlalchemy import create_engine
#from sqlalchemy_utils import database_exists, create_database
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
#import psycopg2
from sklearn.metrics.pairwise import cosine_similarity
pd.options.display.max_columns=25


global user_rating, already_seen_asins, currently_shown, df, df_ids, collab, filepath
# get asin title data (connect to database later)
df = pd.read_csv('recallapp/data/asinCombo.csv')
df.drop(df.columns[0],axis=1,inplace=True)

# get asins that are in the collab filter model (not all are)
df_ids = pd.read_csv('recallapp/data/collabAsinOrder.csv')

# load collab-filter data
collab = np.load('recallapp/data/predMatrix.npy')
pop_recs_idx = np.argsort(-np.mean(collab,axis=0))
#print(pop_recs_idx[0])
pop_recs = df_ids['0'].iloc[pop_recs_idx[:]].to_numpy()
pop_recs = [(t,) for t in pop_recs]

# load doc2vec model first
docvecs = KeyedVectors.load('recallapp/data/d2v_docvecs.bin', mmap='r')
wordvecs = KeyedVectors.load('recallapp/data/d2v_wvs.bin', mmap='r')
word_vocab = list(wordvecs.vocab.keys())
#model = Doc2Vec.load('recallapp/data/d2v_2.model') 
#model_vocab = list(model.wv.vocab.keys())

# initialize recommendations for a new user:
#user_vec = 3*np.ones([1,np.shape(collab)[1]])
already_seen_asins = []
currently_shown = []

def getInfo(asin):
  return df[df['asin']==asin].values.tolist()

def find_sim_profile(user_ratings):
  sims = cosine_similarity(user_ratings, collab)
  return np.argmax(sims)

def get_collab_rec(profile, itemType):
  global user_rating, already_seen_asins
  top_recs = np.argsort(-(collab[profile,:]),kind='mergesort')
  for tr in top_recs:
    if abs(user_rating[0][tr]-3) < 1e-6:
      item_rec_asin = df_ids['0'].iloc[tr]
      if (item_rec_asin not in already_seen_asins) and (item_rec_asin not in currently_shown):
        # check to make sure its a book or a movie
        itemInfo = getInfo(item_rec_asin)
        if itemInfo[0][2] == itemType:
          break
  return itemInfo[0]

def get_doc_rec(keywords):
  if keywords is None:
    keywords = ''
  # parse
  search_keywords = keywords.replace(';',' ').replace(',',' ').lower().split(' ')
  # submit to doc2vec
  word_embedding = np.zeros(300,)
  startChar = ''
  for w in search_keywords:
      if w == '':
        continue
      w = startChar + w
      if w == 'not' or w == '-':
        startChar = '-'
        continue
      else:
        startChar = ''
      if w[0]=='-':
        if w[1:] in word_vocab:
          word_embedding -= wordvecs[w[1:]]
      elif w in word_vocab:
        word_embedding += wordvecs[w]
  # if there was no keyword found, return most popular titles
  if np.count_nonzero(word_embedding)==0:
    recs = pop_recs[0:1000]
  else:
    recs = docvecs.most_similar([word_embedding], topn=1000)
  return recs

#user = 'genna' #add your username here (same as previous postgreSQL)            
#host = 'localhost'
#dbname = 'bookmovie_db'
#db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
#con = None
#con = psycopg2.connect(database = dbname, user = user)

@app.route('/')
def index():
    global user_rating, collab, filepath
    IP = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
    filepath = 'recallapp/data/user_info/' + IP.replace('.','_') + '.npy'
    print(filepath)
    try:
      np.load(filepath)
    except:
      user_rating = 3*np.ones([1,np.shape(collab)[1]])
    return render_template('index.html')

@app.route('/text_recommender', methods=['GET', 'POST'])
def text_recommender():
  # get input

  global key_input, already_seen_asins, currently_shown
  key_input = request.form.get('medium_keywords')
  
  rec_vecs = get_doc_rec(key_input)

  book_info = []
  movie_info = []
  currently_shown = []
  for r in rec_vecs:
    item_info = getInfo(r[0])[0]
    item_info[3] = item_info[3].replace('amp;','')
    item_info[3] = item_info[3].replace('&quot;',"'")
    item_info[3] = item_info[3].replace('&Eacute;','E')

    if item_info[0] not in already_seen_asins:
      if item_info[2] == 'book':
        #print(book_info)
        if len(book_info) < 4:

          book_info.append(item_info + ['https://www.amazon.com/dp/' + r[0]])
          currently_shown.append(r[0])
      elif item_info[2] == 'movie':
        if len(movie_info) < 4:
          movie_info.append(item_info + ['https://www.amazon.com/dp/' + r[0]])
          currently_shown.append(r[0])
      if len(book_info) >=4 and len(movie_info) >= 4:
        break
  
  #movie_info = ''
  # get title out of database
  #book_titles = []
  #for vec in book_vecs:
  #  query = "SELECT * FROM item_names WHERE asin='%s'" % book_vecs[0]
  #  book_titles.append(pd.read_sql_query(query,con))
  return render_template('text_recommender.html', book_recs = book_info, movie_recs = movie_info, medium_keywords=key_input)

@app.route('/receivedInfo', methods=['GET', 'POST'])
def recFunc():
  global user_rating, filepath, already_seen_asins, currently_shown, df_ids
  argument = request.args

  itemId = argument.get('id')
  recVal = argument.get('recVal')
  itemType = argument.get('type')

  # check to see if this particular item is in collab filter 
  if (df_ids['0']==itemId).any():
    id_idx = df_ids[df_ids['0']==itemId].index[0]
    user_rating[0][id_idx] = recVal
    np.save(filepath, user_rating)
    newRec = get_collab_rec(find_sim_profile(user_rating), itemType)
  else:
    rec_vecs = get_doc_rec(key_input)
    for r in rec_vecs:
      item_info = getInfo(r[0])[0]
      if item_info[2] == itemType:
        if (item_info[0] not in already_seen_asins) and (item_info[0] not in currently_shown):
          newRec = item_info
          break

  #print(currently_shown)
  already_seen_asins.append(itemId)
  currently_shown[currently_shown.index(itemId)] = newRec[0]
  product_url = 'https://www.amazon.com/dp/' + newRec[0]
  #print(currently_shown)

  response = jsonify({ 'message': 'Data received.', 'asin': newRec[0], 'title': newRec[3], 'img_url': newRec[1], 'product_url': product_url}), 200
  
  return response