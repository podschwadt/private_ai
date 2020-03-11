import json
import os
import numpy as np
import pickle
import time
import sys
import matplotlib as mpl
mpl.use( 'Agg' )
import matplotlib.pyplot as plt
import re

# GLOBALS

# set this to point to the folder that contains the data
DATA_HOME = '/home/rpodschwadt1/workspace/data/private_ai'

## Files
# this is the raw json file as it is downloaded
JSON_FILE = 'reviews_Movies_and_TV_5.json'
RAW_TEXT_FILE = 'movies_raw.bin'
LABELS_FILE = 'labels.bin'
TOKENIZED_FILE = 'tokenized.bin'
TOKENIZED_FILES = 'tokenized_{}.bin'
NO_FILES = 17
WORD_FREQUNCIES_DICT_FILE = 'word_freq_dict.bin'

# number of reviews that should be in the file
NO_REVIEWS = 1697533

# regex pattern
REGEX = '[^a-z]'


def parse_json( file ):
  """
  Parses the original json file and extracts the review text and the rating.
  """
  
  reviews = []
  ratings = []
  count = 0
  words = 0
  with open( file, 'r' ) as f:
    start = time.time()
    for line in f.readlines():
      j = json.loads( line )
      words += len( j[ 'reviewText' ].split( ' ' ) )
      reviews.append( j[ 'reviewText' ] )
      ratings.append( int( j[ 'overall' ] ) )
      count += 1
      if count % 10000 == 0:
        sys.stdout.write( '\r processed: {}/{} reviews in {}s'.format( count, NO_REVIEWS, time.time() - start ) )
    sys.stdout.write( '\r processed: {}/{} reviews in {}s\n'.format( count, NO_REVIEWS, time.time() - start ) )

    print( 'total number of words:', words )
    print( 'avg words/sample:', words / NO_REVIEWS )

  return reviews, ratings


def rating_distribution( ratings, plot=True ):
  """
  Expects a numpy arrays containing ratings. 
  Returns a numpy array with the number of times with which each rating is represented. 
  """
  possible_ratings = np.unique( ratings )
  dist = []
  for pr in possible_ratings:
    dist.append( np.count_nonzero( ratings == pr ) )
  dist = np.array( dist )
  print( 'distribution of ratings', dist )
  if plot:
    plt.bar( possible_ratings, dist )
    plt.savefig( 'rating_distribution.pdf' )

    
def preprocessor( text, delimiter=' ' ):
  """
  Takes a string splits it at the delimiter and returns a list of strings that:
    - are all lower case
    - have no non-alpha characters
    - no empty strings
  """
  # lower case and split into tokens, remove any trailing whitespaces
  tokens = text.lower().rstrip().split( delimiter )
  # remove non alpha charcaters
  tokens = [ re.sub( REGEX, '', token ) for token in tokens ]
  # remove anything empty
  tokens = [ token for token in tokens if len( token ) > 0 ]

  return tokens


def word_frequency( tokenized, dic ):
  """
  Takes a list of tokenized texts. Counts the number of words and puts it in the dictonary `dict`.
  """
  print( 'computing word frequencies' )
  start = time.time()
  for i, text in enumerate( tokenized ):
    for token in text:
      if token not in dic:
        dic[ token ] = 1
      else:
        dic[ token ] += 1
    if i % 10000 == 0:
      sys.stdout.write( '\rprocessed : {}/{} reviews in {}s'.format( i, NO_REVIEWS, time.time() - start ) )
  sys.stdout.write( '\rprocessed : {}/{} reviews in {}s\n'.format( i, NO_REVIEWS, time.time() - start ) )


def tokenize_and_save( inputs ):
  """
  Runs the tokenizer and saves the results into pickle files with 100k instances each.
  (Done so I can run this on my poor machine)
  """
  # split into words
  # split on whitespaces and make it all lowercase
  print( 'performing word tokenization' )
  tokenized = []
  file_count = 0
  start = time.time()
  for i, r in enumerate( inputs ):
      tokenized.append( preprocessor( r ) )
      if i % 10000 == 0:
        sys.stdout.write( '\r tokenized: {}/{} reviews in {}s'.format( i, NO_REVIEWS, time.time() - start ) )
      if i != 0 and i % 100000 == 0:
        sys.stdout.write( '\n' )
        print( 'saving tokinzed data to file' )
        pickle.dump( tokenized, open( os.path.join( DATA_HOME, TOKENIZED_FILES.format( file_count ) ), 'wb' ) )
        file_count += 1
        tokenized = []
      
  sys.stdout.write( '\r processed: {}/{} reviews in {}s\n'.format( i, NO_REVIEWS, time.time() - start ) )
  print( 'saving tokinzed data to file' )
  pickle.dump( tokenized, open( os.path.join( DATA_HOME, TOKENIZED_FILES.format( file_count ) ), 'wb' ) )
  

def load_labels():
  if not os.path.exists( os.path.join( DATA_HOME, LABELS_FILE ) ):
    print( 'no prior files found. staring from scratch' )
    _, rat = parse_json( os.path.join( DATA_HOME, JSON_FILE ) )
    y = np.array( rat )
    print( 'saving data to files' )
    pickle.dump( y , open( os.path.join( DATA_HOME, LABELS_FILE ), 'wb' ) )
  else:
    print( 'found raw text and labes. loading...' )
    y = pickle.load( open( os.path.join( DATA_HOME, LABELS_FILE ), 'rb' ) )
    print( 'done' )

  return y


def load_raw_text():
  """
  Extra the raw texts and labels.
  
  Returns list of the reviews, labels
  """
  if not os.path.exists( os.path.join( DATA_HOME, RAW_TEXT_FILE ) ) or \
   not os.path.exists( os.path.join( DATA_HOME, LABELS_FILE ) ):
    print( 'no prior files found. staring from scratch' )
    rev, rat = parse_json( os.path.join( DATA_HOME, JSON_FILE ) )
    y = np.array( rat )
    print( 'saving data to files' )
    pickle.dump( rev , open( os.path.join( DATA_HOME, RAW_TEXT_FILE ), 'wb' ) )
    pickle.dump( y , open( os.path.join( DATA_HOME, LABELS_FILE ), 'wb' ) )
  else:
    print( 'found raw text and labes. loading...' )
    rev = pickle.load( open( os.path.join( DATA_HOME, RAW_TEXT_FILE ), 'rb' ) )
    y = pickle.load( open( os.path.join( DATA_HOME, LABELS_FILE ), 'rb' ) )
    print( 'done' )
    
  return rev, y
    
  

#########################
# Data processing block #
#########################
if __name__ == '__main__':
  # perform tokenization if necessary
  if not os.path.exists( os.path.join( DATA_HOME, TOKENIZED_FILES.format( 0 ) ) ):
    print( 'no tokinzed files found.' )
    # read json file if necessary
    if not os.path.exists( os.path.join( DATA_HOME, RAW_TEXT_FILE ) ) or \
     not os.path.exists( os.path.join( DATA_HOME, LABELS_FILE ) ):
      print( 'no prior files found. staring from scratch' )
      rev, rat = parse_json( os.path.join( DATA_HOME, JSON_FILE ) )
      y = np.array( rat )
      pickle.dump( rev , open( os.path.join( DATA_HOME, RAW_TEXT_FILE ), 'wb' ) )
      pickle.dump( y , open( os.path.join( DATA_HOME, LABELS_FILE ), 'wb' ) )
    else:
      print( 'found raw text and labes. loading...' )
      rev = pickle.load( open( os.path.join( DATA_HOME, RAW_TEXT_FILE ), 'rb' ) )
      y = pickle.load( open( os.path.join( DATA_HOME, LABELS_FILE ), 'rb' ) )
      print( 'done' )
    tokenize_and_save( rev )
    del rev
    rev = None

  # not pretty but is its research right?
  print( 'found tokinzed data. loading....' )
  tokenized = []
  for i in range( NO_FILES ):
    tokenized.extend( pickle.load( open( os.path.join( DATA_HOME, TOKENIZED_FILES.format( i ) ), 'rb' ) ) )
  y = pickle.load( open( os.path.join( DATA_HOME, LABELS_FILE ), 'rb' ) )

  #########################
  # Data analytics block  #
  #########################
  rating_distribution( y )
  if not os.path.exists( os.path.join( DATA_HOME, WORD_FREQUNCIES_DICT_FILE ) ):
    freq_dic = {}
    word_frequency( tokenized, freq_dic )
    print( 'writing word frequncy dictonary to file' )
    pickle.dump( freq_dic, open( os.path.join( DATA_HOME, WORD_FREQUNCIES_DICT_FILE ), 'wb' ) )
  else:
    print( 'loading word frequncy dictonary' )
    freq_dic = pickle.load( open( os.path.join( DATA_HOME, WORD_FREQUNCIES_DICT_FILE ), 'rb' ) )

  # there is probably a smarter way of doing this
  words = list( freq_dic.keys() )
  frequencies = [ freq_dic[ w ] for w in words ]

  # sort the lists
  frequencies, words = ( list( t ) for t in zip( *sorted( zip( frequencies, words ), reverse=True ) ) )

  print( 'top 100 words:' )
  for i, ( word, freq ) in enumerate( zip( words, frequencies ) ):
    print( word, freq )
    if i == 100:
      break
