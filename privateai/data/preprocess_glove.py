from keras.preprocessing.text import Tokenizer, tokenizer_from_json
import os
import pickle
import numpy as np
import keras
from keras.layers import Embedding
from preprocess import load_raw_text, load_labels
import preprocess

# GLOBALS

# nubmer of words
MAX_WORDS = 20000
# dimension of the embeddings
EMBEDDING_DIM = 100

# where is all our data
DATA_HOME = preprocess.DATA_HOME
# set this to point to the folder that contains the pretrained glove embeddings
GLOVE_DIR = '/home/rpodschwadt1/workspace/glove/glove.6B'

# tokinzer file pattern. the number should be replaced with max words.
# must not be used with out formating it
__TOKENIZER_FILE = 'tokenizer_{}.json'
# directory to save all of our hard work
SAVE_DIR = 'glove'
# files that the sequnences get stored in
# must not be used with out formating it
__SEQUNECES_FILE = 'sequnce_{}.bin'
# files that the numpy data arrays are stored in
# must not be used with out formating it
# formatting is MAX_WORDS, SEQUNCE_LENGHTS
__NUMPY_FILE = 'data_{}_{}.npz'
# file that stores the embedding matrix
# must not be used with out formating it
# formatting is MAX_WORDS, SEQUNCE_LENGHTS
__EMBEDDING_MATRIX_FILE = 'embedding_{}_{}.npz'


def stick_everything_into_cwd():
  """
  Sets all the paths to empty so all files are looked for in the current directory.
  Not recommended on local machines. but makes things easier on colab
  """
  global DATA_HOME
  global GLOVE_DIR
  global SAVE_DIR

  DATA_HOME = ''
  GLOVE_DIR = ''
  SAVE_DIR = ''


# make sure the correct directories exists
def check_dirs():
  if not os.path.exists( os.path.join( DATA_HOME, SAVE_DIR ) ):
    print( 'could not find output directory. creating...' )
    os.mkdir( os.path.join( DATA_HOME, SAVE_DIR ) )
    print( 'created:', os.path.join( DATA_HOME, SAVE_DIR ) )


def load_tokenizer( texts=None, num_words=MAX_WORDS ):
  file = os.path.join( DATA_HOME, SAVE_DIR, __TOKENIZER_FILE.format( num_words ) )
  # tokenizer config file exists. load it and return tokenizer
  if os.path.exists( file ):
    print( 'loading tokenizer' )
    with open( file, 'r' ) as f:
      return tokenizer_from_json( f.readline() )

  if texts is None:
    texts, _ = load_raw_text()  # load the review data
  tokenizer = Tokenizer( num_words=MAX_WORDS )
  print( 'fitting tokenizer' )
  tokenizer.fit_on_texts( texts )
  json = tokenizer.to_json()
  print( 'saving tokenizer' )
  with open( file, 'w' ) as f:
    f.write( json )

  return tokenizer


def load_sequences( texts=None, num_words=MAX_WORDS ):
  """
  Loads or computes the data as sequences and returns it
  """
  file = os.path.join( DATA_HOME, SAVE_DIR, __SEQUNECES_FILE.format( num_words ) )
  if os.path.exists( file ):
    seq = pickle.load( open( file, 'rb' ) )
  else:
    # could not find a saved file
    # need to compute it

    # load the review data
    if texts is None:
      texts, _ = load_raw_text()

    # load a tokenizer
    tokenizer = load_tokenizer( texts=texts , num_words=num_words )

    print( 'transforming texts to sequences' )
    seq = tokenizer.texts_to_sequences( texts )

    word_index = tokenizer.word_index
    print( 'Found %s unique tokens.' % len( word_index ) )

    print( 'saving sequences to file' )
    pickle.dump( seq, open( file, 'wb' ) )

  return seq


def load_embedding_layer( num_words=MAX_WORDS, max_sequences_length=200 ):
  """
  Returns an instace of keras.layers.Embedding with weights already loaded
  """

  file = os.path.join( DATA_HOME, SAVE_DIR, __EMBEDDING_MATRIX_FILE.format( num_words, max_sequences_length ) )
  if os.path.exists( file ):
    embedding_matrix = np.load( file, allow_pickle=True )
  else:
    print( 'Preparing embedding matrix.' )
    tokenizer = load_tokenizer( num_words=num_words )
    embeddings_index = load_embedding_dict()
    # prepare embedding matrix
    num_words = min( num_words, len( tokenizer.word_index ) + 1 )
    embedding_matrix = np.zeros( ( num_words, EMBEDDING_DIM ) )
    for word, i in tokenizer.word_index.items():
      if i >= num_words:
        continue
      embedding_vector = embeddings_index.get( word )
      if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    np.savez( file, embedding_matrix )

  # load pre-trained word embeddings into an Embedding layer
  # note that we set trainable = False so as to keep the embeddings fixed
  embedding_layer = Embedding( num_words,
                              EMBEDDING_DIM,
                              embeddings_initializer=keras.initializers.Constant( embedding_matrix ),
                              input_length=max_sequences_length,
                              trainable=False )

  return embedding_layer


def load_data( num_words=MAX_WORDS, max_sequences_length=200, validation_split=0.2, seed=7 ):
  """
  returns (x_train, y_train), (x_val, y_val)
  """

  file = os.path.join( DATA_HOME, SAVE_DIR, __NUMPY_FILE.format( num_words, max_sequences_length ) )
  if os.path.exists( file ):
    print( 'loading data from file' )
    f = np.load( file, allow_pickle=True )
    x = f[ 'x' ]
    x = f[ 'y' ]
  else:
    print( 'processing data' )
    seq = load_sequences( num_words=num_words )
    x = keras.preprocessing.sequence.pad_sequences( seq, maxlen=max_sequences_length )
    print( 'saving to file' )
    y = load_labels()
    # labels are 1-5, need to be 0-4
    y = y - 1
    np.savez_compressed( file, x=x, y=y )

  y = keras.utils.to_categorical( np.asarray( y ) )
  print( 'Shape of data tensor:', x.shape )
  print( 'Shape of label tensor:', y.shape )
  
  # split the data into a training set and a validation set
  indices = np.arange( x.shape[0] )
  np.random.seed( seed )
  np.random.shuffle( indices )
  x = x[ indices ]
  y = y[ indices ]
  num_validation_samples = int( validation_split * x.shape[0] )

  x_train = x[ :-num_validation_samples ]
  y_train = y[ :-num_validation_samples ]
  x_val = x[ -num_validation_samples: ]
  y_val = y[ -num_validation_samples: ]

  return ( x_train, y_train ), ( x_val, y_val )


def load_embedding_dict():
  """  
  Returns a python dictonary containing a words to embeddings mapping
  """
  # first, build index mapping words in the embeddings set
  # to their embedding vector
  print( 'Indexing word vectors.' )
  embeddings_index = {}
  with open( os.path.join( GLOVE_DIR, 'glove.6B.100d.txt' ) ) as f:
    for line in f:
      word, coefs = line.split( maxsplit=1 )
      coefs = np.fromstring( coefs, 'f', sep=' ' )
      embeddings_index[ word ] = coefs

  return embeddings_index


if __name__ == '__main__':
  load_data()
  load_embedding_layer()

