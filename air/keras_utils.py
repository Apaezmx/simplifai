import json
import re
import numpy as np
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers.core import Dense, Dropout, Activation, Reshape, Flatten, Merge
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop
from keras.utils import np_utils

def striphtml(data):
    p = re.compile(r'<.*?>')
    return p.sub('', data)

nb_classes = 3
nb_epoch = 20
batch_size = 100
interest_map = {
  'high': [1, 0, 0],
  'medium': [0, 1, 0],
  'low': [0, 0, 1],
}

with open('train.json') as data_file:
    trdata = json.load(data_file)

with open('test.json') as data_file:
    tedata = json.load(data_file)

train_length = len(trdata['listing_id'].keys())
data = trdata
for feature_name, value_dict in tedata.iteritems():
  tr_last_index =max([int(x) for x in trdata[feature_name].keys()])
  for rid, value in value_dict.iteritems():
    data[feature_name][str(int(rid)+tr_last_index + 1)] = value

# To rows
features = []
indexes = [int(x) for x in data['listing_id'].keys()]
feature_index = {v: i for i, v in enumerate(data.keys())}
print data.keys()
for i in sorted(indexes):
  row = []
  for k in data.keys():
    if k == 'description':
      row.append(text_to_word_sequence(striphtml(data[k][str(i)])))
    elif 'address' in k:
      row.append(text_to_word_sequence(data[k][str(i)]))
    elif k == 'features':
      row.append([feature.lower() for feature in data[k][str(i)]])
    elif k == 'interest_level':
      if str(i) in data[k]:
        row.append(data[k][str(i)])
      else:
        row.append('low')
    else:
      row.append(data[k][str(i)])
  features.append(row)

print len(features), features[0]

def process_text_feature(feature):
  word_freq = {}
  for f in feature:
    if isinstance(f, list):
      for word in f:
        word_freq[word] = 1 if word not in word_freq else word_freq[word] + 1
    else:
      word_freq[f] = 1 if f not in word_freq else word_freq[f] + 1
  words = {word: i for i, word in enumerate(sorted(word_freq, key=lambda i: -int(word_freq[i])))}
  counts = [word_freq[word] for word in words.keys()]
  return words, counts

desc_dict, _ = process_text_feature([x[feature_index['description']] for x in features])
address_features = [x[feature_index['display_address']] for x in features]
address_features.extend([x[feature_index['street_address']] for x in features])
address_dict, _ = process_text_feature(address_features)
apt_features_dict, _ = process_text_feature([x[feature_index['features']] for x in features])
manager_id_dict, _ = process_text_feature([x[feature_index['manager_id']] for x in features])
building_id_dict, _ = process_text_feature([x[feature_index['building_id']] for x in features])

# Replace string features for integers.
description_size = 500  # Upto lists of 700 words...
address_size = 80
feature_size = 40
for row in features:
  row[feature_index['description']] = pad_sequences([[desc_dict[word] for word in row[feature_index['description']]]], maxlen=description_size, padding='post', truncating='post')[0]
  row[feature_index['display_address']] = pad_sequences([[address_dict[word] for word in row[feature_index['display_address']]]], maxlen=address_size, padding='post', truncating='post')[0]
  row[feature_index['street_address']] = pad_sequences([[address_dict[word] for word in row[feature_index['street_address']]]], maxlen=address_size, padding='post', truncating='post')[0]
  row[feature_index['features']] = pad_sequences([[apt_features_dict[word] for word in row[feature_index['features']]]], maxlen=feature_size, padding='post', truncating='post')[0]
  row[feature_index['manager_id']] = manager_id_dict[row[feature_index['manager_id']]]
  row[feature_index['building_id']] = building_id_dict[row[feature_index['building_id']]]
  row[feature_index['price']] = row[feature_index['price']] / 15000.
  row[feature_index['bedrooms']] = row[feature_index['bedrooms']] / 10.
  row[feature_index['bathrooms']] = row[feature_index['bathrooms']] / 10.
  row[feature_index['longitude']] = row[feature_index['longitude']] / 90.
  row[feature_index['latitude']] = row[feature_index['latitude']] / 90.

print 'Integerized strings ', len(features), features[0]

description_embedding_size = 20
address_embedding_size = 5
feature_embedding_size = 5
building_id_embedding_size = 3
manager_id_embedding_size = 3
listing_id_embedding_size = 3
model1 = Sequential()
model1.add(Embedding(len(desc_dict.keys()), description_embedding_size, input_length=description_size))
model1.add(Flatten())
model1.add(Dropout(0.4))

model2 = Sequential()
model2.add(Embedding(len(address_dict.keys()), address_embedding_size, input_length=address_size))
model2.add(Flatten())
model2.add(Dropout(0.4))

model3 = Sequential()
model3.add(Embedding(len(address_dict.keys()), address_embedding_size, input_length=address_size))
model3.add(Flatten())
model3.add(Dropout(0.4))

model4 = Sequential()
model4.add(Embedding(len(apt_features_dict.keys()), feature_embedding_size, input_length=feature_size))
model4.add(Flatten())
model4.add(Dropout(0.4))

model5 = Sequential()
model5.add(Embedding(len(manager_id_dict.keys()), manager_id_embedding_size, input_length=1))
model5.add(Flatten())
model5.add(Dropout(0.4))

model6 = Sequential()
model6.add(Embedding(len(building_id_dict.keys()), building_id_embedding_size, input_length=1))
model6.add(Flatten())
model6.add(Dropout(0.4))

model7 = Sequential()
model7.add(Embedding(len(set([row[feature_index['listing_id']] for row in features])), listing_id_embedding_size, input_length=1))
model7.add(Flatten())
model7.add(Dropout(0.4))

model8 = Sequential()
model8.add(Dense(5, input_shape=(5,)))
model8.add(Activation('relu'))
model8.add(Dropout(0.4))

merged_model = Sequential()
merged_model.add(Merge([model1, model2, model3, model4, model5, model6, model7, model8], mode='concat', concat_axis=1))
concat_size = description_embedding_size * description_size + 2 * address_embedding_size * address_size + feature_embedding_size * feature_size + manager_id_embedding_size + building_id_embedding_size + listing_id_embedding_size + 5
merged_model.add(Dense(2048, input_shape=(concat_size,)))
merged_model.add(Activation('relu'))
merged_model.add(Dropout(0.4))
merged_model.add(Dense(1024))
merged_model.add(Activation('relu'))
merged_model.add(Dropout(0.4))
merged_model.add(Dense(nb_classes))
merged_model.add(Activation('softmax'))

merged_model.summary()

merged_model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
usable_features = [u'description', u'display_address', u'street_address', u'features', u'manager_id', u'building_id', u'listing_id', [u'price', u'bedrooms', u'longitude', u'latitude', u'bathrooms']]
X_train = []
Y_train = []
X_test = []
Y_test = []
test_percentage = 0.05
test_idx = int((1 - test_percentage) * train_length)
print('Test idx', test_idx)
for key in usable_features:
  if isinstance(key, list):
    # Group all float features together
    arr = []
    for row in features[:test_idx]:
      feats = []
      for k in key:
        feats.append(row[feature_index[k]])
      arr.append(feats)
    X_train.append(np.array(arr))
  else:
    X_train.append(np.array([row[feature_index[key]] for row in features[:test_idx]]))

Y_train = np.array([interest_map[row[feature_index['interest_level']]] for row in features[:test_idx]])

for key in usable_features:
  if isinstance(key, list):
    # Group all float features together
    arr = []
    for row in features[test_idx:train_length]:
      feats = []
      for k in key:
        feats.append(row[feature_index[k]])
      arr.append(feats)
    X_test.append(np.array(arr))
  else:
    X_test.append(np.array([row[feature_index[key]] for row in features[test_idx:train_length]]))

Y_test = np.array([interest_map[row[feature_index['interest_level']]] for row in features[test_idx:train_length]])

print([arr.shape for arr in X_train], 'train samples')
print([arr.shape for arr in X_test], 'test samples')

# convert class vectors to binary class matrices
print(Y_train.shape, 'train labels')
print(Y_test.shape, 'test labels')
history = merged_model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test))
score = merged_model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

X_test = []
for key in usable_features:
  if isinstance(key, list):
    # Group all float features together
    arr = []
    for row in features[train_length:]:
      feats = []
      for k in key:
        feats.append(row[feature_index[k]])
      arr.append(feats)
    X_test.append(np.array(arr))
  else:
    X_test.append(np.array([row[feature_index[key]] for row in features[train_length:]]))

print([arr.shape for arr in X_test], 'output shape')
output = merged_model.predict(X_test)

with open('output.csv', 'w') as f:
  f.write('listing_id,high,medium,low\n')
  i = 0
  for row in output:
    f.write(str(X_test[6][i][0]) + ',' + ','.join([str(val) for val in row]) + '\n')
    i += 1
  f.close()



