# Copyright (c) 2017 NVIDIA Corporation
import sys
import datetime
import random
from math import floor

def print_stats(data):
  total_ratings = 0
  print("STATS")
  for user in data:
    total_ratings += len(data[user])
  print("Total Ratings: {}".format(total_ratings))
  print("Total User count: {}".format(len(data.keys())))

def save_data_to_file(data, filename):
  with open(filename, 'w') as out:
    for userId in data:
      for record in data[userId]:
        out.write("{}\t{}\t{}\n".format(userId, record[0], record[1]))

def main(args):
  inpt = args[1]
  out_prefix = args[2]
  percent = 0.7
  user2id_map = dict()
  item2id_map = dict()
  userId = 0
  itemId = 0
  data = dict()

  min_ts = 100000000000
  max_ts = 0
  total_rating_count = 0
  with open(inpt, 'r') as inpt_f: #ratings.csv headers: userId,movieId,rating,timestamp
    for line in inpt_f:
      if 'userId' in line:
        continue
      parts = line.split(',')
      user = int(parts[0])
      item = int(parts[1])
      rating = float(parts[2])
      ts = int(parts[3])
      if min_ts > ts:
        min_ts = ts
      if max_ts < ts:
        max_ts = ts
      if not user in user2id_map:
        user2id_map[user] = userId
        userId += 1
      if not item in item2id_map:
        item2id_map[item] = itemId
        itemId += 1
      total_rating_count += 1
      if user2id_map[user] not in data:
        data[user2id_map[user]] = []
      data[user2id_map[user]].append((item2id_map[item], rating, ts))

  print("STATS")
  print("Total Ratings: {}".format(total_rating_count))
  print("Total User count: {}".format(len(user2id_map)))
  print("Total Item count: {}".format(len(item2id_map)))
  print("Minimum ts: {}, which is {}".format(min_ts, datetime.datetime.fromtimestamp(min_ts).strftime('%Y-%m-%d')))
  print("Maximum ts: {}, which is {}".format(max_ts, datetime.datetime.fromtimestamp(max_ts).strftime('%Y-%m-%d')))

  training_data = dict()
  validation_data = dict()
  test_data = dict()
  train_set_items = set()

  for userId in data.keys():
    if len(data[userId]) < 2:
      #print("WARNING, userId {} has less than 2 ratings, skipping user...".format(userId))
      continue
    time_sorted_ratings = sorted(data[userId], key=lambda x: x[2]) # sort by timestamp
    last_train_ind = floor(percent * len(time_sorted_ratings))
    training_data[userId] = time_sorted_ratings[:last_train_ind]
    for rating_item in time_sorted_ratings[:last_train_ind]:
        train_set_items.add(rating_item[0]) # keep track of items from training set
    p = random.random()
    if p <= 0.5:
      validation_data[userId] = time_sorted_ratings[last_train_ind:]
    else:
      test_data[userId] = time_sorted_ratings[last_train_ind:]

  # remove items not not seen in training set
  for userId, userRatings in test_data.items():
    test_data[userId] = [rating for rating in userRatings if rating[0] in train_set_items]
  for userId, userRatings in validation_data.items():
    validation_data[userId] = [rating for rating in userRatings if rating[0] in train_set_items]    
    
  print("Training Data")
  print_stats(training_data)
  save_data_to_file(training_data, out_prefix+".train")
  print("Validation Data")
  print_stats(validation_data)
  save_data_to_file(validation_data, out_prefix + ".valid")
  print("Test Data")
  print_stats(test_data)
  save_data_to_file(test_data, out_prefix + ".test")



if __name__ == "__main__":
  main(sys.argv)
  
