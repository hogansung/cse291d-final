# generate data to do clustering
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import numpy as np
import os

# load data w/o review; parse date
dat = pd.read_csv('yelp_academic_dataset_review_nr.csv',parse_dates=['date'])
print len(dat)
#print dat
# it generates 4153150 rows x 9 columns

# load business data
meta = pd.read_csv('yelp_academic_dataset_business.csv')

# remain only 'Restaurants'
print len(dat.business_id.unique())
print dat.business_id.unique()

restaurant_id = set([i for i, c in meta.loc[pd.notnull(meta.categories),['business_id','categories']].values if 'Restaurants' in c])

print len(restaurant_id)

dat = dat[dat.business_id.isin(restaurant_id)]
print len(dat)

# filter based on item counts
ITEM_THRES = 60
item_count = dat.business_id.value_counts()
print 'number of items: %d' % len(item_count)

dat = dat.set_index('business_id', drop=False)
dat['item_count'] = item_count
dat = dat.loc[dat['item_count'] >= ITEM_THRES]

# filter based on user counts
USER_THRES = 5

user_count = dat.user_id.value_counts()
print 'number of users: %d' % len(user_count)
#print user_count

dat = dat.set_index('user_id', drop=False)
dat['user_count'] = user_count
#print dat
dat = dat.loc[dat['user_count'] >= USER_THRES]
#print dat

# unique items and users
uni_item = dat.business_id.unique()
NI = len(uni_item)
uni_user = dat.user_id.unique()
NU = len(uni_user)
print 'number of items : %d' % NI
print 'number of users : %d' % NU


# write lists of users
with open('userList_60', 'w') as f:
    f.write(str(len(uni_user)) + '\n')
    f.write('\n'.join(uni_user))

# write lists of items
with open('itemList_60', 'w') as f:
    f.write(str(len(uni_item)) + '\n')
    f.write('\n'.join(uni_item))

# create mapping for item
item_map = dict()
for idx, item_id in enumerate(uni_item):
    item_map[item_id] = idx
    
# create mapping for user
user_map = dict()
for idx, user_id in enumerate(uni_user):
    user_map[user_id] = idx



# create dataset in tuple
tn = np.array([(user_map[u], item_map[i], r, t.year, t.month, t.day, t.dayofweek) \
               for u, i, r, t in dat.loc[:,['user_id', 'business_id', 'stars', 'date']].values \
               if t.year <= 2015])
tt = np.array([(user_map[u], item_map[i], r, t.year, t.month, t.day, t.dayofweek) \
               for u, i, r, t in dat.loc[:,['user_id', 'business_id', 'stars', 'date']].values \
               if t.year == 2016])

# write tn as tuple
with open('tnList_60', 'w') as f:
    f.write(str(tn.shape[0]) + '\n')
    f.write('\n'.join([','.join(map(str,e)) for e in tn]))
    
# write tn as tuple
with open('ttList_60', 'w') as f:
    f.write(str(tt.shape[0]) + '\n')
    f.write('\n'.join([','.join(map(str,e)) for e in tt]))

print tn.shape
print tt.shape

