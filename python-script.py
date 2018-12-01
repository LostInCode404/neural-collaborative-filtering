#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[142]:


# Import
import random
import heapq
import numpy as np
from tqdm import tqdm
from keras.models import Model, load_model, save_model
from keras.layers import Dense, BatchNormalization, Embedding, Input, Flatten, Dropout, merge
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
from keras.losses import binary_crossentropy, mean_absolute_error, mean_squared_error
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


# # Read And Pre-Process Data

# In[78]:


# Constants
DATA_DIR = "./data/"
RATINGS_FILE = "train"
USERS_FILE = "users.dat"
MOVIES_FILE = "movies.dat"
USER_INDEX = 0
ITEM_INDEX = 1
NUM_NEG_SAMPLES = 100


# In[79]:


# Read users
def read_raw_data(path):
    
    # Data array
    data = []
    
    # Open file
    k = 0
    with open(path) as f:
        
        # Read line by line
        lines = f.readlines()
        
        # Loop each line
        for line in lines:
            
            # Parse line
            line = line.strip().split('\t')
            
            # Append to array
            data.append([int(line[0]), int(line[1])])
            
            # Break
#             k += 1
#             if(k==10000):
#                 break
            
    # Return 
    return data


# In[80]:


# Get max user index
def get_item_index_range(path):
    
    # Open file
    with open(path, encoding='ISO-8859-1') as f:
        
        # Read lines
        lines = f.readlines()
        
        # All ids
        ids = []
        
        # Loop lines
        for line in lines:
            
            # Get ID
            user_id = int(line.strip().split('::')[0])
            ids.append(user_id)
            
    # Return
    return min(ids), max(ids)


# In[81]:


# Get max user index
def get_user_index_range(path):
    
    # Open file
    with open(path) as f:
        
        # Read lines
        lines = f.readlines()
        
        # All ids
        ids = []
        
        # Loop lines
        for line in lines:
            
            # Get ID
            user_id = int(line.strip().split('::')[0])
            ids.append(user_id)
            
    # Return
    return min(ids), max(ids)


# In[82]:


# Make user_item mapping
def make_user_item_map(data):
    
    # Map
    user_item_map = {}
    
    # Loop data
    for datum in data:
        
        # Get data
        user = datum[USER_INDEX]
        item = datum[ITEM_INDEX]
        
        # Add to map
        if(user not in user_item_map):
            user_item_map[user] = set()
        user_item_map[user].add(item)
        
    # Return set
    return user_item_map


# In[83]:


# Make positive samples
def make_positive_samples(user_item_map):
    
    # Final data
    user_data = []
    item_data = []
    
    # Loop the map
    for user in user_item_map.keys():
        
        # Loop all items
        for item in user_item_map[user]:
            
            # Add data to array
            user_data.append(user)
            item_data.append(item)
            
    # Return data
    return user_data, item_data


# In[87]:


# Make negative samples
def make_negative_samples(user_item_map):
    
    # Final data
    user_data = []
    item_data = []
    
    # Load range
#     min_val, max_val = get_item_index_range(DATA_DIR+MOVIES_FILE)
    
    # Loop the map
    for user in user_item_map.keys():
        
        # Loop for negative samples
        uniq_set = set()
        for i in range(NUM_NEG_SAMPLES):
            
            # Generate random movie
            while(True):
                
                # Generate
                item = random.randint(0, 3952)
                
                # Check
                if(item not in user_item_map[user] and item not in uniq_set):
                    user_data.append(user)
                    item_data.append(item)
                    uniq_set.add(item)
                    break
                    
    # Return
    return user_data, item_data


# In[85]:


# Split data
def train_test_split_data(positive_data, negative_data):
    
    # Separate data
    pos_user_data, pos_item_data = positive_data
    neg_user_data, neg_item_data = negative_data
    user_data = pos_user_data + neg_user_data
    item_data = pos_item_data + neg_item_data
    labels = [1]*len(pos_user_data) + [0]*len(neg_user_data)
    
    # Split
    user_data_train, user_data_test, item_data_train, item_data_test, labels_train, labels_test = train_test_split(user_data, item_data, labels, test_size=0.05)
    
    # Convert to np array
    user_data_train = np.array(user_data_train, dtype=np.int32)
    item_data_train = np.array(item_data_train, dtype=np.int32)
    labels_train = np.array(labels_train, dtype=np.int32)
    user_data_test = np.array(user_data_test, dtype=np.int32)
    item_data_test = np.array(item_data_test, dtype=np.int32)
    labels_test = np.array(labels_test, dtype=np.int32)
    
    # Return with labels
    return user_data_train, item_data_train, labels_train, user_data_test, item_data_test, labels_test  


# In[88]:


# Read raw data
raw_data = read_raw_data(DATA_DIR+RATINGS_FILE)

# Make user item map
user_item_map = make_user_item_map(raw_data)

# Make positive samples
pos_samples = make_positive_samples(user_item_map)

# Make positive samples
neg_samples = make_negative_samples(user_item_map)

# Make train and test data
user_data_train, item_data_train, labels_train, user_data_test, item_data_test, labels_test = train_test_split_data(pos_samples, neg_samples)


# In[14]:


# Load test data by paper
paper_user_data = []
paper_item_data = []
paper_lables = []

# Open file
with


# # Model

# In[97]:


# Method to get model
def get_model(user_count, item_count, mf_dim, layer_config):
    
    # Input layers
    user_input = Input(shape=(1,))
    item_input = Input(shape=(1,))
    
    # MF layers for user
    mf_user = Embedding(input_dim=user_count+1, output_dim=mf_dim, embeddings_regularizer=l2(0), embeddings_initializer='normal', input_length=1)(user_input)
    mf_user = Flatten()(mf_user)
    
    # MF layers for item
    mf_item = Embedding(input_dim=item_count+1, output_dim=mf_dim, embeddings_regularizer=l2(0), embeddings_initializer='normal', input_length=1)(item_input)
    mf_item = Flatten()(mf_item)
    
    # Merge MF layers by multiplying
    mf_layer = merge.multiply([mf_user, mf_item])
    
    # MLP layers for user
    mlp_user = Embedding(input_dim=user_count+1, output_dim=32, embeddings_regularizer=l2(0), embeddings_initializer='normal', input_length=1)(user_input)
    mlp_user = Flatten()(mlp_user)
    
    # MLP layers for item
    mlp_item = Embedding(input_dim=item_count+1, output_dim=32, embeddings_regularizer=l2(0), embeddings_initializer='normal', input_length=1)(item_input)
    mlp_item = Flatten()(mlp_item)
    
    # Merge MLP layers by concatenating
    mlp_layer = merge.concatenate([mlp_user, mlp_item])
    
    # Add dense layers to MLP part after concat
    for layer_dim in layer_config:
        
        # Add layer
        mlp_layer = Dense(layer_dim, kernel_regularizer=l2(0), activation='relu')(mlp_layer)
#         mlp_layer = BatchNormalization()(mlp_layer)
        
    # Concat MF and MLP layers
    concat_layer = merge.concatenate([mf_layer, mlp_layer])
    
    # Prediction layer
    output = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform')(concat_layer)
    
    # Define model
    model = Model(inputs=[user_input,item_input], outputs=output)
    
    # Return model
    return model


# In[98]:


# User range
user_min, user_max = 0, 6039
item_min, item_max = 0, 3952


# In[99]:


# Get user neg_samples map
d2ef get_user_neg_map(neg_samples):
    
    # Separate data
    neg_user, neg_item = neg_samples
    
    # Map
    user_neg_map = {}
    
    # Loop data
    for i in range(len(neg_user)):
        
        # Add to map
        if(neg_user[i] not in user_neg_map):
            user_neg_map[neg_user[i]] = set()
        user_neg_map[neg_user[i]].add(neg_item[i])
    
    # Return map
    return user_neg_map


# In[100]:


# Get user neg map
user_neg_map = get_user_neg_map(neg_samples)


# In[107]:


# Get hit ratio
def get_hit_ratio(model):

    # Calculate hit ratio
    hr = []

    # Loop test data
    for i in tqdm(range(len(user_data_test))):

        # Get test data
        user = user_data_test[i]
        item = item_data_test[i]
        if(labels_test[i]==0):
            pass

        # Neg samples
        neg = list(user_neg_map[user])
        neg.append(item)

        # Predict
        preds = model.predict([np.array([user]*len(neg)),np.array(neg)])

        # Prepare ranklist
        ranklist = []
        for i in range(len(neg)):
            ranklist.append((preds[i][0],neg[i]))
        ranklist = sorted(ranklist)
        ranklist = ranklist[-20:]

        # Check if item in top 10
        h = 0
        for val in ranklist:
            if(val[1]==item):
                h = 1
                break
        hr.append(h)
        
    # Return
    return np.mean(hr)


# In[130]:


# Get model
model = get_model(user_max, item_max, 8, [64,32,16,8])

# Optimizer
optimizer = Adam(lr=0.0001)

# Compile model
model.compile(optimizer=optimizer, loss=binary_crossentropy, metrics=['accuracy'])

# Train model
model.fit([user_data_train, item_data_train], labels_train, validation_data=([user_data_test, item_data_test], labels_test), batch_size = 256, epochs = 15)

# Save model
save_model(model, "./models/mf_dim_8")


# In[131]:


get_hit_ratio(model)


# In[132]:


# Get model
model = get_model(user_max, item_max, 16, [64,32,16,8])

# Optimizer
optimizer = Adam(lr=0.0001)

# Compile model
model.compile(optimizer=optimizer, loss=binary_crossentropy, metrics=['accuracy'])

# Train model
model.fit([user_data_train, item_data_train], labels_train, validation_data=([user_data_test, item_data_test], labels_test), batch_size = 256, epochs = 15)

# Save model
save_model(model, "./models/mf_dim_16")


# In[133]:


get_hit_ratio(model)


# In[134]:


# Get model
model = get_model(user_max, item_max, 32, [64,32,16,8])

# Optimizer
optimizer = Adam(lr=0.0001)

# Compile model
model.compile(optimizer=optimizer, loss=binary_crossentropy, metrics=['accuracy'])

# Train model
model.fit([user_data_train, item_data_train], labels_train, validation_data=([user_data_test, item_data_test], labels_test), batch_size = 256, epochs = 15)

# Save model
save_model(model, "./models/mf_dim_32")


# In[135]:


get_hit_ratio(model)


# In[136]:


# Get model
model = get_model(user_max, item_max, 64, [64,32,16,8])

# Optimizer
optimizer = Adam(lr=0.0001)

# Compile model
model.compile(optimizer=optimizer, loss=binary_crossentropy, metrics=['accuracy'])

# Train model
model.fit([user_data_train, item_data_train], labels_train, validation_data=([user_data_test, item_data_test], labels_test), batch_size = 256, epochs = 15)

# Save model
save_model(model, "./models/mf_dim_64")


# In[137]:


get_hit_ratio(model)


# In[138]:


# Get model
model = get_model(user_max, item_max, 128, [64,32,16,8])

# Optimizer
optimizer = Adam(lr=0.0001)

# Compile model
model.compile(optimizer=optimizer, loss=binary_crossentropy, metrics=['accuracy'])

# Train model
model.fit([user_data_train, item_data_train], labels_train, validation_data=([user_data_test, item_data_test], labels_test), batch_size = 256, epochs = 15)

# Save model
save_model(model, "./models/mf_dim_128")


# In[139]:


get_hit_ratio(model)


# In[149]:


plt.plot([8,16,32,64,128],[0.637,0.642,0.666,0.698,0.734])
plt.scatter([8,16,32,64,128],[0.637,0.642,0.666,0.698,0.734])
plt.xlabel("Dimension of latent factors")
plt.ylabel("Hit ratio")
plt.show()

