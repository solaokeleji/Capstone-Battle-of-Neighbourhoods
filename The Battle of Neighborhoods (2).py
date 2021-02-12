#!/usr/bin/env python
# coding: utf-8

# # The Battle of Neighborhoods

# This project aims to identify the best location for setting up a coffee shop in the City of Toronto

# ### Import the necessary libraries

# In[3]:


#import libraries
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.mode.chained_assignment = None  # default='warn'

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import matplotlib.colors as colors

get_ipython().system('pip install geocoder')
import geocoder # import geocoder

#!conda install -c conda-forge geopy --yes 
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# import k-means from clustering stage
from sklearn.cluster import KMeans

#!conda install -c conda-forge folium=0.5.0 --yes 
import folium # map rendering library


print('Libraries imported.')


# In[4]:


#read first table from wikipedia page

df_canada= pd.read_html('https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M')


# In[5]:


#check the total number of tables in the page

print(f'Total tables: {len(df_canada)}')


# In[6]:


#to make the table selection easier, use the match parameter to select a subset of tables.

df_canada = pd.read_html('https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M', match='Borough')
len(df_canada)


# In[7]:


#there is a match! Extract the matched table which has been stored in df_canada

df = df_canada[0]
df.head()


# In[8]:


#drop rows where borough is Not assigned
df.drop(df.loc[df['Borough']=='Not assigned'].index, inplace=True)

#reset the index numbering
df.reset_index(drop=True, inplace=True)
df.head()


# In[9]:


#display the number of rows in the dataframe
df.shape


# #### Import the geographical coordinates of each postal code

# In[10]:


#read dataframe with geographical coordinates of each postal code
df2= pd.read_csv('http://cocl.us/Geospatial_data')


#merge the two dataframes (boroughs and geographical coordinates) to show geographical coordinates of each postal code
df_new = df.merge(df2, on='Postal Code')
df_new.head(5)


# ### Explore Neighborhoods in Toronto

# In[11]:


print('The dataframe has {} boroughs and {} neighbourhoods.'.format(
        len(df_new['Borough'].unique()),
        df_new.shape[0]
    )
)


# #### Create a map of Toronto with neighborhoods superimposed on top

# Use geopy library to obtain the latitude and longitude values of Toronto

# In[12]:


address = 'Toronto'

geolocator = Nominatim(user_agent="ny_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Toronto are {}, {}.'.format(latitude, longitude))


# #### Create map of Toronto using latitude and longitude values

# In[79]:


from IPython.display import display
map_toronto = folium.Map(location=[latitude, longitude], zoom_start=12)

# add markers to map
for lat, lng, borough, neighbourhood in zip(df_new['Latitude'], df_new['Longitude'], df_new['Borough'], df_new['Neighbourhood']):
    label = '{}, {}'.format(neighbourhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto)  
    
display(map_toronto)


# #### Define Foursquare Credentials and Version

# In[14]:


CLIENT_ID = '2LFZEE5KY44WGBFIAOZBAJUJRUN1VRYJTVNGW3GW4URRSPEK' 
CLIENT_SECRET = 'SZ0KO1GZVONF0HUDFI0P3X3GEVZTWOPQJZHLGDTG0VAXZZCJ' 
ACCESS_TOKEN = 'L10JDGKBDSWTLXHGFG1RBL5WMBPA5XAIHH03MSI2MTR3DAP4' 
VERSION = '20180604'
LIMIT = 100

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# ### Let's start Exploring Neighborhoods in Toronto

# In[15]:


LIMIT = 100
def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(CLIENT_ID,CLIENT_SECRET, VERSION, lat, lng, radius, LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighbourhood', 
                  'Neighbourhood Latitude', 
                  'Neighbourhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# Let the above function run on each neighborhood and save in a new dataframe called toronto_venues.

# In[16]:


toronto_venues = getNearbyVenues(names=df_new['Neighbourhood'],latitudes=df_new['Latitude'], longitudes=df_new['Longitude'])


# In[17]:


#how many venues were returned
print(toronto_venues.shape)
toronto_venues.head()


# How many venues were returned for each neighborhood?

# In[18]:


toron= toronto_venues.groupby('Neighbourhood').count().sort_values(by= 'Venue Category',ascending=False, axis=0).reset_index()
toron.head()


# Visualize neighborhoods with more than 45 venues
# 

# In[19]:


#create new dataframe
toron2= toron.loc[toron['Venue Category'] >= 45]
get_ipython().run_line_magic('matplotlib', 'inline')

toron2.plot(kind= 'bar', x="Neighbourhood", y="Venue Category", rot=70, figsize= (10,6), color='darkblue', fontsize= 8)

plt.title('Neighbourhoods in Toronto with more than 45 venues', fontsize=15)

plt.xlabel('Neighbourhoods', fontsize= 12)
plt.ylabel('Venues', fontsize= 12)


plt.show()


# How many unique categories can be curated from all the returned venues?

# In[20]:


print('There are {} uniques venue categories.'.format(len(toronto_venues['Venue Category'].unique())))


# In[21]:


toronto_venues['Venue Category'].unique()


# ## Analyze Each Neighborhood

# Now, let's analyze each of the 103 neighborhoods

# In[22]:


# one hot encoding
toronto_onehot = pd.get_dummies(toronto_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighbourhood column back to dataframe
toronto_onehot['Neighbourhood'] = toronto_venues['Neighbourhood'] 

# move neighbourhood column to the first column
fixed_columns = [toronto_onehot.columns[-1]] + list(toronto_onehot.columns[:-1])
toronto_onehot = toronto_onehot[fixed_columns]

toronto_onehot.head()


# Examine the new dataframe above

# In[23]:


toronto_onehot.shape


# #### Group rows by neighborhood and by taking the mean of the frequency of occurrence of each category

# In[24]:


toronto_grouped = toronto_onehot.groupby('Neighbourhood').mean().reset_index()
toronto_grouped.head()


# The dataframe above shows the frequecy of the venues present in 97 neighborhoods

# Confirm the new size after grouping

# In[25]:


toronto_grouped.shape


# In[26]:


toronto_grouped_count = toronto_onehot.groupby('Neighbourhood').sum().reset_index()
toronto_grouped_count


# In[27]:


toronto_grouped_count['Coffee Shop'].sum()


# #### Let's check the number of coffee shops currently in each neighborhood

# Also sort the data to know which areas has the most number of coffee shops and further explore the reasons behind this

# In[28]:


coffee_shop= toronto_grouped_count.loc[:, ["Neighbourhood", "Coffee Shop"]].sort_values(by= 'Coffee Shop',ascending=False, axis=0)
coffee_shop.head()


# In[29]:


#include the boroughs, too
df_borough= df_new[['Borough', 'Neighbourhood']]

#create a new dataframe to show the number of coffee shops in the boroughs/neighborhoods
df_coffee_shops = df_borough.merge(coffee_shop, on='Neighbourhood').sort_values(by= 'Coffee Shop',ascending=False, axis=0)

#remove boroughs which do not have coffee shops
df_coffee_shops =df_coffee_shops.loc[df_coffee_shops['Coffee Shop'] != 0]
df_coffee_shops.head()


# Let's compare the number of coffee shops in a neighbourhood to the total number of other venues in that neighbourhood

# In[30]:


df_coffee_shops.reset_index()
df_coffee_shop= df_coffee_shops[['Neighbourhood', 'Coffee Shop']]. reset_index()
toronto= toron[['Neighbourhood', 'Venue Category']].reset_index()

df_coffee_neigh= df_coffee_shop.merge(toronto, on='Neighbourhood')
#drop the columns not required
df_coffee_neigh.drop(columns=['index_x', 'index_y'], inplace=True)

#reset the index numbering
df_coffee_neigh.reset_index(drop=True, inplace=True)
df_coffee_neigh['Proportion %']= df_coffee_neigh['Coffee Shop']/df_coffee_neigh['Venue Category']*100
df_coffee_neigh.head()


# In[31]:


#stacked bar chart of neighbourhoods with more than 5 coffee shops

#create new dataframe
neighbour= df_coffee_neigh.loc[df_coffee_neigh['Coffee Shop']>=5]

neigh=neighbour[['Neighbourhood', 'Coffee Shop', 'Venue Category']]
neigh.rename(columns={'Venue Category':'Number of Venues'}, inplace=True)
neigh.plot(kind='bar', x="Neighbourhood", stacked=True, rot=70, figsize= (10,6), fontsize= 8)


plt.title('Neighbourhoods in Toronto with more than 5 Coffee Shops', fontsize=15)

plt.xlabel('Neighbourhoods', fontsize= 12)
plt.ylabel('Number of Venues', fontsize= 12)

plt.show()


# In[32]:


df_coffee_shops.shape


# In[33]:


df_coffee_shops = df_coffee_shops.groupby('Borough').sum()
df_coffee_shops.head()


# There are 184 coffee shops in the City of Toronto. The shops are in 47 Neighbourhoods and 9 Boroughs in Toronto City.

# ## Cluster the Neighbourhoods

# #### Print each neighborhood along with the top 10 most common venues

# With this, we can see the neighborhoods where coffee shop is among the most common venues, as well as the other common venues. We should be on the lookout for: businesses, kindergartens, schools, universities, business district, public transportation, consistent foot traffic. Other places include, smoothie joints, juice bars, bagel places and fast-food chains.
# 
# Neighborods will then be compared and places without coffee shops as one of the 10 most common venues, but which has similar venues as the other neighborhoods(with coffee shop) will be set apart as potential coffee establishment areas

# In[76]:


num_top_venues = 10

for hood in toronto_grouped['Neighbourhood']:
    print("----"+hood+"----")
    temp = toronto_grouped[toronto_grouped['Neighbourhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# #### Let's put that into a pandas dataframe

# First, let's write a function to sort the venues in descending order.

# In[35]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# Now let's create the new dataframe and display the top 10 venues for each neighborhood.

# In[36]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighbourhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighbourhoods_venues_sorted = pd.DataFrame(columns=columns)
neighbourhoods_venues_sorted['Neighbourhood'] = toronto_grouped['Neighbourhood']

for ind in np.arange(toronto_grouped.shape[0]):
    neighbourhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(toronto_grouped.iloc[ind, :], num_top_venues)
    
neighbourhoods_venues_sorted.head()


# In[37]:


neighbourhoods_venues_sorted.shape


# Now, let's cluster the neighboorhoods to determine the discriminating venue categories that distinguish each cluster.

# Create 10 clusters and use the elbow method to obtain the optimal number (k) of clusters

# In[38]:


toronto_grouped_clustering = toronto_grouped.drop('Neighbourhood', 1)
sum_of_squared_distances = []

# maximum of 10 clusters
K = range(1,10)

for k in K:
    print(k, end=' ')
    kmeans = KMeans(n_clusters=k, random_state=1, n_init=20).fit(toronto_grouped_clustering)
    sum_of_squared_distances.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(K, sum_of_squared_distances, 'bx-')
plt.title('Elbow Method For Optimal k')


# Although 2 seems the most optimal value as per diagram above, we are using 5 (the second best k) since this way we can break down more the number of neighbourhoods.

# In[39]:


# set number of clusters
kclusters = 5

toronto_grouped_clustering = toronto_grouped.drop('Neighbourhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(toronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 

# add clustering labels
neighbourhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
toronto_merged = df_new.join(neighbourhoods_venues_sorted.set_index('Neighbourhood'), on='Neighbourhood')

toronto_merged.head(2)


# In[40]:


#drop neighborhoods whose venues could not be sorted
toronto_merged.dropna(subset=['Cluster Labels'], axis=0, inplace= True)
toronto_merged.reset_index(drop=True, inplace=True)
toronto_merged.head()


# We can visualise what are the most common venues and the cluster (from 0 to 5) that has been assigned to each area.

# In[41]:


#change the cluster labels column type to integer
toronto_merged['Cluster Labels']= toronto_merged['Cluster Labels'].astype(int)

# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(toronto_merged['Latitude'], toronto_merged['Longitude'], toronto_merged['Neighbourhood'], toronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# The different colours represent different clusters. Now, we can examine the most popular venues of each cluster in order to assign a name that best represents each of them.

# # Examine Clusters

#  Examine each cluster and determine the discriminating venue categories that distinguish each cluster. 

# #### Cluster 1

# In[56]:


cluster_one= toronto_merged.loc[toronto_merged['Cluster Labels'] == 0, toronto_merged.columns]
cluster_one


# #### Cluster 2

# In[57]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 1, toronto_merged.columns[[1] +[2]  + list(range(5, toronto_merged.shape[1]))]]


# #### Cluster 3

# In[58]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 2, toronto_merged.columns[[1] +[2] + list(range(5, toronto_merged.shape[1]))]]


# #### Cluster 4

# In[59]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 3, toronto_merged.columns[[1] +[2] + list(range(5, toronto_merged.shape[1]))]]


# #### Cluster 5

# In[60]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 4, toronto_merged.columns[[1] +[2] + list(range(5, toronto_merged.shape[1]))]]


# #### Analyze and Select the best cluster for setting up the coffee shop

# Based on the different neighborhoods in each clusters, Clusters 1 and 2 seems to be the best clusters for establishing a coffee store, as they both have plenty of venues, businesses, offices, universities, collehigh passing foot traffic, 
# 
# However, although, coffee shops are one of the most common venues in Cluster 1, it is recommended for the client to establish the coffee shop here as the cluster is large, and has a lot of businesses, schools, parks, bus stations, and places with guaranteeed foot traffic.
# 
# The client will also have to further explore the menu, loyalty discounts, etc, offered by other  coffee s Attention will also be paid to smoothie joints, juice bars, bagel places and even fast-food chains, as they are potential competitors and will determine the products a new coffee shop should serve.

# Let's try to get a list of the coffee shops, using Toronto City Hall as a centre point.

# In[61]:


address = 'Toronto City Hall'

geolocator = Nominatim(user_agent="ny_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude

CLIENT_ID = '2LFZEE5KY44WGBFIAOZBAJUJRUN1VRYJTVNGW3GW4URRSPEK' # your Foursquare ID
CLIENT_SECRET = 'SZ0KO1GZVONF0HUDFI0P3X3GEVZTWOPQJZHLGDTG0VAXZZCJ' # your Foursquare Secret
ACCESS_TOKEN = 'L10JDGKBDSWTLXHGFG1RBL5WMBPA5XAIHH03MSI2MTR3DAP4' # your FourSquare Access Token

#get category id of coffee shop

search_query = 'Coffee'
radius = 3000

#define url
url = 'https://api.foursquare.com/v2/venues/search?client_id={}&client_secret={}&ll={},{}&oauth_token={}&v={}&query={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET, latitude, longitude,ACCESS_TOKEN, VERSION, search_query, radius, LIMIT)

results = requests.get(url).json()

#assign relevant part of JSON to venues
venues = results['response']['venues']

# tranform venues into a dataframe
dataframe = json_normalize(venues)

dataframe.head()


# In[62]:


'There are {} coffee shops around Toronto city Hall between a distance of {} and {}'.format(dataframe.shape[0],(dataframe['location.distance'].min()), (dataframe['location.distance'].max()) )


# In[63]:


# keep only columns that include venue name, and anything that is associated with location
filtered_columns = ['name', 'categories'] + [col for col in dataframe.columns if col.startswith('location.')] + ['id']
dataframe_filtered = dataframe.loc[:, filtered_columns]

# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']

# filter the category for each row
dataframe_filtered['categories'] = dataframe_filtered.apply(get_category_type, axis=1)

# clean column names by keeping only last term
dataframe_filtered.columns = [column.split('.')[-1] for column in dataframe_filtered.columns]

#rename the columns
dataframe_filtered.rename(columns={'postalCode':'Postal Code'}, inplace=True)

dataframe_filtered.head()


# In[64]:


#drop coffee shops without postal codes

dataframe_filtered2= dataframe_filtered[['name', 'address', 'lat', 'lng', 'Postal Code','distance','formattedAddress']]

dataframe_filtered2.dropna(subset=['Postal Code'], axis=0, inplace= True)
dataframe_filtered2.reset_index(drop=True, inplace=True)

#lets split the postal code and keep only the first part of the postal code
dataframe_filtered2[['First','Last']] = dataframe_filtered2['Postal Code'].str.split(' ',expand=True)
dataframe_filtered2.rename(columns={'Postal Code':'Postals', 'First':'Postal Code'}, inplace= True)
dataframe_filtered2.head()


# Now, let's extract the coffee shops in the recommended cluster: Cluster 1, from the list of Coffee shops around our reference point, i.e., Toronto City Hall

# In[72]:


cluster_rec= cluster_one[['Borough', 'Neighbourhood', 'Postal Code']]
# Format with commas and round off to two decimal places in pandas 
#pd.options.display.float_format = '{:.2f}'.format

cluster1_recommend= dataframe_filtered2.merge(cluster_rec, on='Postal Code')
cluster1_recommend.head()


# In[66]:


'Out of the 50 coffee shops around Toronto City Hall, there are {} coffee shops in Cluster 1 between a distance of {} and {} from the Hall'.format(cluster1_recommend.shape[0],(cluster1_recommend['distance'].min()), (cluster1_recommend['distance'].max()) )


# In[70]:


cluster1_recommend.name.value_counts().reset_index().rename(columns={'name':'count', 'index':'name'}).head()


# Timothy's World Coffee seems to be the leading Coffee provider in Cluster 1 with 8 outlets, floowed by Balzac's Coffee with 3 outlets. 

# Now, let's visualize the Coffee Shops in Cluster 1

# In[68]:


venues_map = folium.Map(location=[latitude, longitude], zoom_start=13) # generate map centred around the Conrad Hotel

# add a red circle marker to represent the Toronto City Hall
folium.CircleMarker(
    [latitude, longitude],
    radius=10,
    color='red',
    popup='Toronto City Hall',
    fill = True,
    fill_color = 'red',
    fill_opacity = 0.6
).add_to(venues_map)

# add the Coffee Shops as blue circle markers
for lat, lng, name, Neighbourhood in zip(cluster1_recommend.lat, cluster1_recommend.lng, cluster1_recommend.name, cluster1_recommend.Neighbourhood):
    label= '{}, {}'.format(name, Neighbourhood)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        color='blue',
        popup=label,
        fill = True,
        fill_color='blue',
        fill_opacity=0.6
    ).add_to(venues_map)

# display map
venues_map


# In[71]:


#let's compare the number of coffee shops in cluster 1 to the total number of coffee shops per neighbourhood

cluster1_recommended= cluster1_recommend.merge(df_coffee_neigh, on='Neighbourhood')

cluster1_recommended_shops= cluster1_recommended[['name', 'address','Neighbourhood', 'Borough', 'Coffee Shop', 'Venue Category','distance', 'Proportion %' ]]
cluster1_recommended_shops.head()


# ## Recommendation/Conclusion

# If we compare the number of coffee shops in cluster 1 to the total number of coffee shops per neighbourhood, we can identify two (2) unique neighbourhoods: Garden District Ryeson and Central Bay Street. While Central Bay Street seems to have enough number of coffee shops: 11 coffee shops out of 60 venues (8 of these coffee shops are in Cluster 1), in Garden District Ryeson, out of the total 100 venues there are 9 coffee shops, 4 of which are in cluster 1.
# 
# Therefore, the best neighbourhood for establishing a new coffee shop is Garden District Ryeson. This recommended neighborhood has a businesses, universities, parks, gardens, bus lines, and guaranteed passing foot trafic. Foursquare API should be further used to get the competitors reviews and users tips to have an idea of the deals/ discounts/ loyalty schemes the competitors are offering
