
# coding: utf-8

# In[241]:


import pandas as pd
import json
import shapely
from shapely.geometry import Point
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from pyproj import Proj, transform
from ipyleaflet import Map, Heatmap
import re
get_ipython().run_line_magic('matplotlib', 'inline')
import sys
sys.path.append('../scrape/funda/A360631/utils/')
#%run processors.py


# In[242]:


import processors
_num = processors.Number()
to_number = lambda x: u" ".join(_num(x))
_price = processors.Price()
to_price = lambda x: " ".join(_price(x))
identity = lambda x, *args: (x,) + args if args else x


# In[243]:


df = pd.read_json('../data/funda_rented_utrecht.json',lines=True)


# In[244]:


station = Point( 5.109995, 52.089627 )


# In[245]:


def get_url(s):
    try:
        return s.split('/{')[0] 
    except:
        np.NaN
    
def get_ll(s):
    try:
        url = s.split('/{')[0] 
        len_url = len(url)
        j = json.loads(s[len_url+1:])
        return  Point(float(j['lng']), float(j['lat']))
    except:
        return Point()#np.NaN
def get_nums(s):
    try:
        ns = u" ".join(_num([s]))
        return ns if len(ns)>0 else np.NaN
    except:
        return np.NaN

def get_price(s):
    try:
        ns = _price([s])
        return float(ns[0]) if len(ns)>0 else np.NaN
    except:
        return np.NaN
    
def get_tup(s,index=0,sep=' '):
    try:
        return s.split(sep)[index]
    except:
        return np.NaN
def haversine_np(point1,point2):
    try:
        lon1, lat1, lon2, lat2 = map(np.radians, [point1.x, point1.y, point2.x, point2.y])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

        c = 2 * np.arcsin(np.sqrt(a))
        m = 6367000 * c
        return m
    except:
        return np.NaN
def get_year(s):
    try:
        return int(re.findall('[\d]{4}',s)[0])
    except:
        return np.NaN
def get_layer(s):
    try:
        return re.search('(\d) residential layer',s).group(1)
    except:
        return np.NaN
def get_floor(s):
    try:
        if s == 'Ground floor':
            return 0
        else:
            return re.search('(\d+).*level of residential structure',s).group(1)
    except:
        return np.NaN    
def get_street(s):
    try:
        return re.sub('\d+.*$','',s).strip()
    except:
        return np.NaN


# In[246]:


df['url'] = df['geo'].apply(get_url)
df['geometry'] = df['geo'].apply(get_ll)
del df['geo']


# In[247]:


df['energy_rating'] = df['energy'].apply(lambda x: float(get_tup(x,1)))
df['energy_label'] = df['energy'].apply(lambda x: get_tup(x,0))


# In[248]:


df['no_rooms'] = df['no_rooms'].apply(get_nums)
df['no_bedrooms'] = df['no_rooms'].apply(lambda x: float(get_tup(x,1,' ')))
df['no_rooms'] = df['no_rooms'].apply(lambda x: float(get_tup(x,0,' ')))


# In[249]:


df['price'] = df['price'].apply(get_price)


# In[250]:


df['living_area'] = df['living_area'].apply(get_nums)
df['living_area'] = df['living_area'].apply(lambda x: float(get_tup(x,0,' ')))


# In[251]:


df['volume'] = df['volume'].apply(get_nums).apply(float)


# In[252]:


df['ppm'] = np.divide(df['price'], df['living_area'])
#df['price']


# In[253]:


df['station_distance'] = df['geometry'].apply(lambda x: haversine_np(x,station))


# In[254]:


df['construction_year'] = df['construction_year'].apply(get_year)


# In[255]:


df['no_layers'] = df['no_layers'].apply(get_layer)


# In[256]:


df['layer'] = df['layer'].apply(get_floor)


# In[257]:


df['type'] = df['type'].astype('category')
df['building_type'] = df['building_type'].astype('category')
df['parking'] = df['parking'].astype('category')
df['heating'] = df['heating'].astype('category')
df['insulation'] = df['insulation'].astype('category')
df['hot_water'] = df['hot_water'].astype('category')
df['address'] = df['address'].astype('category')
df['street'] = df['house_no'].apply(get_street).astype('category')
df['area_code'] = df['address'].apply(lambda x: int(get_tup(x,0))).astype('category')
df['energy_label'] = df['energy_label'].astype('category')


# In[ ]:


df['listed_from'] = pd.to_datetime(df['listed_from'])
df['listed_to'] = pd.to_datetime(df['listed_to'])


# In[258]:


gdf = gpd.GeoDataFrame(df)


# In[268]:


gdf.to_pickle('../data/funda_utrecht.en.pkl')

