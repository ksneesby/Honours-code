# import the functions in other python script
import regrid_swaths as rs

# dates and maths
from datetime import datetime
import numpy as np

# for plotting
import matplotlib.pyplot as plt

from calendar import monthrange

import calendar

import itertools

no_PFT = 4
no_HCHO = 4
no_clusters = max(no_PFT,no_HCHO)       ### Number of clusters, use highest K

PFT_type = "all"   ### 'all' or 'dominant'
PFT_name = PFT_type

def mask_fire(HCHO,fire):
    fmask=fire>5 # mask where there are more than a few fire pixels
    HCHO=np.ma.array(HCHO, mask=fmask)
    return HCHO

month = 1
month_text = calendar.month_name[month]

last_day = monthrange(2005, month)

d0=datetime(2005,month,1)
dN=datetime(2005,month,last_day[1])


## Read a single day and plot the data
##


data,attr=rs.read_regridded_swath(d0)

#print(data.keys()) # shows what keys are in our data structure
#print(data['VC_C'].shape) # shape of VC_C array
#print(attr['VC_C']) # attributes for VC_C

HCHO=data['VC_C']
fire=data['fires']
fmask=fire>5 # mask where there are more than a few fire pixels
HCHO_masked=np.ma.array(HCHO, mask=fmask)

# HCHO[fmask]=np.NaN  # alternative way of masking
lats=data['lats']
lons=data['lons']

from scipy import stats


# =============================================================================
# # Create a figure:
# plt.figure(figsize=(10,7))
# 
# # Plot columns:
# plt.subplot(211) # two rows, 1 column, first subplot
# rs.plot_map(HCHO,lats,lons,linear=False,vmin=1e14,vmax=1e16,
#                 cbarlabel='molec/cm2')
# plt.title('OMI HCHO Columns')
# 
# # plot fire mask:
# plt.subplot(223) # two rows, two columns, third subplot
# rs.plot_map(fire,lats,lons,linear=False,vmin=1,vmax=1e8, 
#                 cbarlabel='fire pixels/day')
# plt.title('MOD14A1 Fire')
# 
# # plot masked by fires:
# plt.subplot(224) # 
# rs.plot_map(HCHO_masked,lats,lons, linear=False,vmin=1e14,vmax=1e16,
#                 cbarlabel='molec/cm2')
# plt.title('OMI masked by fires>500')
# 
# 
# # save the figure:
# plt.suptitle(d0.strftime('Plots for %Y %m %d'))
# plt.savefig('test_plot.png')
# plt.close()
# print('test_plot.png saved')
# 
# =============================================================================
###
### Now grab a bunch of days at once:




data, fires, days, lats, lons=rs.read_key(d0,dN)


print(data.shape) # shape of VC_C array

HCHO=data
fire=fires
lats=lats
lons=lons

plt.figure(figsize=(10,12))


mask_fire(HCHO,fire)

# =============================================================================
# 
# # month avg 
# plt.subplot(311)
# 
# rs.plot_map(HCHO_05_mean,lats,lons,linear=False,
#             vmin=1e14,vmax=1e16,cbarlabel='molec/cm2')
# plt.title('HCHO averaged over a month')
# 
# for i in range(4):
#     plt.subplot(323+i)
#     rs.plot_map(HCHO[i,:,:],lats,lons,linear=False,
#             vmin=1e14,vmax=1e16,cbarlabel='molec/cm2')
# #    plt.title(data['time'][i].strftime('OMI hcho %Y %m %d'))
# 
# plt.savefig('test_plot2.png')
# plt.close()
# print('test_plot2.png saved')
# 
# 
# =============================================================================


HCHO_05_mean=np.nanmean(HCHO,axis=0)

## Get more years for mean

last_day = monthrange(2006, month)
d0=datetime(2006,month,1)
dN=datetime(2006,month,last_day[1])
HCHO, fire, days, lats, lons=rs.read_key(d0,dN)
mask_fire(HCHO,fire)
HCHO_06=HCHO
HCHO_06_mean=np.nanmean(HCHO_06,axis=0)

last_day = monthrange(2007, month)
d0=datetime(2007,month,1)
dN=datetime(2007,month,last_day[1])
HCHO, fire, days, lats, lons=rs.read_key(d0,dN)
mask_fire(HCHO,fire)
HCHO_07=HCHO
HCHO_07_mean=np.nanmean(HCHO_07,axis=0)

last_day = monthrange(2008, month)
d0=datetime(2008,month,1)
dN=datetime(2008,month,last_day[1])
HCHO, fire, days, lats, lons=rs.read_key(d0,dN)
mask_fire(HCHO,fire)
HCHO_08=HCHO
HCHO_08_mean=np.nanmean(HCHO_08,axis=0)

last_day = monthrange(2009, month)
d0=datetime(2009,month,1)
dN=datetime(2009,month,last_day[1])
HCHO, fire, days, lats, lons=rs.read_key(d0,dN)
mask_fire(HCHO,fire)
HCHO_09=HCHO
HCHO_09_mean=np.nanmean(HCHO_09,axis=0)

last_day = monthrange(2010, month)
d0=datetime(2010,month,1)
dN=datetime(2010,month,last_day[1])
HCHO, fire, days, lats, lons=rs.read_key(d0,dN)
mask_fire(HCHO,fire)
HCHO_10=HCHO
HCHO_10_mean=np.nanmean(HCHO_10,axis=0)

last_day = monthrange(2011, month)
d0=datetime(2011,month,1)
dN=datetime(2011,month,last_day[1])
HCHO, fire, days, lats, lons=rs.read_key(d0,dN)
mask_fire(HCHO,fire)
HCHO_11=HCHO
HCHO_11_mean=np.nanmean(HCHO_11,axis=0)

last_day = monthrange(2012, month)
d0=datetime(2012,month,1)
dN=datetime(2012,month,last_day[1])
HCHO, fire, days, lats, lons=rs.read_key(d0,dN)
mask_fire(HCHO,fire)
HCHO_12=HCHO
HCHO_12_mean=np.nanmean(HCHO_12,axis=0)

last_day = monthrange(2013, month)
d0=datetime(2013,month,1)
dN=datetime(2013,month,last_day[1])
HCHO, fire, days, lats, lons=rs.read_key(d0,dN)
mask_fire(HCHO,fire)
HCHO_13=HCHO
HCHO_13_mean=np.nanmean(HCHO_13,axis=0)

last_day = monthrange(2014, month)
d0=datetime(2014,month,1)
dN=datetime(2014,month,last_day[1])
HCHO, fire, days, lats, lons=rs.read_key(d0,dN)
mask_fire(HCHO,fire)
HCHO_14=HCHO
HCHO_14_mean=np.nanmean(HCHO_14,axis=0)

HCHO_mean = np.stack((HCHO_05_mean,HCHO_06_mean,HCHO_07_mean,HCHO_08_mean,HCHO_09_mean,HCHO_10_mean,HCHO_11_mean,HCHO_12_mean,HCHO_13_mean,HCHO_14_mean),axis=0)
HCHO_std = np.nanstd(HCHO_mean,axis=0)
HCHO_mean=np.nanmean(HCHO_mean,axis=0)


###
### Limit HCHO_mean to just Australian region data ~ S W N E ~ [-45, 108.75, -10, 156.25]

for i in range (len(lats)):
    if lats[i] < -45:
        lats_min = i
    if lats[i] <= -10:
        lats_max = i

for i in range (len(lons)):
    if lons[i] < 108.75:
        lons_min = i
    if lons[i] <= 154:
        lons_max = i

HCHO_mean_Aus = HCHO_mean[lats_min:lats_max,lons_min:lons_max]
Aus_lats = lats[lats_min:lats_max]
Aus_lons = lons[lons_min:lons_max]
HCHO_std_Aus = HCHO_std[lats_min:lats_max,lons_min:lons_max]


Aus_reshape = np.reshape(HCHO_mean_Aus,(len(HCHO_mean_Aus)*len(HCHO_mean_Aus[0])))
std_reshape = np.reshape(HCHO_std_Aus,(len(HCHO_std_Aus)*len(HCHO_std_Aus[0])))
Aus_lats_reshape = np.repeat(Aus_lats, len(Aus_lons))
Aus_lons_reshape = np.tile(Aus_lons, len(Aus_lats))
HCHO = np.full((len(Aus_reshape),4), np.nan) 
HCHO[:,0] = Aus_reshape
HCHO[:,3] = Aus_lats_reshape 
HCHO[:,2] = Aus_lons_reshape
HCHO[:,1] = std_reshape

## Remove oceans

from mpl_toolkits.basemap import Basemap
from matplotlib.path import Path


# S W N E
__AUSREGION__=[-45, 108.75, -7, 156.25]
region=__AUSREGION__
map =Basemap(llcrnrlat=region[0], urcrnrlat=region[2], llcrnrlon=region[1], urcrnrlon=region[3],
                  resolution='i', projection='merc')

x, y = map(HCHO[:,2], HCHO[:,3])

locations = np.c_[x, y]

polygons = [Path(p.boundary) for p in map.landpolygons]

result = np.zeros(len(locations), dtype=bool) 

for polygon in polygons:

    result += np.array(polygon.contains_points(locations))


for i in reversed(range(len(result))):
    if result[i] == False:
        HCHO = np.delete(HCHO,i,0)


no_ocean_lons = np.unique(HCHO[:,2])
no_ocean_lons_HCHO = no_ocean_lons
no_ocean_lats = np.unique(HCHO[:,3])
no_ocean_lats_HCHO = no_ocean_lats
no_ocean_HCHO = np.full((len(no_ocean_lats),len(no_ocean_lons)), np.nan)
no_ocean_std = np.full((len(no_ocean_lats),len(no_ocean_lons)), np.nan)

for i in range(len(HCHO)):
    value_lon = HCHO[i,2]
    value_lat = HCHO[i,3]
    pos_lon = np.where( no_ocean_lons==value_lon )
    pos_lat = np.where( no_ocean_lats==value_lat )
    no_ocean_HCHO[pos_lat,pos_lon] = HCHO[i,0]
    no_ocean_std[pos_lat,pos_lon] = HCHO[i,1]
    
# =============================================================================
# rs.plot_map(no_ocean_HCHO,no_ocean_lats,no_ocean_lons,linear=False,
#             vmin=1e14,vmax=1e16,cbarlabel='molec/cm2')
# 
# plt.savefig('test_plot3.png')
# plt.close()
# print('test_plot3.png saved')
# 
# =============================================================================

###
### K-Means clustering


# =============================================================================
# from mpl_toolkits.mplot3d import Axes3D
# 
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=X[:,0])
# 
# # rotate the axes and update
# for angle in range(0, 360):
#     ax.view_init(90, angle)
# =============================================================================

Aus_reshape = np.reshape(no_ocean_HCHO,(len(no_ocean_HCHO)*len(no_ocean_HCHO[0])))
std_reshape = np.reshape(no_ocean_std,(len(no_ocean_std)*len(no_ocean_std[0])))
Aus_lats_reshape = np.repeat(no_ocean_lats, len(no_ocean_lons))
Aus_lons_reshape = np.tile(no_ocean_lons, len(no_ocean_lats))
HCHO = np.full((len(Aus_reshape),4), np.nan) 
HCHO[:,0] = Aus_reshape
HCHO[:,2] = Aus_lats_reshape 
HCHO[:,3] = Aus_lons_reshape
HCHO[:,1] = std_reshape

to_delete = np.full(len(HCHO)*2,np.nan)
count = 0
for y in range(len(HCHO[0])):
    for i in range(len(HCHO)):
        if not np.isfinite(HCHO[i,y]):
            to_delete[count] = i
            count +=1
        
to_delete = np.unique(to_delete)
to_delete = np.lib.pad(to_delete, (0,(len(HCHO)*2-len(to_delete))), 'constant', constant_values=(np.nan))

         
for i in reversed(range(len(HCHO))):
    if np.isfinite(to_delete[i]):
        HCHO = np.delete(HCHO,to_delete[i],0)


## Normalise variables
        
from sklearn import preprocessing

HCHO = preprocessing.RobustScaler().fit_transform(HCHO)


# =============================================================================
# W = np.full((8900,2), np.nan) 
# W[:,1] = X[:,3]
# W[:,0] = X[:,2]
# =============================================================================




from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=no_HCHO)
kmeans = kmeans.fit(HCHO)
labels = kmeans.predict(HCHO)
C_HCHO = kmeans.cluster_centers_

L_HCHO = kmeans.labels_
L_HCHO = np.array(L_HCHO, dtype=float)
for i in range(len(to_delete)):
    if np.isfinite(to_delete[i]):
        L_HCHO = np.insert(L_HCHO,to_delete[i].astype(int),np.nan)


# =============================================================================
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=L)
# ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c='#050505', s=1000)
# =============================================================================
           

L_HCHO_reshape = np.reshape(L_HCHO,(len(no_ocean_lats),len(no_ocean_lons)))
      
plt.subplot(221) # one row, two columns, first subplot
rs.plot_map(L_HCHO_reshape,no_ocean_lats,no_ocean_lons,linear=True,
            vmin=0,vmax=no_clusters,cbarlabel='Cluster', cmap="tab10")




############### PFT


# import the functions in other python script
import regrid_swaths as rs

# dates and maths
from datetime import datetime
import numpy as np

# for plotting
import matplotlib.pyplot as plt

import pandas as pd
import xarray as xr

import bisect




## Read in data

ds=xr.open_dataset('PFT/CLM4_PFT.generic.025x03125.nc')

lats = ds.lat
lons = ds.lon

df = ds.to_dataframe()
df = df.reset_index()
df = df.drop('time',1)

### 
### Limit df to just Australian region data ~ S W N E ~ [-45, 108.75, -10, 156.25]

for i in reversed(range(len(df))):
    if df.loc[i,'lat'] >= -10.2:
        lat_max = i

for i in range(len(df)):
    if df.loc[i,'lat'] < -45:
        lat_min = i



X = df.iloc[lat_min:lat_max,:]
X = X.reset_index(drop=True)

X = X.sort_values(['lon','lat'])
X = X.reset_index(drop=True)

for i in reversed(range(len(X))):
    if X.loc[i,'lon'] >= 154:
        lon_max = i

for i in range(len(X)):
    if X.loc[i,'lon'] < 108.75:
        lon_min = i
 
    
Aus_lons = np.unique(X.loc[lon_min:lon_max,'lon'])
Aus_lats = np.unique(df.loc[lat_min:lat_max,'lat'])


df = X.iloc[lon_min:lon_max,:]
df = df.reset_index(drop=True)

X = X.sort_values(['lat','lon'])
X = X.reset_index(drop=True)
df = df.sort_values(['lat','lon'])
df = df.reset_index(drop=True)


## Remove oceans

from mpl_toolkits.basemap import Basemap
from matplotlib.path import Path


# S W N E
__AUSREGION__=[-45, 108.75, -7, 156.25]
region=__AUSREGION__
map =Basemap(llcrnrlat=region[0], urcrnrlat=region[2], llcrnrlon=region[1], urcrnrlon=region[3],
                  resolution='i', projection='merc')

x, y = map(df.loc[:,'lon'].values, df.loc[:,'lat'].values)

locations = np.c_[x, y]

polygons = [Path(p.boundary) for p in map.landpolygons]

result = np.zeros(len(locations), dtype=bool) 

for polygon in polygons:

    result += np.array(polygon.contains_points(locations))

df = df[~result==False]
df = df.reset_index(drop=True)

no_ocean_lons = np.unique(df.loc[:,'lon'])
no_ocean_lats = np.unique(df.loc[:,'lat'])
no_ocean_data1 = np.full((len(no_ocean_lats),len(no_ocean_lons)), np.nan)
no_ocean_data2 = np.full((len(no_ocean_lats),len(no_ocean_lons)), np.nan)
no_ocean_data3 = np.full((len(no_ocean_lats),len(no_ocean_lons)), np.nan)
no_ocean_data4 = np.full((len(no_ocean_lats),len(no_ocean_lons)), np.nan)
no_ocean_data5 = np.full((len(no_ocean_lats),len(no_ocean_lons)), np.nan)
no_ocean_data6 = np.full((len(no_ocean_lats),len(no_ocean_lons)), np.nan)
no_ocean_data7 = np.full((len(no_ocean_lats),len(no_ocean_lons)), np.nan)
no_ocean_data8 = np.full((len(no_ocean_lats),len(no_ocean_lons)), np.nan)
no_ocean_data9 = np.full((len(no_ocean_lats),len(no_ocean_lons)), np.nan)
no_ocean_data10 = np.full((len(no_ocean_lats),len(no_ocean_lons)), np.nan)
no_ocean_data11 = np.full((len(no_ocean_lats),len(no_ocean_lons)), np.nan)
no_ocean_data12 = np.full((len(no_ocean_lats),len(no_ocean_lons)), np.nan)
no_ocean_data13 = np.full((len(no_ocean_lats),len(no_ocean_lons)), np.nan)
no_ocean_data14 = np.full((len(no_ocean_lats),len(no_ocean_lons)), np.nan)
no_ocean_data15 = np.full((len(no_ocean_lats),len(no_ocean_lons)), np.nan)
no_ocean_data16 = np.full((len(no_ocean_lats),len(no_ocean_lons)), np.nan)

for i in range(len(df)):
    value_lon = df.loc[i,'lon']
    value_lat = df.loc[i,'lat']
    pos_lon = np.where( no_ocean_lons==value_lon )
    pos_lat = np.where( no_ocean_lats==value_lat )
    no_ocean_data1[pos_lat,pos_lon] = df.iloc[i,2]
    no_ocean_data2[pos_lat,pos_lon] = df.iloc[i,3]
    no_ocean_data3[pos_lat,pos_lon] = df.iloc[i,4]
    no_ocean_data4[pos_lat,pos_lon] = df.iloc[i,5]
    no_ocean_data5[pos_lat,pos_lon] = df.iloc[i,6]
    no_ocean_data6[pos_lat,pos_lon] = df.iloc[i,7]
    no_ocean_data7[pos_lat,pos_lon] = df.iloc[i,8]
    no_ocean_data8[pos_lat,pos_lon] = df.iloc[i,9]
    no_ocean_data9[pos_lat,pos_lon] = df.iloc[i,10]
    no_ocean_data10[pos_lat,pos_lon] = df.iloc[i,11]
    no_ocean_data11[pos_lat,pos_lon] = df.iloc[i,12]
    no_ocean_data12[pos_lat,pos_lon] = df.iloc[i,13]
    no_ocean_data13[pos_lat,pos_lon] = df.iloc[i,14]
    no_ocean_data14[pos_lat,pos_lon] = df.iloc[i,15]
    no_ocean_data15[pos_lat,pos_lon] = df.iloc[i,16]
    no_ocean_data16[pos_lat,pos_lon] = df.iloc[i,17]    

# =============================================================================
# rs.plot_map(no_ocean_data1,no_ocean_lats,no_ocean_lons,linear=False,
#             vmin=0.0000001,vmax=1,cbarlabel='molec/cm2')
# 
# plt.savefig('test_plot3.png')
# plt.close()
# print('test_plot3.png saved')
# 
# 
# =============================================================================

no_ocean_data1_reshape = np.reshape(no_ocean_data1,(len(no_ocean_data1)*len(no_ocean_data1[0])))
no_ocean_data2_reshape = np.reshape(no_ocean_data2,(len(no_ocean_data2)*len(no_ocean_data2[0])))
no_ocean_data3_reshape = np.reshape(no_ocean_data3,(len(no_ocean_data3)*len(no_ocean_data3[0])))
no_ocean_data4_reshape = np.reshape(no_ocean_data4,(len(no_ocean_data4)*len(no_ocean_data4[0])))
no_ocean_data5_reshape = np.reshape(no_ocean_data5,(len(no_ocean_data5)*len(no_ocean_data5[0])))
no_ocean_data6_reshape = np.reshape(no_ocean_data6,(len(no_ocean_data6)*len(no_ocean_data6[0])))
no_ocean_data7_reshape = np.reshape(no_ocean_data7,(len(no_ocean_data7)*len(no_ocean_data7[0])))
no_ocean_data8_reshape = np.reshape(no_ocean_data8,(len(no_ocean_data8)*len(no_ocean_data8[0])))
no_ocean_data9_reshape = np.reshape(no_ocean_data9,(len(no_ocean_data9)*len(no_ocean_data9[0])))
no_ocean_data10_reshape = np.reshape(no_ocean_data10,(len(no_ocean_data10)*len(no_ocean_data10[0])))
no_ocean_data11_reshape = np.reshape(no_ocean_data11,(len(no_ocean_data11)*len(no_ocean_data11[0])))
no_ocean_data12_reshape = np.reshape(no_ocean_data12,(len(no_ocean_data12)*len(no_ocean_data12[0])))
no_ocean_data13_reshape = np.reshape(no_ocean_data13,(len(no_ocean_data13)*len(no_ocean_data13[0])))
no_ocean_data14_reshape = np.reshape(no_ocean_data14,(len(no_ocean_data14)*len(no_ocean_data14[0])))
no_ocean_data15_reshape = np.reshape(no_ocean_data15,(len(no_ocean_data15)*len(no_ocean_data15[0])))
no_ocean_data16_reshape = np.reshape(no_ocean_data16,(len(no_ocean_data16)*len(no_ocean_data16[0])))
no_ocean_lats_reshape = np.repeat(no_ocean_lats, len(no_ocean_lons))
no_ocean_lons_reshape = np.tile(no_ocean_lons, len(no_ocean_lats))
X = np.full((len(no_ocean_data1_reshape),18), np.nan) 
X[:,0] = no_ocean_lats_reshape
X[:,1] = no_ocean_lons_reshape
X[:,2] = no_ocean_data1_reshape
X[:,3] = no_ocean_data2_reshape 
X[:,4] = no_ocean_data3_reshape
X[:,5] = no_ocean_data4_reshape
X[:,6] = no_ocean_data5_reshape
X[:,7] = no_ocean_data6_reshape
X[:,8] = no_ocean_data7_reshape
X[:,9] = no_ocean_data8_reshape
X[:,10] = no_ocean_data9_reshape
X[:,11] = no_ocean_data10_reshape
X[:,12] = no_ocean_data11_reshape 
X[:,13] = no_ocean_data12_reshape
X[:,14] = no_ocean_data13_reshape
X[:,15] = no_ocean_data14_reshape
X[:,16] = no_ocean_data15_reshape
X[:,17] = no_ocean_data16_reshape


for i in reversed(range(len(X[0]))):
    if (np.nanmax(X[:,i],axis=0))==0:
        X = np.delete(X,i,1)


to_delete = np.full(len(X)*18,np.nan)
count = 0
for y in range(len(X[0])):
    for i in range(len(X)):
        if not np.isfinite(X[i,y]):
            to_delete[count] = i
            count +=1
        
to_delete = np.unique(to_delete)
# =============================================================================
# to_delete = np.lib.pad(to_delete, (0,(len(X)-len(to_delete))), 'constant', constant_values=(np.nan))
# =============================================================================

         
for i in reversed(range(len(X))):
    if np.isfinite(to_delete[i]):
        X = np.delete(X,to_delete[i],0)




###
### K-Means clustering


## Normalise variables
        
from sklearn import preprocessing

d = preprocessing.MinMaxScaler().fit_transform(X)
X[:,0] = d[:,0]
X[:,1] = d[:,1]


# Use only highest value variable for each location

max_values = np.full((len(X),1),np.nan)
max_values = X[:,2:18].max(axis=1)
W = np.full((len(max_values),3),np.nan)
W[:,0] = df.loc[:,'lat']
W[:,1] = df.loc[:,'lon']
W[:,2] = max_values


# Plot dominant species
for i in range(len(max_values)):
    W[i,2] = np.argwhere(X[i,:]==max_values[i])
    
W_lons = np.unique(df.loc[:,'lon'])
W_lats = np.unique(df.loc[:,'lat'])
W_data = np.full((len(W_lats),len(W_lons)), np.nan)




lowest = 0
numbers = np.unique(W[:,2])
finished = False

while not finished:
    next_lowest = numbers[lowest]
    next_lowest = int(next_lowest)
          
    for i in range(len(W)):
        if W[i,2] == next_lowest:
            W[i,2] = lowest

    if lowest == np.max(W[:,2],axis=0):
        finished = True
        
    lowest +=1

    
for i in range(len(W)):
    value_lon = W[i,1]
    value_lat = W[i,0]
    pos_lon = np.where( W_lons==value_lon )
    pos_lat = np.where( W_lats==value_lat )
    W_data[pos_lat,pos_lon] = W[i,2]



# =============================================================================
# rs.plot_map(W_data,W_lats,W_lons,linear=True,
#             vmin=0,vmax=9,cbarlabel='cluster',cmap="tab10")
# 
# 
# plt.savefig('test_plot5.png')
# plt.close()
# print('test_plot5.png saved')
# 
# 
# =============================================================================

W = preprocessing.MinMaxScaler().fit_transform(W)




from sklearn.cluster import KMeans

if PFT_type == 'dominant':
    PFT_type = W
    
if PFT_type == 'all':
    PFT_type = X

 
kmeans = KMeans(n_clusters=no_PFT)
kmeans = kmeans.fit(PFT_type)
labels = kmeans.predict(PFT_type)
C_PFT = kmeans.cluster_centers_

L_PFT = kmeans.labels_
L_PFT = np.array(L_PFT, dtype=float)
for i in range(len(to_delete)):
    if np.isfinite(to_delete[i]):
        L_PFT = np.insert(L_PFT,to_delete[i].astype(int),np.nan)


L_PFT_reshape = np.reshape(L_PFT,(len(no_ocean_lats),len(no_ocean_lons)))



diff = L_HCHO_reshape-L_PFT_reshape
diff[np.isnan(diff)] = 999
diff[(diff != 0) & (diff <100)] = 1
diff[diff == 999] = np.nan

for l in range(len(to_delete)):
        if np.isfinite(to_delete[l]):
            kmeans.labels_ = np.insert(kmeans.labels_.astype(float),to_delete[l].astype(int),np.nan)

kmeans.labels_[np.isnan(kmeans.labels_)] = no_PFT
kmeans.labels_ = kmeans.labels_.astype(int)


all_lut = list(itertools.permutations(np.arange(0,no_PFT,1)))

for i in range(len(all_lut)):
    temp = all_lut[i]
    temp = np.array(temp, dtype=int)
    temp = np.append(temp,np.nan)
    temp_PFT = temp[kmeans.labels_]
    temp_PFT = np.array(temp_PFT, dtype=float)
    
    temp_PFT_reshape = np.reshape(temp_PFT,(len(no_ocean_lats),len(no_ocean_lons)))
    temp_diff = L_HCHO_reshape-temp_PFT_reshape
    temp_diff[np.isnan(temp_diff)] = 999
    temp_diff[(temp_diff != 0) & (temp_diff <100)] = 1
    temp_diff[temp_diff == 999] = np.nan
    if np.nansum(temp_diff) < np.nansum(diff):
        L_PFT_reshape = temp_PFT_reshape
        diff = temp_diff




diff = L_HCHO_reshape-L_PFT_reshape
diff[np.isnan(diff)] = 999
diff[(diff != 0) & (diff <100)] = 1
diff[diff == 999] = np.nan


plt.subplot(222) # one row, two columns, second subplot
rs.plot_map(L_PFT_reshape,no_ocean_lats,no_ocean_lons,linear=True,
            vmin=0,vmax=no_clusters,cbarlabel='Cluster', cmap="tab10")
plt.show







plt.subplot(223)
rs.plot_map(diff,no_ocean_lats,no_ocean_lons,linear=True,
            vmin=0,vmax=1,cbarlabel='diff = '+ str(np.nansum(diff)), cmap="Purples")

plt.savefig('Difference Plots/HCHO-PFT/' + PFT_name + '/' + month_text + ' HCHO-PFT Difference Map (K=' + str(no_HCHO) + ', ' + str(no_PFT) + ')' + PFT_name + '.png')
plt.close()





























# =============================================================================
# 
# ######## Plot Elbow Curve
# 
# from scipy.spatial.distance import cdist
# 
# # create new plot and data
# plt.plot()
# colors = ['b', 'g', 'r']
# markers = ['o', 'v', 's']
#  
# # k means determine k
# distortions = []
# K = range(1,10)
# for k in K:
#     kmeanModel = KMeans(n_clusters=k).fit(X)
#     kmeanModel.fit(X)
#     distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
#  
# # Plot the elbow
# plt.plot(K, distortions, 'bx-')
# plt.xlabel('k')
# plt.ylabel('Distortion')
# plt.title('The Elbow Method showing the optimal k')
# # =============================================================================
# # plt.savefig('Elbow Plot.png')
# # =============================================================================
# plt.show()
# =============================================================================

# =============================================================================
# 
# ##### Silhouette Method
# 
# from sklearn.metrics import silhouette_samples, silhouette_score
# import matplotlib.cm as cm
# 
# range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
# silhouette_width = np.full(9, np.nan) 
# 
# for n_clusters in range_n_clusters:
#     # Create a subplot with 1 row and 2 columns
#     fig, (ax1, ax2) = plt.subplots(1, 2)
#     fig.set_size_inches(18, 7)
# 
#     # The 1st subplot is the silhouette plot
#     # The silhouette coefficient can range from -1, 1 but in this example all
#     # lie within [-0.1, 1]
#     ax1.set_xlim([-0.1, 1])
#     # The (n_clusters+1)*10 is for inserting blank space between silhouette
#     # plots of individual clusters, to demarcate them clearly.
#     ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
# 
#     # Initialize the clusterer with n_clusters value and a random generator
#     # seed of 10 for reproducibility.
#     clusterer = KMeans(n_clusters=n_clusters, random_state=10)
#     cluster_labels = clusterer.fit_predict(X)
# 
#     # The silhouette_score gives the average value for all the samples.
#     # This gives a perspective into the density and separation of the formed
#     # clusters
#     silhouette_avg = silhouette_score(X, cluster_labels)
#     print("For n_clusters =", n_clusters,
#           "The average silhouette_score is :", silhouette_avg)
#     silhouette_width[n_clusters-2] = silhouette_avg
# 
#     # Compute the silhouette scores for each sample
#     sample_silhouette_values = silhouette_samples(X, cluster_labels)
# 
#     y_lower = 10
#     for i in range(n_clusters):
#         # Aggregate the silhouette scores for samples belonging to
#         # cluster i, and sort them
#         ith_cluster_silhouette_values = \
#             sample_silhouette_values[cluster_labels == i]
# 
#         ith_cluster_silhouette_values.sort()
# 
#         size_cluster_i = ith_cluster_silhouette_values.shape[0]
#         y_upper = y_lower + size_cluster_i
# 
#         color = cm.tab10(float(i) / n_clusters)
#         ax1.fill_betweenx(np.arange(y_lower, y_upper),
#                           0, ith_cluster_silhouette_values,
#                           facecolor=color, edgecolor=color, alpha=0.7)
# 
#         # Label the silhouette plots with their cluster numbers at the middle
#         ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
# 
#         # Compute the new y_lower for next plot
#         y_lower = y_upper + 10  # 10 for the 0 samples
# 
#     ax1.set_title("The silhouette plot for the various clusters.")
#     ax1.set_xlabel("The silhouette coefficient values")
#     ax1.set_ylabel("Cluster label")
# 
#     # The vertical line for average silhouette score of all the values
#     ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
# 
#     ax1.set_yticks([])  # Clear the yaxis labels / ticks
#     ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
# 
#     # 2nd Plot showing the actual clusters formed
#     L = np.array(cluster_labels, dtype=float)
#     for i in range(len(to_delete)):
#         if np.isfinite(to_delete[i]):
#             L = np.insert(L,to_delete[i].astype(int),np.nan)
#     L_reshape = np.reshape(L,(len(no_ocean_lats),len(no_ocean_lons)))
#    
# 
#     rs.plot_map(L_reshape,no_ocean_lats,no_ocean_lons,linear=True,
#                 vmin=0,vmax=n_clusters,cbarlabel='Cluster', cmap="tab10")
# 
# 
#     # Labeling the clusters
#     centers = clusterer.cluster_centers_
#     # Draw white circles at cluster centers
#     ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
#                 c="white", alpha=1, s=200, edgecolor='k')
# 
#     for i, c in enumerate(centers):
#         ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
#                     s=50, edgecolor='k')
# 
#     ax2.set_title("The visualization of the clustered data.")
#     ax2.set_xlabel("Feature space for the 1st feature")
#     ax2.set_ylabel("Feature space for the 2nd feature")
# 
#     plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
#                   "with n_clusters = %d" % n_clusters),
#                  fontsize=14, fontweight='bold')
# 
# # =============================================================================
# #     plt.savefig('Silhouette Plot' + str(n_clusters) + '.png')
# # =============================================================================
# 
#     plt.show()
#     
# # Plot silhouette width comparison
# plt.plot(range_n_clusters, silhouette_width, 'bx-')
# plt.xlabel('Number of Clusters')
# plt.ylabel('Silhouette Width')
# plt.title('The Silhouette Method showing the optimal number of clusters')
# 
# # =============================================================================
# # plt.savefig('Silhouette Plot for all clusters.png')
# # =============================================================================
# 
# plt.show()
# =============================================================================

