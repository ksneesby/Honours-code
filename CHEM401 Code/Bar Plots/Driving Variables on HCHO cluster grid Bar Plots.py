# import the functions in other python script
import regrid_swaths as rs

# dates and maths
from datetime import datetime
import numpy as np

# for plotting
import matplotlib.pyplot as plt

import calendar

from copy import copy

from calendar import monthrange


def mask_fire(HCHO,fire):
    fmask=fire>5 # mask where there are more than a few fire pixels
    HCHO=np.ma.array(HCHO, mask=fmask)
    return HCHO

no_clusters = 6
month = 1
month_num = "01"

driving_variable = "LAI"    #### SM or EF or LAI

colour = ('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan')

cluster_means = {}
for i in range(no_clusters):
    cluster_means[i] = np.nan
    
spatial_std = {}
for i in range(no_clusters):
    spatial_std[i] = np.nan    
    
cluster_stds = {}
for i in range(no_clusters):
    cluster_stds[i] = np.nan
    
std_of_stds = {}
for i in range(no_clusters):
    std_of_stds[i] = np.nan
    
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



# Create a figure:
plt.figure(figsize=(10,7))

# Plot columns:
plt.subplot(211) # two rows, 1 column, first subplot
rs.plot_map(HCHO,lats,lons,linear=False,vmin=1e14,vmax=1e16,
                cbarlabel='molec/cm2')
plt.title('OMI HCHO Columns')

# plot fire mask:
plt.subplot(223) # two rows, two columns, third subplot
rs.plot_map(fire,lats,lons,linear=False,vmin=1,vmax=1e8, 
                cbarlabel='fire pixels/day')
plt.title('MOD14A1 Fire')

# plot masked by fires:
plt.subplot(224) # 
rs.plot_map(HCHO_masked,lats,lons, linear=False,vmin=1e14,vmax=1e16,
                cbarlabel='molec/cm2')
plt.title('OMI masked by fires>500')


# save the figure:
plt.suptitle(d0.strftime('Plots for %Y %m %d'))
# =============================================================================
# plt.savefig('test_plot.png')
# =============================================================================
plt.close()
print('test_plot.png saved')

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


# month avg 
plt.subplot(311)
HCHO_05_mean=np.nanmean(HCHO,axis=0)
rs.plot_map(HCHO_05_mean,lats,lons,linear=False,
            vmin=1e14,vmax=1e16,cbarlabel='molec/cm2')
plt.title('HCHO averaged over a month')

for i in range(4):
    plt.subplot(323+i)
    rs.plot_map(HCHO[i,:,:],lats,lons,linear=False,
            vmin=1e14,vmax=1e16,cbarlabel='molec/cm2')
#    plt.title(data['time'][i].strftime('OMI hcho %Y %m %d'))

# =============================================================================
# plt.savefig('test_plot2.png')
# =============================================================================
plt.close()
print('test_plot2.png saved')




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
X = np.full((len(Aus_reshape),4), np.nan) 
X[:,0] = Aus_reshape
X[:,3] = Aus_lats_reshape 
X[:,2] = Aus_lons_reshape
X[:,1] = std_reshape



## Remove oceans

from mpl_toolkits.basemap import Basemap
from matplotlib.path import Path


# S W N E
__AUSREGION__=[-45, 108.75, -7, 156.25]
region=__AUSREGION__
map =Basemap(llcrnrlat=region[0], urcrnrlat=region[2], llcrnrlon=region[1], urcrnrlon=region[3],
                  resolution='i', projection='merc')

x, y = map(X[:,2], X[:,3])

locations = np.c_[x, y]

polygons = [Path(p.boundary) for p in map.landpolygons]

result = np.zeros(len(locations), dtype=bool) 

for polygon in polygons:

    result += np.array(polygon.contains_points(locations))


for i in reversed(range(len(result))):
    if result[i] == False:
        X = np.delete(X,i,0)


no_ocean_lons = np.unique(X[:,2])
no_ocean_lats = np.unique(X[:,3])
no_ocean_HCHO = np.full((len(no_ocean_lats),len(no_ocean_lons)), np.nan)
no_ocean_std = np.full((len(no_ocean_lats),len(no_ocean_lons)), np.nan)

for i in range(len(X)):
    value_lon = X[i,2]
    value_lat = X[i,3]
    pos_lon = np.where( no_ocean_lons==value_lon )
    pos_lat = np.where( no_ocean_lats==value_lat )
    no_ocean_HCHO[pos_lat,pos_lon] = X[i,0]
    no_ocean_std[pos_lat,pos_lon] = X[i,1]
    
rs.plot_map(no_ocean_HCHO,no_ocean_lats,no_ocean_lons,linear=False,
            vmin=1e14,vmax=1e16,cbarlabel='molec/cm2')

# =============================================================================
# plt.savefig('test_plot3.png')
# =============================================================================
plt.close()
print('test_plot3.png saved')


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
X = np.full((len(Aus_reshape),4), np.nan) 
X[:,0] = Aus_reshape
X[:,2] = Aus_lats_reshape 
X[:,3] = Aus_lons_reshape
X[:,1] = std_reshape

to_delete = np.full(len(X)*2,np.nan)
count = 0
for y in range(len(X[0])):
    for i in range(len(X)):
        if not np.isfinite(X[i,y]):
            to_delete[count] = i
            count +=1
        
to_delete = np.unique(to_delete)
to_delete = np.lib.pad(to_delete, (0,(len(X)*2-len(to_delete))), 'constant', constant_values=(np.nan))

         
for i in reversed(range(len(X))):
    if np.isfinite(to_delete[i]):
        X = np.delete(X,to_delete[i],0)


## Normalise variables
        
from sklearn import preprocessing

Y = copy(X)
X = preprocessing.RobustScaler().fit_transform(X)


# =============================================================================
# W = np.full((8900,2), np.nan) 
# W[:,1] = X[:,3]
# W[:,0] = X[:,2]
# =============================================================================


HCHO_centres = np.full((6,4),np.nan)

HCHO_centres[0,:] = -0.446915,-0.272873,-0.677987,0.458636
HCHO_centres[1,:] = 0.820176,1.28032,0.822617,-0.0384529
HCHO_centres[2,:] = -0.791179,-0.603774,-0.364873,-0.513872
HCHO_centres[3,:] = -0.116309,0.359271,0.0821762,-0.713959
HCHO_centres[4,:] = 0.438717,0.997839,-0.458883,0.725548
HCHO_centres[5,:] = 0.418992,-0.153473,0.607921,0.0619376


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=no_clusters, init=HCHO_centres)
kmeans = kmeans.fit(X)
labels = kmeans.predict(X)
C = kmeans.cluster_centers_
L = kmeans.labels_
# =============================================================================
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=L)
# ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c='#050505', s=1000)
# =============================================================================
    
import matplotlib.cm as cm
import matplotlib.colors as clr
       
L = np.array(L, dtype=float)
for i in range(len(to_delete)):
    if np.isfinite(to_delete[i]):
        L = np.insert(L,to_delete[i].astype(int),np.nan)
L_reshape = np.reshape(L,(len(no_ocean_lats),len(no_ocean_lons)))

plt.figure(figsize=(15,10))      
plt.subplot(2,2,1)
plt.title("K = " + str(no_clusters))
rs.plot_map(L_reshape,no_ocean_lats,no_ocean_lons,linear=True,
            vmin=0,vmax=no_clusters,cbarlabel='Cluster', cmap="tab10")

norm = clr.Normalize(vmin=0,vmax=no_clusters)
colours = np.full(no_clusters,np.nan,dtype=tuple)
for i in range(no_clusters):
    colours[i] = cm.tab10(norm(i))

######### Read in other data


def read_EF():
    ## Read in data

    import xarray as xr

    ds=xr.open_dataset('EF/Data/regridded-MEGAN2.1_EF.geos.025x03125.ugm-2hr-1.nc')
    
    lats = ds.lat.values
    lons = ds.lon.values
    
    dx = ds.AEF_ISOPRENE.to_dataframe()
    dx = dx.reset_index()
    df = dx.iloc[:,1:]
    
    
    
    ### 
    ### Limit df to just Australian region data ~ S W N E ~ [-45, 108.75, -10, 156.25]
    
    for i in (range(len(df))):
        if df.loc[i,'lat'] <= -10.2:
            lats_max = i
    
    for i in range(len(df)):
        if df.loc[i,'lat'] < -45:
            lats_min = i
    
    
    
    df = df.iloc[lats_min:lats_max,:]
    df = df.reset_index(drop=True)
    
    df = df.sort_values(['lon','lat'])
    df = df.reset_index(drop=True)
    
    
    
    for i in (range(len(df))):
        if df.loc[i,'lon'] <= 154:
            lons_max = i
    
    for i in range(len(df)):
        if df.loc[i,'lon'] < 108.75:
            lons_min = i
     
    df = df.iloc[lons_min:lons_max,:]
    df = df.reset_index(drop=True)
        
    Aus_lons = np.unique(df.loc[:,'lon'])
    Aus_lats = np.unique(df.loc[:,'lat'])
    
    
    
    X = copy(df)
    X = X.sort_values(['lat','lon'])
    X = X.reset_index(drop=True)

    
    
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
    no_ocean_data = np.full((len(no_ocean_lats),len(no_ocean_lons)), np.nan)
    
    
    for i in range(len(df)):
        value_lon = df.loc[i,'lon']
        value_lat = df.loc[i,'lat']
        pos_lon = np.where( no_ocean_lons==value_lon )
        pos_lat = np.where( no_ocean_lats==value_lat )
        no_ocean_data[pos_lat,pos_lon] = df.iloc[i,2]
      
    

    
    
    no_ocean_data_reshape = np.reshape(no_ocean_data,(len(no_ocean_data)*len(no_ocean_data[0])))
    
    no_ocean_lats_reshape = np.repeat(no_ocean_lats, len(no_ocean_lons))
    no_ocean_lons_reshape = np.tile(no_ocean_lons, len(no_ocean_lats))
    X = np.full((len(no_ocean_data_reshape),3), np.nan) 
    X[:,0] = no_ocean_lats_reshape
    X[:,1] = no_ocean_lons_reshape
    X[:,2] = no_ocean_data_reshape
    
    
    
    
    to_delete = np.full(len(X)*3,np.nan)
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
            
    return (no_ocean_data,no_ocean_lats,no_ocean_lons)


def read_SM():
    
    import xarray as xr
    
    ds05=xr.open_mfdataset("Soil Moisture/Data/2005/regridded-ESACCI-SOILMOISTURE-L3S-SSMV-COMBINED-2005" + month_num + "*.nc",concat_dim="time",autoclose=True)
    ds05 = ds05.drop(("sm_uncertainty","dnflag","flag","freqbandID","mode","sensor","time"))
    ds05=ds05.mean(dim='time',skipna=True)
    
    ds06=xr.open_mfdataset("Soil Moisture/Data/2006/regridded-ESACCI-SOILMOISTURE-L3S-SSMV-COMBINED-2006" + month_num + "*.nc",concat_dim="time",autoclose=True)
    ds06 = ds06.drop(("sm_uncertainty","dnflag","flag","freqbandID","mode","sensor","time"))
    ds06=ds06.mean(dim='time',skipna=True)
    ds07=xr.open_mfdataset("Soil Moisture/Data/2007/regridded-ESACCI-SOILMOISTURE-L3S-SSMV-COMBINED-2007" + month_num + "*.nc",concat_dim="time",autoclose=True)
    ds07 = ds07.drop(("sm_uncertainty","dnflag","flag","freqbandID","mode","sensor","time"))
    ds07=ds07.mean(dim='time',skipna=True)
    ds08=xr.open_mfdataset("Soil Moisture/Data/2008/regridded-ESACCI-SOILMOISTURE-L3S-SSMV-COMBINED-2008" + month_num + "*.nc",concat_dim="time",autoclose=True)
    ds08 = ds08.drop(("sm_uncertainty","dnflag","flag","freqbandID","mode","sensor","time"))
    ds08=ds08.mean(dim='time',skipna=True)
    ds09=xr.open_mfdataset("Soil Moisture/Data/2009/regridded-ESACCI-SOILMOISTURE-L3S-SSMV-COMBINED-2009" + month_num + "*.nc",concat_dim="time",autoclose=True)
    ds09 = ds09.drop(("sm_uncertainty","dnflag","flag","freqbandID","mode","sensor","time"))
    ds09=ds09.mean(dim='time',skipna=True)
    ds10=xr.open_mfdataset("Soil Moisture/Data/2010/regridded-ESACCI-SOILMOISTURE-L3S-SSMV-COMBINED-2010" + month_num + "*.nc",concat_dim="time",autoclose=True)
    ds10 = ds10.drop(("sm_uncertainty","dnflag","flag","freqbandID","mode","sensor","time"))
    ds10=ds10.mean(dim='time',skipna=True)
    ds11=xr.open_mfdataset("Soil Moisture/Data/2011/regridded-ESACCI-SOILMOISTURE-L3S-SSMV-COMBINED-2011" + month_num + "*.nc",concat_dim="time",autoclose=True)
    ds11 = ds11.drop(("sm_uncertainty","dnflag","flag","freqbandID","mode","sensor","time"))
    ds11=ds11.mean(dim='time',skipna=True)
    ds12=xr.open_mfdataset("Soil Moisture/Data/2012/regridded-ESACCI-SOILMOISTURE-L3S-SSMV-COMBINED-2012" + month_num + "*.nc",concat_dim="time",autoclose=True)
    ds12 = ds12.drop(("sm_uncertainty","dnflag","flag","freqbandID","mode","sensor","time"))
    ds12=ds12.mean(dim='time',skipna=True)
    ds13=xr.open_mfdataset("Soil Moisture/Data/2013/regridded-ESACCI-SOILMOISTURE-L3S-SSMV-COMBINED-2013" + month_num + "*.nc",concat_dim="time",autoclose=True)
    ds13 = ds13.drop(("sm_uncertainty","dnflag","flag","freqbandID","mode","sensor","time"))
    ds13=ds13.mean(dim='time',skipna=True)
    ds14=xr.open_mfdataset("Soil Moisture/Data/2014/regridded-ESACCI-SOILMOISTURE-L3S-SSMV-COMBINED-2014" + month_num + "*.nc",concat_dim="time",autoclose=True)
    ds14 = ds14.drop(("sm_uncertainty","dnflag","flag","freqbandID","mode","sensor","time"))
    ds14=ds14.mean(dim='time',skipna=True)
    
    ds = xr.concat([ds05,ds06,ds07,ds08,ds09,ds10,ds11,ds12,ds13,ds14], dim='year')
    
    
    std=ds.std(dim='year',skipna=True)
    ds=ds.mean(dim='year',skipna=True)
    
    
    
    lats = ds.lat
    lons = ds.lon
    
    lons = np.unique(lons)
    lats = np.unique(lats)
    
    
    sm_std = std.to_dataframe()
    sm_std = sm_std.reset_index()
    sm_std = sm_std.values
    
    sm_std=np.delete(sm_std,(0,1),axis=1)
    sm_std = sm_std.ravel()
    sm_std = np.reshape(sm_std,(len(lats),len(lons)))
    
    sm_mean = ds.to_dataframe()
    sm_mean = sm_mean.reset_index()
    sm_mean = sm_mean.values
    
    
    sm_mean=np.delete(sm_mean,(0,1),axis=1)
    sm_mean = sm_mean.ravel()
    sm_mean = np.reshape(sm_mean,(len(lats),len(lons)))
    

    
    
    
    # =============================================================================
    # plt.savefig('test_plot2.png')
    # plt.close()
    # print('test_plot2.png saved')
    # =============================================================================
    
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
    
    sm_mean_Aus = sm_mean[lats_min:lats_max,lons_min:lons_max]
    Aus_lats = lats[lats_min:lats_max]
    Aus_lons = lons[lons_min:lons_max]
    sm_std_Aus = sm_std[lats_min:lats_max,lons_min:lons_max]
    
    
    Aus_reshape = np.reshape(sm_mean_Aus,(len(sm_mean_Aus)*len(sm_mean_Aus[0])))
    std_reshape = np.reshape(sm_std_Aus,(len(sm_std_Aus)*len(sm_std_Aus[0])))
    Aus_lats_reshape = np.repeat(Aus_lats, len(Aus_lons))
    Aus_lons_reshape = np.tile(Aus_lons, len(Aus_lats))
    X = np.full((len(Aus_reshape),4), np.nan) 
    X[:,0] = Aus_reshape
    X[:,3] = Aus_lats_reshape 
    X[:,2] = Aus_lons_reshape
    X[:,1] = std_reshape
    
    
    ## Remove oceans
    
    from mpl_toolkits.basemap import Basemap
    from matplotlib.path import Path
    
    
    # S W N E
    __AUSREGION__=[-45, 108.75, -7, 156.25]
    region=__AUSREGION__
    map =Basemap(llcrnrlat=region[0], urcrnrlat=region[2], llcrnrlon=region[1], urcrnrlon=region[3],
                      resolution='i', projection='merc')
    
    x, y = map(X[:,2], X[:,3])
    
    locations = np.c_[x, y]
    
    polygons = [Path(p.boundary) for p in map.landpolygons]
    
    result = np.zeros(len(locations), dtype=bool) 
    
    for polygon in polygons:
    
        result += np.array(polygon.contains_points(locations))
    
    
    for i in reversed(range(len(result))):
        if result[i] == False:
            X = np.delete(X,i,0)
    
    
    no_ocean_lons = np.unique(X[:,2])
    no_ocean_lats = np.unique(X[:,3])
    no_ocean_sm = np.full((len(no_ocean_lats),len(no_ocean_lons)), np.nan)
    no_ocean_std = np.full((len(no_ocean_lats),len(no_ocean_lons)), np.nan)
    
    for i in range(len(X)):
        value_lon = X[i,2]
        value_lat = X[i,3]
        pos_lon = np.where( no_ocean_lons==value_lon )
        pos_lat = np.where( no_ocean_lats==value_lat )
        no_ocean_sm[pos_lat,pos_lon] = X[i,0]
        no_ocean_std[pos_lat,pos_lon] = X[i,1]
        
    
    # =============================================================================
    # plt.savefig('test_plot3.png')
    # plt.close()
    # print('test_plot3.png saved')
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
    
    Aus_reshape = np.reshape(no_ocean_sm,(len(no_ocean_sm)*len(no_ocean_sm[0])))
    std_reshape = np.reshape(no_ocean_std,(len(no_ocean_std)*len(no_ocean_std[0])))
    Aus_lats_reshape = np.repeat(no_ocean_lats, len(no_ocean_lons))
    Aus_lons_reshape = np.tile(no_ocean_lons, len(no_ocean_lats))
    X = np.full((len(Aus_reshape),4), np.nan) 
    X[:,0] = Aus_reshape
    X[:,2] = Aus_lats_reshape 
    X[:,3] = Aus_lons_reshape
    X[:,1] = std_reshape
    
    to_delete = np.full(len(X),np.nan)
    count = 0
    for y in range(len(X[0])):
        for i in range(len(X)):
            if not np.isfinite(X[i,y]):
                to_delete[count] = i
                count +=1
            
    to_delete = np.unique(to_delete)
    to_delete = np.lib.pad(to_delete, (0,(len(X)-len(to_delete))), 'constant', constant_values=(np.nan))
    
             
    for i in reversed(range(len(X))):
        if np.isfinite(to_delete[i]):
            X = np.delete(X,to_delete[i],0)

    no_ocean_data = copy(no_ocean_sm)
    
    
    
    return (no_ocean_data, no_ocean_std, no_ocean_lats, no_ocean_lons)


def read_LAI():
    
    from netCDF4 import Dataset
    
    ds05 = Dataset("LAI2/Data/MODIS.LAI.vBNU.generic.025x03125.2005.nc")
    ds06 = Dataset("LAI2/Data/MODIS.LAI.vBNU.generic.025x03125.2006.nc")
    ds07 = Dataset("LAI2/Data/MODIS.LAI.vBNU.generic.025x03125.2007.nc")
    ds08 = Dataset("LAI2/Data/MODIS.LAI.vBNU.generic.025x03125.2008.nc")
    ds09 = Dataset("LAI2/Data/MODIS.LAI.vBNU.generic.025x03125.2009.nc")
    ds10 = Dataset("LAI2/Data/MODIS.LAI.vBNU.generic.025x03125.2010.nc")
    ds11 = Dataset("LAI2/Data/MODIS.LAI.vBNU.generic.025x03125.2011.nc")
    ds12 = Dataset("LAI2/Data/MODIS.LAI.vBNU.generic.025x03125.2012.nc")
    ds13 = Dataset("LAI2/Data/MODIS.LAI.vBNU.generic.025x03125.2013.nc")
    ds14 = Dataset("LAI2/Data/MODIS.LAI.vBNU.generic.025x03125.2014.nc")
    
    
    lons = ds05.variables['lon'][:]
    lats = ds05.variables['lat'][:]
    time = ds10.variables['time'][:]
    LAI = ds05.variables['MODIS'][:]
    
    ds05 = LAI[4*month-4:4*month,:,:]
    ds05 = np.nanmean(ds05,axis=0)
    LAI = ds06.variables['MODIS'][:]
    ds06 = LAI[4*month-4:4*month,:,:]
    ds06 = np.nanmean(ds06,axis=0)
    LAI = ds07.variables['MODIS'][:]
    ds07 = LAI[4*month-4:4*month,:,:]
    ds07 = np.nanmean(ds07,axis=0)
    LAI = ds08.variables['MODIS'][:]
    ds08 = LAI[4*month-4:4*month,:,:]
    ds08 = np.nanmean(ds08,axis=0)
    LAI = ds09.variables['MODIS'][:]
    ds09 = LAI[4*month-4:4*month,:,:]
    ds09 = np.nanmean(ds09,axis=0)
    LAI = ds10.variables['MODIS'][:]
    ds10 = LAI[4*month-4:4*month,:,:]
    ds10 = np.nanmean(ds10,axis=0)
    LAI = ds11.variables['MODIS'][:]
    ds11 = LAI[4*month-4:4*month,:,:]
    ds11 = np.nanmean(ds11,axis=0)
    LAI = ds12.variables['MODIS'][:]
    ds12 = LAI[4*month-4:4*month,:,:]
    ds12 = np.nanmean(ds12,axis=0)
    LAI = ds13.variables['MODIS'][:]
    ds13 = LAI[4*month-4:4*month,:,:]
    ds13 = np.nanmean(ds13,axis=0)
    LAI = ds14.variables['MODIS'][:]
    ds14 = LAI[4*month-4:4*month,:,:]
    ds14 = np.nanmean(ds14,axis=0)
    
    LAI_mean = np.stack((ds05,ds06,ds07,ds08,ds09,ds10, ds11, ds12, ds13, ds14),axis=0)
    LAI_std = np.nanstd(LAI_mean,axis=0)
    LAI_mean=np.nanmean(LAI_mean,axis=0)
     
    
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
    
    LAI_mean_Aus = LAI_mean[lats_min:lats_max,lons_min:lons_max]
    Aus_lats = lats[lats_min:lats_max]
    Aus_lons = lons[lons_min:lons_max]
    LAI_std_Aus = LAI_std[lats_min:lats_max,lons_min:lons_max]
    
    
    Aus_reshape = np.reshape(LAI_mean_Aus,(len(LAI_mean_Aus)*len(LAI_mean_Aus[0])))
    std_reshape = np.reshape(LAI_std_Aus,(len(LAI_std_Aus)*len(LAI_std_Aus[0])))
    Aus_lats_reshape = np.repeat(Aus_lats, len(Aus_lons))
    Aus_lons_reshape = np.tile(Aus_lons, len(Aus_lats))
    X = np.full((len(Aus_reshape),4), np.nan) 
    X[:,0] = Aus_reshape
    X[:,3] = Aus_lats_reshape 
    X[:,2] = Aus_lons_reshape
    X[:,1] = std_reshape
    
    ## Remove oceans
    
    from mpl_toolkits.basemap import Basemap
    from matplotlib.path import Path
    
    
    # S W N E
    __AUSREGION__=[-45, 108.75, -7, 156.25]
    region=__AUSREGION__
    map =Basemap(llcrnrlat=region[0], urcrnrlat=region[2], llcrnrlon=region[1], urcrnrlon=region[3],
                      resolution='i', projection='merc')
    
    x, y = map(X[:,2], X[:,3])
    
    locations = np.c_[x, y]
    
    polygons = [Path(p.boundary) for p in map.landpolygons]
    
    result = np.zeros(len(locations), dtype=bool) 
    
    for polygon in polygons:
    
        result += np.array(polygon.contains_points(locations))
    
    
    for i in reversed(range(len(result))):
        if result[i] == False:
            X = np.delete(X,i,0)
    
    
    no_ocean_lons = np.unique(X[:,2])
    no_ocean_lats = np.unique(X[:,3])
    no_ocean_LAI = np.full((len(no_ocean_lats),len(no_ocean_lons)), np.nan)
    no_ocean_std = np.full((len(no_ocean_lats),len(no_ocean_lons)), np.nan)
    
    for i in range(len(X)):
        value_lon = X[i,2]
        value_lat = X[i,3]
        pos_lon = np.where( no_ocean_lons==value_lon )
        pos_lat = np.where( no_ocean_lats==value_lat )
        no_ocean_LAI[pos_lat,pos_lon] = X[i,0]
        no_ocean_std[pos_lat,pos_lon] = X[i,1]
        
    
    
    
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
    
    Aus_reshape = np.reshape(no_ocean_LAI,(len(no_ocean_LAI)*len(no_ocean_LAI[0])))
    std_reshape = np.reshape(no_ocean_std,(len(no_ocean_std)*len(no_ocean_std[0])))
    Aus_lats_reshape = np.repeat(no_ocean_lats, len(no_ocean_lons))
    Aus_lons_reshape = np.tile(no_ocean_lons, len(no_ocean_lats))
    X = np.full((len(Aus_reshape),4), np.nan) 
    X[:,0] = Aus_reshape
    X[:,2] = Aus_lats_reshape 
    X[:,3] = Aus_lons_reshape
    X[:,1] = std_reshape
    
    to_delete = np.full(len(X),np.nan)
    count = 0
    for y in range(len(X[0])):
        for i in range(len(X)):
            if not np.isfinite(X[i,y]):
                to_delete[count] = i
                count +=1
            
    to_delete = np.unique(to_delete)
    to_delete = np.lib.pad(to_delete, (0,(len(X)-len(to_delete))), 'constant', constant_values=(np.nan))
    
             
    for i in reversed(range(len(X))):
        if np.isfinite(to_delete[i]):
            X = np.delete(X,to_delete[i],0)
            
    no_ocean_data = copy(no_ocean_LAI)
    
    return (no_ocean_data, no_ocean_std, no_ocean_lats, no_ocean_lons)



###### Read data
    
if driving_variable == "LAI":
    no_ocean_data, no_ocean_std, no_ocean_lats, no_ocean_lons = read_LAI()
elif driving_variable == "SM":
    no_ocean_data, no_ocean_std, no_ocean_lats, no_ocean_lons = read_SM()
elif driving_variable == "EF":
    no_ocean_data,no_ocean_lats,no_ocean_lons = read_EF()






####### Bar Plot

data_reshape = np.reshape(no_ocean_data,(len(no_ocean_data[0])*len(no_ocean_data)))
if driving_variable == "LAI" or driving_variable == "SM":
    std_reshape = np.reshape(no_ocean_std, (len(no_ocean_data[0])*len(no_ocean_data)))
lats_reshape = np.repeat(no_ocean_lats, len(no_ocean_lons))
lons_reshape = np.tile(no_ocean_lons, len(no_ocean_lats))
X = np.full((len(data_reshape),4), np.nan) 
X[:,0] = data_reshape
if driving_variable == "LAI" or driving_variable == "SM":
    X[:,1] = std_reshape
X[:,3] = lats_reshape 
X[:,2] = lons_reshape

for i in range(no_clusters):
    cluster = i
    is_cluster = np.where(L==cluster,True,False)
    cluster_means[cluster] = data_reshape[is_cluster]
    spatial_std[cluster] = np.nanstd(cluster_means[cluster])
    cluster_means[cluster] = np.nanmean(cluster_means[cluster])
   
    if driving_variable == "LAI" or driving_variable == "SM":
        cluster_stds[cluster] = std_reshape[is_cluster]
        std_of_stds[cluster] = np.nanstd(cluster_stds[cluster])
        cluster_stds[cluster] = np.nanmean(cluster_stds[cluster])


cluster_means_ = np.full(no_clusters,np.nan)
spatial_std_ = np.full(no_clusters,np.nan)
if driving_variable == "LAI" or driving_variable == "SM":
    cluster_stds_ = np.full(no_clusters,np.nan)
    std_of_stds_ = np.full(no_clusters,np.nan)
colour_ = np.full(no_clusters,"", dtype = object)


for i in range(no_clusters):
    cluster_means_[i] = cluster_means[i]
    spatial_std_[i] = spatial_std[i]
    if driving_variable == "LAI" or driving_variable == "SM":
        cluster_stds_[i] = cluster_stds[i]
        std_of_stds_[i] = std_of_stds[i]

plt.subplot(2,2,3)
plt.title("Mean")
plt.bar(np.arange(0,no_clusters,1),cluster_means_,yerr=spatial_std_,color = colours)
if driving_variable == "LAI" or driving_variable == "SM":
    plt.subplot(2,2,4)
    plt.title("Standard Deviation")
    plt.bar(np.arange(0,no_clusters,1),cluster_stds_,yerr=std_of_stds_,color = colours)
plt.savefig('Bar Plots/' + driving_variable + ' ' + month_text +' K = '+ str(no_clusters) + '.png')
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
# # plt.savefig('Silhouette and Elbow Plots/'+ str(month) + ' ' + month_text + ' 2005-2014/Elbow Plot.png')
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
# #     plt.savefig('Silhouette and Elbow Plots/'+ str(month) + ' ' + month_text + ' 2005-2014/Silhouette Plot' + str(n_clusters) + '.png')
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
# # plt.savefig('Silhouette and Elbow Plots/'+ str(month) + ' ' + month_text + ' 2005-2014/Silhouette Plot for all clusters.png')
# # =============================================================================
# 
# plt.show()
# =============================================================================

