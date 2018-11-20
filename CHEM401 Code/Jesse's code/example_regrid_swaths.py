# import the functions in other python script
import regrid_swaths as rs

# dates and maths
from datetime import datetime
import numpy as np

# for plotting
import matplotlib.pyplot as plt

## Read a single day and plot the data
##

d0=datetime(2005,1,1)

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
plt.savefig('test_plot.png')
plt.close()
print('test_plot.png saved')

###
### Now grab a bunch of days at once:


dN=datetime(2005,1,4)

data,attr=rs.read_regridded_swath(d0,dN)

print(data.keys()) # shows what keys are in our data structure

print(data['VC_C'].shape) # shape of VC_C array

print(attr['VC_C']) # attributes for VC_C

HCHO=data['VC_C']
fire=data['fires']
lats=data['lats']
lons=data['lons']

plt.figure(figsize=(10,12))

# month avg 
plt.subplot(311)
HCHO_mean=np.nanmean(HCHO,axis=0)
rs.plot_map(HCHO_mean,lats,lons,linear=False,
            vmin=1e14,vmax=1e16,cbarlabel='molec/cm2')
plt.title('HCHO averaged over a few days')

for i in range(4):
    plt.subplot(323+i)
    rs.plot_map(HCHO[i,:,:],lats,lons,linear=False,
            vmin=1e14,vmax=1e16,cbarlabel='molec/cm2')
    plt.title(data['time'][i].strftime('OMI hcho %Y %m %d'))

plt.savefig('test_plot2.png')
plt.close()
print('test_plot2.png saved')


# Try reading just VC_C
dN=datetime(2005,4,1)
vcc,fires,days,lats,lons = rs.read_key(d0, dN,key='VC_C')
# avg global vcc with/without fire columns:
vcc_fire=np.nanmean(vcc, axis=(1,2))
vcc_nofire=np.copy(vcc)
vcc_nofire[fires>0]=np.NaN
vcc_nofire=np.nanmean(vcc_nofire,axis=(1,2))
plt.figure()
plt.plot(days,vcc_fire,label='VC_C')
plt.plot(days,vcc_nofire,label='VC_C fire removed')
plt.legend()
plt.savefig('test_plot3.png')
plt.close()
print('test_plot3.png saved')

