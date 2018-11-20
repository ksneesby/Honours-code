This folder contains scripts which create cluster maps for each variable. These scripts use a function from "regrid_swaths" from Jesse Greenslade's code to create the pcolormesh maps, and the HCHO script uses functions from that script to read in the data. These scripts read in data for each variable, reduce data to just the Australian region, remove ocean data, normalise the data, perform K-Means clustering on the data, and plot the cluster map. At the end is included code which was not written by me which performs the elbow method and silhouette method to attempt to determine the optimal number of clusters. Scripts included in this folder are:

* Climate Clustering
Using data from the Koppen-Geiger climate classification

* EF Clustering
Using data from the GEOS-Chem transport model

* HCHO Clustering
Using data obtained from Jesse Greenslade who processed data from NASA.

* LAI Clustering
Using data from NASA's MODIS

* LAI2 Clustering
Using data from NASA's MODIS

* PFT & LAI Clustering
Combines both PFT data and the LAI data used in LAI2 Clustering. A multiplication factor is likely needed to be included as clustering is weighted toward PFT data.

* PFT Clustering
Using data from the Community Land Model version 4.0.

* Soil Moisture Clustering
Using data from the European Space Agency.