The following subfolders are contained in this folder:
* Bar Plots
Contains scripts which create bar plots showing the average mean and interannual standard deviation per cluster for each variable, creates bar plots for each variable showing average mean and interannual standard deviation for each cluster using the HCHO clusters, and creates bar plots showing the PFT breakdown for each cluster.

* Clustering
Contains scripts which create cluster maps for the Australian landmass using K-means clustering for several variables including HCHO, soil moisture, plant functional type, MEGAN emission factors, and leaf area index.

* Difference Maps
Contains scripts which compare cluster maps, where all clusters are compared simultaneously. As clusters don't align perfectly across variables, differences depend heavily on cluster assignmnet.

* Focused Difference Maps
Contains scripts which compare cluster maps, where only one cluster is compared at a time. These scripts compare the southwest cluster, the eastern cluster, and the northern cluster.

* Jesse's code
Contains code written by Jesse Greenslade. Functions from "regrid_swaths" were used in my scripts to read in OMI HCHO data and to create pcolormesh maps.