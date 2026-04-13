
import numpy as np
import xarray as xr
import pyproj



# Path to original data (as downloaded)
p_mask = 'data/cells_to_include.nc'
p_dist = 'data/dist.nc'
    
mask = xr.open_dataarray(p_mask)

mask_stacked = mask.where(mask).stack(lon_lat=("lon", "lat")).dropna("lon_lat")
lon_lats     = mask_stacked.lon_lat.values
n    = len(lon_lats)
dist = np.ones((n, n), dtype="int16")

print("Number of bytes of dist-matrix: ", dist.nbytes)

g = pyproj.Geod(ellps='WGS84')

for i, (lon1, lat1) in enumerate(lon_lats): 
    for j in range(i, n):
        lon2, lat2 = lon_lats[j]
        d = g.inv(lon1, lat1, lon2, lat2)[2]
        if d == 0: d = 32767
        else:      d = d//1000 # save in km
        if np.isnan(d): d = 32767
        dist[i,j] = d
        dist[j,i] = d
        
xr.DataArray(dist, dims = ("lon_lat", "lon_lat_y")).to_netcdf(p_dist, mode="w")