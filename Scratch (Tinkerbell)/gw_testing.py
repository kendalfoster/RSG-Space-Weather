import os
import numpy as np
import scipy as sp
import pandas as pd
import xarray as xr # if gives warning/error, just rerun
import matplotlib.pyplot as plt
from scipy import signal
from numpy import hstack
import spaceweather.analysis.gen_data as gen_data
import spaceweather.analysis.cca as sac
import spaceweather.analysis.data_funcs as sad
import spaceweather.analysis.threshold as sat

import spaceweather.visualisation.heatmaps as svh

def inter_phase_dir_corr(ds,station1,station2,wind_start1,wind_start2,readings=None):
     #check if readings are provided
     if readings is None:
         readings = ['N', 'E', 'Z']

     # universally necessary things
     num_read = len(readings)

     # setup row array for the correlation coefficients
     cca_coeffs = np.zeros(shape = (1, num_read), dtype = float)

     # get readings for the station
     data = sm.window(ds,128)
     data1 = data.measurements.loc[dict(station = station1)][dict(win_start = wind_start1)]
     data2 = data.measurements.loc[dict(station = station2)][dict(win_start = wind_start2)]


#data1 = data.measurements.loc[dict(station = station1)].loc[dict(win_start = wind[wind_start1])]
#data2 = data.measurements.loc[dict(station = station2)].loc[dict(win_start = wind[wind_start2])]
     #Calculate the cannonical correlation between the directional meaurements on each station
     for i in range(num_read):
         first_read = data1[:,i].data
         first_read = np.reshape(first_read, newshape=[len(first_read),1])

         second_read = data2[:,i].data
         second_read = np.reshape(second_read, newshape=[len(second_read),1])

         temp_cca = rcca.CCA(kernelcca = False, reg = 0., numCC = 1)
         cca_coeffs[0,i] = abs(temp_cca.train([first_read, second_read]).cancorrs[0])
     return cca_coeffs


##### Function that finds the index of the point where the phases are at their highest
#     correlation
#     input: ds - dataset output from mag_csv_to_Dataset Function
#           station 1 - 3 letter code for the station 1 as a string, ex: 'BLC'
#           station 2 - 3 letter code for the station 2 as a string, ex: 'BLC'
#           wind_start1 - Index of start of both winow
#    output: shift - the amount of shfit needed to move station 2's window inline


def phase_finder(ds, station1, station2, start):

    ### Get the data windows
    data = sm.window(ds,128)
    data1 = data.measurements.loc[dict(station = station1)]
    data2 = data.measurements.loc[dict(station = station2)]

    ## Set up matrix to put our correlation parameters into
    corr_coeff = np.zeros(shape = (21), dtype = float)


    ## Shift the second window amongst the first one and caluclate mean
    ## of the correlation readings for each shift
    for i in range(21):
         wind2 = start - 10 + i
         x = inter_phase_dir_corr(ds,station1,station2,start,wind2)
         corr_coeff[i] = np.mean(x)

    ## Find where the correlations are highest
    s = np.where(corr_coeff == np.amax(corr_coeff))
    shift = -10 + s[0][0]
    return shift





# def angle(ts1,ts2):
#     # input 2 time series, of the same length
#     # each data point in the time series should be a 3-dimensional vector with N, E and Z components
#     # return np.rad2deg(subspace_angles(bbb_a, bbb_b))
#      return np.rad2deg(subspace_angles(ts1, ts2))



ds1 = sad.csv_to_Dataset(csv_file = "Data/20190403-00-22-supermag.csv",MLT = True, MLAT = True)
ds2 = sad.csv_to_Dataset(csv_file = "Data/20010305-16-38-supermag.csv",MLT = True, MLAT = True)
ds = sad.csv_to_Dataset(csv_file = "Data/20190521-14-08-supermag.csv",MLT = True, MLAT = True)

components=['N', 'E', 'Z']

scratch = sac.lag_mat(ds, lag_range=10, station1 = 'NAL', station2 = 'LYR', win_len=128,plot=True)


scratch_lag_coeffs = np.array(scratch[0].lag_coeffs)
scratch_lag_coeffs[0][340]


for i in range(21):
    for j in range(573):
        if scratch_lag_coeffs[i][j] < -0.2:
            print('%d, %d, %d' % (i,j,scratch_lag_coeefs[i][j]))


lag_range=10
station1 = 'NAL'
station2 = 'LYR'
win_len=128
plot=True

nt = len(ds.time.values)
if nt < win_len + 2*lag_range:
    print('Error: ds timeseries < win_len + 2*lag_range')
    # return 'Error: ds timeseries < win_len + 2*lag_range'

# check if stations are provided
stations = ds.station.values
if len(stations) <= 1:
    print('Error: only one station in Dataset')
    # return 'Error: only one station in Dataset'
if station1 is None:
    print('No station1 provided; using station1 = %s' % (stations[0]))
    station1 = stations[0]
    if station2 is None:
        print('No station2 provided; using station2 = %s' % (stations[1]))
        station2 = stations[1]
elif station2 is None and not station1 == stations[0]:
    print('No station2 provided; using station2 = %s' % (stations[0]))
    station2 = stations[0]
elif station2 is None and station1 == stations[0]:
    print('No station2 provided; using station2 = %s' % (stations[1]))
    station2 = stations[1]

# Select the stations and window the data
ds = ds.loc[dict(station = [station1,station2])]
windowed = sad.window(ds,win_len)
ts1 = windowed.loc[dict(station = station1)].measurements
ts2 = windowed.loc[dict(station = station2)].measurements
ts1 = ts1.transpose('win_len', 'component', 'win_start')
ts2 = ts2.transpose('win_len', 'component', 'win_start')

# Set up array
time = range(lag_range+1, len(windowed.win_start)-lag_range+1)

lag = range(-lag_range, lag_range+1)
corr = np.zeros(shape = (len(lag), len(time)))

# Calculate correlations
for j in range(len(time)):
    for i in range(len(lag)):
        ts1_temp = ts1[dict(win_start = time[j]-1)]
        ts2_temp = ts2[dict(win_start = time[j]+lag[i]-1)]
        # run cca, suppress rcca output
        temp_cca = rcca.CCA(kernelcca = False, reg = 0., numCC = 1, verbose = False)
        ccac = temp_cca.train([ts1_temp, ts2_temp])
        corr[i,j] = ccac.cancorrs[0]

i=0
j=340
scratch_ts1_temp = ts1[dict(win_start = time[j]-1)]
scratch_ts2_temp = ts2[dict(win_start = time[j]+lag[i]-1)]
temp_cca = rcca.CCA(kernelcca = False, reg = 0., numCC = 1, verbose = False)
scratch__ccac = temp_cca.train([scratch_ts1_temp, scratch_ts2_temp])
scratch__ccac.cancorrs[0]
scratch__ccac.ws

np.savetxt("old_cca_component_1.csv", scratch__ccac.comps[0], delimiter=",")
np.savetxt("old_cca_component_2.csv", scratch__ccac.comps[1], delimiter=",")


lag_mat = xr.Dataset(data_vars = {'lag_coeffs': (['lag', 'time_win'], corr)},
                     coords = {'lag': lag,
                               'time_win': time})

# plot adjacency matrix
if plot:
    fig = svh.plot_lag_mat(lag_mat = lag_mat, time_win = time, lag = lag)
    # return lag_mat, fig
# else:
    # return lag_mat

lag_mat.lag_coeffs[0][340]

ds

len(ds.loc[dict(station = 'NAL')].measurements)

scratch_x = ds.loc[dict(station = 'NAL')].measurements[350:478]

scratch_y = ds.loc[dict(station = 'LYR')].measurements[340:468]



scratch_temp_cca = rcca.CCA(kernelcca = False, reg = 0., numCC = 1, verbose = False)
scratch_ccac = temp_cca.train([scratch_x, scratch_y])
scratch_ccac.cancorrs[0]

rdc.cca(scratch_x_array,scratch_y_array)

rdc.rdc(scratch_x_array,scratch_y_array)




from sklearn.cross_decomposition import CCA as new_cca

scratch_x_array = np.array(scratch_x)
scratch_y_array = np.array(scratch_y)

scratch_cca = new_cca(n_components=1)
scratch_cca.fit(scratch_x_array, scratch_y_array)
scratch_result_x_c, scratch_result_y_c = scratch_cca.transform(scratch_x_array, scratch_y_array)
scratch_result_x_c

scratch_result = scratch_cca.fit_transform(scratch_x_array, scratch_y_array)
np.shape(scratch_result)
scratch_result
scratch_result_2 = [[scratch_result[0][i][0], scratch_result[1][i][0]] for i in range(128)]
scratch_result_2

np.savetxt("new_cca_components.csv", scratch_result_2, delimiter=",")


x_matrix = np.array(scratch_x[:3])
x_matrix

y_matrix = np.array(scratch_y[:3])
y_matrix

np.savetxt("old_cca_a.csv",np.linalg.solve(x_matrix,scratch__ccac.comps[0][:3]), delimiter=",")
np.savetxt("old_cca_b.csv",np.linalg.solve(y_matrix,scratch__ccac.comps[1][:3]), delimiter=",")


x_matrix

b_x = [scratch_result_2[i][0] for i in range(3)]
b_y = [scratch_result_2[i][1] for i in range(3)]

np.savetxt("new_cca_a.csv",np.linalg.solve(x_matrix,b_x), delimiter=",")
np.savetxt("new_cca_b.csv",np.linalg.solve(y_matrix,b_y), delimiter=",")

np.corrcoef(scratch_result_x_c.T, scratch_result_y_c.T)[0,1]






ds = sad.csv_to_Dataset(csv_file = "Data/20190521-14-08-supermag.csv",MLT = True, MLAT = True)

scratch_x = ds.loc[dict(station = 'NAL')].measurements[350:478]

scratch_y = ds.loc[dict(station = 'LYR')].measurements[340:468]


# old cca
scratch_temp_cca = rcca.CCA(kernelcca = False, reg = 0., numCC = 2, verbose = False)
scratch_ccac = scratch_temp_cca.train([scratch_x, scratch_y])
scratch_ccac.cancorrs[0]

# new cca
from sklearn.cross_decomposition import CCA as new_cca

scratch_x_array = np.array(scratch_x)
scratch_y_array = np.array(scratch_y)

scratch_cca = new_cca(n_components=1)
scratch_cca.fit(scratch_x_array, scratch_y_array)
scratch_result_x_c, scratch_result_y_c = scratch_cca.transform(scratch_x_array, scratch_y_array)

np.shape(scratch_result_x_c)

np.corrcoef(scratch_result_x_c.T, scratch_result_y_c.T)[0,1]

coeffs, scratch_u, scratch_v = rdc.cca(scratch_x_array,scratch_y_array)

np.savetxt("scratch_u.csv",scratch_u, delimiter=",")
np.savetxt("scratch_v.csv",scratch_v, delimiter=",")

scratch_u
scratch_v
coeffs


test_arr_1 = [[4,7,9],[2,5,2],[3,3,7],[4,8,3],[1,1,1],[1,2,1],[4,2,1]]
test_arr_1
test_arr_2 = [[2,3,1],[3,8,3],[1,5,0],[0,1,1],[1,1,1],[2,1,9],[0,2,2]]
test_arr_2

test_x_array = np.array(test_arr_1)
test_y_array = np.array(test_arr_2)

scratch_temp_cca = rcca.CCA(kernelcca = False, reg = 0., numCC = 1, verbose = False)
test_ccac =scratch_temp_cca.train([test_x_array, test_y_array])
test_ccac.cancorrs[0]

test_new_cca = new_cca(n_components=1)
test_new_cca.fit(test_x_array, test_y_array)
test_result_x_c, test_result_y_c = test_new_cca.transform(test_x_array, test_y_array)

np.corrcoef(test_result_x_c.T, test_result_y_c.T)[0,1]


import depmeas_master.python.rdc as rdc

rdc.cca(test_x_array,test_y_array)



test_result_x_c
test_result_y_c

scratch_A = test_arr_1[:3]
scratch_b = test_result_x_c[:3]
scratch_x = np.linalg.solve(scratch_A, scratch_b)

np.dot(np.array(test_arr_1),np.array(scratch_x))




X = test_x_array
Y = test_y_array
X
Y
n,p1 = X.shape
n,p1
n,p2 = Y.shape
n,p2

# center X and Y
meanX = X.mean(axis=0)
meanY = Y.mean(axis=0)


# X = X-meanX[np.newaxis,:]
# Y = Y-meanY[np.newaxis,:]



Qx,Rx = np.linalg.qr(X)
Qy,Ry = np.linalg.qr(Y)



rankX = np.linalg.matrix_rank(Rx)
rankX

if rankX == 0:
    raise Exception('Rank(X) = 0! Bad Data!')
elif rankX < p1:
    #warnings.warn("X not full rank!")
    Qx = Qx[:,0:rankX]
    Rx = Rx[0:rankX,0:rankX]

rankY = np.linalg.matrix_rank(Ry)
if rankY == 0:
    raise Exception('Rank(X) = 0! Bad Data!')
elif rankY < p2:
    #warnings.warn("Y not full rank!")
    Qy = Qy[:,0:rankY]
    Ry = Ry[0:rankY,0:rankY]

d = min(rankX,rankY)
svdInput = np.dot(Qx.T,Qy)
svdInput


U,r,V = np.linalg.svd(svdInput)
r = np.clip(r,0,1)
# A = np.linalg.lstsq(Rx, U[:,0:d], rcond=None) * np.sqrt(n-1)
# B = np.linalg.lstsq(Ry, V[:,0:d], rcond=None) * np.sqrt(n-1)
# print(A)
# print(B)
# TODO: resize A to match inputs

# return (A,B,r)
return r, U, V



import csv

sales1_arr = []
sales2_arr = []
with open("Data/sales1.csv", newline='') as csvfile:
    sales1_csv = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in sales1_csv:
        sales1_arr.append(row)

with open("Data/sales2.csv", newline='') as csvfile:
    sales2_csv = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in sales2_csv:
        sales2_arr.append(row)



sales1_arr = [[float(sales1_arr[i][j]) for j in range(len(sales1_arr[0]))] for i in range(len(sales1_arr)) ]
sales2_arr = [[float(sales2_arr[i][j]) for j in range(len(sales2_arr[0]))] for i in range(len(sales2_arr)) ]


sales1_arr_array = np.array(sales1_arr)
sales2_arr_array = np.array(sales2_arr)



sales_temp_cca = rcca.CCA(kernelcca = False, reg = 0., numCC = 1, verbose = False)
sales_ccac = sales_temp_cca.train([sales1_arr_array, sales2_arr_array])
sales_ccac.cancorrs[0]

sales_new_cca = new_cca(n_components=1)
sales_new_cca.fit(sales1_arr_array, sales2_arr_array)
sales_result_x_c, test_result_y_c = sales_new_cca.transform(sales1_arr_array, sales2_arr_array)
np.corrcoef(sales_result_x_c.T, test_result_y_c.T)[0,1]





import depmeas_master.python.rdc as rdc
coeffs, _, _, a, b = rdc.cca(sales1_arr_array, sales2_arr_array)
coeffs
a

b
sales_cc,sales_a,sales_b = sac.cca(sales1_arr_array,sales2_arr_array)
sales_cc
sales_a
sales_b




# Playing with the equation from https://www.cs.cmu.edu/~tom/10701_sp11/slides/CCA_tutorial.pdf

test_arr_x = [[4,7,9],[2,5,2],[3,3,7],[4,8,3],[1,1,1],[1,2,1],[4,2,1]]
test_arr_y = [[2,3,1],[3,8,3],[1,5,0],[0,1,1],[1,1,1],[2,1,9],[0,2,2]]

test_x_array = np.array(test_arr_x)
test_y_array = np.array(test_arr_y)

# np.savetxt("my_array_x.csv", test_x_array, delimiter=",")
# np.savetxt("my_array_y.csv", test_y_array, delimiter=",")

mean_N_readings_x = np.mean(test_x_array[:,0])
mean_E_readings_x = np.mean(test_x_array[:,1])
mean_Z_readings_x = np.mean(test_x_array[:,2])

mean_N_readings_y = np.mean(test_y_array[:,0])
mean_E_readings_y = np.mean(test_y_array[:,1])
mean_Z_readings_y = np.mean(test_y_array[:,2])

new_test_x_array = np.array([[test_x_array[i,0]-mean_N_readings_x, test_x_array[i,1]-mean_E_readings_x, test_x_array[i,2]-mean_Z_readings_x] for i in range(len(test_x_array))])
new_test_y_array = np.array([[test_y_array[i,0]-mean_N_readings_y, test_y_array[i,1]-mean_E_readings_y, test_y_array[i,2]-mean_Z_readings_y] for i in range(len(test_y_array))])

new_test_x_array
new_test_y_array



Cxx = np.cov(new_test_x_array.T)
Cyy = np.cov(new_test_y_array.T)
Cxy = np.array([[np.cov(new_test_x_array[:,i], new_test_y_array[:,j])[0,1] for j in range(3)] for i in range(3)])
Cyx = np.array([[Cxy[i][j] for i in range(3)] for j in range(3)])

new_test_x_array


new_test_x_array[:,0].T


list(new_test_x_array[:,1].T)

np.cov(list(new_test_x_array[:,0]),list(new_test_x_array[:,1]))

Cxx


scratch_prod_vector = [new_test_x_array[:,0][i] * new_test_x_array[:,0][i] for i in range(len(new_test_x_array[:,0]))]
scratch_prod_vector

np.sum(scratch_prod_vector)
covariance_x0x1 = (1/(len(new_test_x_array[:,0])-1)) * np.sum(scratch_prod_vector)

covariance_x0x1


list(new_test_x_array[:,0])

Cyy



np.cov(new_test_x_array[:,0], new_test_y_array[:,0])


C = np.dot(np.dot(np.dot(np.linalg.inv(Cxx), Cxy), np.linalg.inv(Cyy)), Cyx)
C_alt = np.linalg.inv(Cyy) * Cyx * np.linalg.inv(Cxx) * Cxy

evalues, evectors = np.linalg.eig(C)

evalues

evalues


evectors

evectors

# scratch = gen_data.generate_one_day_time_series('2001-04-03', '08:00:00', 128, 4, phase_shift = [0, np.random.rand(), np.random.rand()], station = ['XXX','YYY'])

scratch = gen_data.generate_one_day_time_series('2001-04-03', '05:00:00', 128, 4, lag = 24, station = ['XXX','YYY'])

scratch.measurements.plot.line(x='time', hue='component', col='station', col_wrap=1, figsize=(20,20))

scratch_window = scratch.measurements[324:428]

ts_1 = np.array(scratch_window.loc[dict(station = 'XXX')])

ts_2 = np.array(scratch_window.loc[dict(station = 'YYY')])

cc, a, b = sac.cca(ts_1,ts_2)
np.savetxt("ts_1_no_gauss_24_offset.csv", ts_1, delimiter=",")
np.savetxt("ts_2_no_gauss_24_offset.csv", ts_2, delimiter=",")

a

u = [ts_1[i][0]*a[0] + ts_1[i][1]*a[1] + ts_1[i][2]*a[2] for i in range(len(ts_1))]
v = [ts_2[i][0]*b[0] + ts_2[i][1]*b[1] + ts_2[i][2]*b[2] for i in range(len(ts_1))]

ts_1[2]

np.corrcoef(u,v)



x_blah = np.array([[100,2,3],[20,2,3],[3,10,1],[3,1,20],[1,1,1]])

y_blah = np.array([[3,200,3],[2,100,3],[20,1,1],[2,1,20],[1,1,300]])

cc_blah, a_blah, b_blah = sac.cca(x_blah,y_blah)

cc_blah
a_blah
b_blah

cc_blah
a_blah
b_blah

cc_blah
a_blah
b_blah

cc_rdc,  _, _,a_rdc, b_rdc = rdc.cca(x_blah,y_blah)
cc_rdc
a_rdc
b_rdc

cc_blah
a_blah
b_blah
