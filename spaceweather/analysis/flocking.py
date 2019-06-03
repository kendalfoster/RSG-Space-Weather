import numpy as np

def order_params(data, normed = False):
    #there might be some difference if i normed each reading before doing the calculations, as some stations are more
    #sensitive than others
    #need to look into dealing with missing readings - currently looks like
    #if readings missing from one station then everything is fucked. should be quite simple.

    n = len(data.station)
    params = np.zeros(len(data.time))
    i = 0

    for t in data.time:
        sum_N = np.nansum(data.measurements.loc[dict(component = "N", time = t)])
        sum_E = np.nansum(data.measurements.loc[dict(component = "E", time = t)])
        sum_Z = np.nansum(data.measurements.loc[dict(component = "Z", time = t)])
        normed_sum = np.sqrt(sum_N**2 + sum_E**2 + sum_Z**2)
        # np.sqrt(sum(sum(N330.measurements.loc[dict(station = s, time = t)] for s in N330.station)**2))

        v_0 = np.sum(np.sqrt(np.nansum(data.measurements.loc[dict(station = s, time = t)]**2)) for s in data.station)/n
        #average absolute velocity

        phi = (normed_sum/(n*v_0))#.data.item()
        #calculates actual order parameter
        params[i] = phi

        i += 1

    return params


def pcf(data, dr = 0.3):
    #HIGHLY INEFFICIENT - TAKES AGES TO RUN. PROCEED WITH CAUTION
    # dr = 0.3
    r_range = np.linspace(0, 2, 21)
    # t = data.time[5]
    N = len(data.station) #number of points
    results = np.zeros((len(r_range), len(data.time))) #pair correlation function

    for time_index in range(len(data.time)):
        t = data.time[time_index]
        for s in data.station:
            data.measurements.loc[dict(station = s, time = t)] = vector_normalise(data.measurements.loc[dict(station = s, time = t)].data)

        for r_index in range(len(r_range)):
            r = r_range[r_index]
            count = 0

            for s1 in data.station:
                for s2 in data.station:
                    diff = data.measurements.loc[dict(station = s1, time = t)] - data.measurements.loc[dict(station = s2, time = t)]
                    dist = np.sqrt(np.nansum(diff**2))
                    if (max(0, r) < dist and dist < r+dr):
                        count += 1

            results[r_index, time_index] = (r*count)/(3*dr*N**2)

    return results


def stackedplot(data, pcf, order):
    nplots = len(data.station)+ 2

    fig = plt.figure(figsize = (20, 30))

    pcm = plt.subplot(nplots, 1, 1)
    osp = plt.subplot(nplots, 1, 2) #order sub plot

    pcm.pcolormesh(pcf)
    # pcm.set_xlabel("time", fontsize = 20)
    # pcm.ylabel("r", fontsize = 20)
    # pcm.colorbar()
    osp.plot(order)

    for i in range(nplots-2):
        s = data_ms.station[i]
        ax = plt.subplot(nplots, 1, i+3, sharex = pcm)
        ax.plot(data_ms.measurements.loc[dict(station = s)])
        ax.title.set_text(s.data)
