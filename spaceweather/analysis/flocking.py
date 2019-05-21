def order_param(data, t, normed = False):
    #there might be some difference if i normed each reading before doing the calculations, as some stations are more
    #sensitive than others
    #need to look into dealing with missing readings - currently looks like
    #if readings missing from one station then everything is fucked. should be quite simple.

    n = len(data.station)

    sum_N = sum(N330.measurements.loc[dict(component = "N", time = t)])
    sum_E = sum(N330.measurements.loc[dict(component = "E", time = t)])
    sum_Z = sum(N330.measurements.loc[dict(component = "Z", time = t)])
    normed_sum = np.sqrt(sum_N**2 + sum_E**2 + sum_Z**2)
    # np.sqrt(sum(sum(N330.measurements.loc[dict(station = s, time = t)] for s in N330.station)**2))

    v_0 = sum(np.sqrt(sum(N330.measurements.loc[dict(station = s, component = c, time = t)]**2 for c in N330.component)) for s in N330.station)/n
    #average absolute velocity

    phi = (normed_sum/(n*v_0)).data.item()
    #calculates actual order parameter

    return phi

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
