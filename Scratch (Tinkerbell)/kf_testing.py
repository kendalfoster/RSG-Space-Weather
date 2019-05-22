scrap way we do adj_mat
for each station pair,
    for each time window,
        get the lag which produces the maximum correlation (like from correlogram stuff)
        store all of this in a 4-dimensional Dataset
        run threshold (like avg over time series) over this to get new-style adj_mat
return the adj_mat with dimensions:
                                   - first_st
                                   - second_st
                                   - time_win (index/win_start_time, ideally win_start)
