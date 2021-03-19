import numpy as np
import numba

@numba.njit
def derive_voa(verts, voa_area_arr, voa_val_arr, voa_rate_arr, beta_weighted_distances):

    # betas are already to the negative exponent, therefore not dividing but multiplying
    # prepare variables for aggregating IDW components
    # essentially, aggregate the items weighted by distance, then divide by similarly aggregated distances

    dist_agg = 0
    voa_area_w_mean = 0
    voa_val_w_mean = 0
    voa_rate_w_mean = 0
    voa_val_w_cov = 0
    dist_val_cov = 0
    voa_rate_w_cov = 0
    dist_rate_cov = 0

    # process each vertex idx in the verts set - which is an array containing locally aggregated items
    # because the items are being gravity weighted, you can't simply mash the nested arrays together, use IDW instead
    for v_idx in verts:

        # use is_finite to check whether any valid entries exist
        # continue if not, otherwise nans will be returned to the aggregation
        n = np.sum(np.isfinite(voa_area_arr[v_idx]))
        if not n:
            continue

        # add the weighted distance for this item * the number of items
        # was: np.sum(np.isfinite(voa_area_arr[v_idx])) * beta_weighted_distances[v_idx]
        # which seems wrong, doesn't seem necessary to count instances first then multiply by betas...?
        # now (and no longer necessary to sum separately for each item?):
        dist_agg += beta_weighted_distances[v_idx]

        # weight each item by the local distance weighted beta, then sum to the aggregating variables above
        # remember, each index location represents an already aggregated array of values
        # these values were collected on the database to the closest adjacent roadnode
        # therefore sum the array at this index location, then weight by the beta weighted distance for this location
        # was: np.sum(voa_area_arr[v_idx][np.isfinite(voa_area_arr[v_idx])] * betas[v_idx])
        # which seems wrong as sum should be weighted rather than individual elements summed
        # now:
        voa_area_w_mean += np.nanmean(voa_area_arr[v_idx]) * beta_weighted_distances[v_idx]
        # similar to above:
        voa_val_w_mean += np.nanmean(voa_val_arr[v_idx]) * beta_weighted_distances[v_idx]
        voa_rate_w_mean += np.nanmean(voa_rate_arr[v_idx]) * beta_weighted_distances[v_idx]
        # per above issues, was:
        # np.sum(voa_val_arr[v_idx][np.isfinite(voa_val_arr[v_idx])] ** 2 * betas[v_idx])
        # items with only 1 element would theoretically add 0 (due to 0 stand dev) so skip
        val_mean = np.nanmean(voa_val_arr[v_idx])  # to catch division by zero
        if n > 1 and val_mean:
            voa_val_w_cov += np.nanstd(voa_val_arr[v_idx]) / val_mean * beta_weighted_distances[v_idx]
            dist_val_cov += beta_weighted_distances[v_idx]
        # repeat for rate
        rate_mean = np.nanmean(voa_rate_arr[v_idx])  # to catch division by zero
        if n > 1 and rate_mean:
            voa_rate_w_cov += np.nanstd(voa_rate_arr[v_idx]) / rate_mean * beta_weighted_distances[v_idx]
            dist_rate_cov += beta_weighted_distances[v_idx]

    # divide aggregated (weighted) values by aggregated (weighted) distances for weighted mean
    # catch division by zero
    if dist_agg:
        voa_area_w_mean = voa_area_w_mean / dist_agg
        voa_val_w_mean = voa_val_w_mean / dist_agg
        voa_rate_w_mean = voa_rate_w_mean / dist_agg
    else:
        voa_area_w_mean = np.nan
        voa_val_w_mean = np.nan
        voa_rate_w_mean = np.nan

    # check cov values separately because there is no guarantee that they exist
    if voa_val_w_cov and dist_val_cov:
        voa_val_w_cov = voa_val_w_cov / dist_val_cov
    else:
        voa_val_w_cov = np.nan

    if voa_rate_w_cov and dist_rate_cov:
        voa_rate_w_cov = voa_rate_w_cov / dist_rate_cov
    else:
        voa_rate_w_cov = np.nan

    return voa_area_w_mean, voa_val_w_mean, voa_val_w_cov, voa_rate_w_mean, voa_rate_w_cov