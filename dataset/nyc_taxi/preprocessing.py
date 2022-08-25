'''
Generate the whole sequence of nyc taxi data from 2017 to 2019
Output:
volume_valid_region_borough.npz: sparse matrix of [#timesteps, #regions+#boroughs, #feature_dim=2(pickup and dropoff)]. Load with sparse.load_npz(file)
edges_valid_region_borough.npz: npz of `edges`: [#edges, 2]; `edge_types`: [#edges,]; `node_types`: [#regions+#boroughs]
edge type: 0 - region to region; 1 - region and borough; 2 - borough to borough
node type: 0 - region; 1 - borough
'''
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import calendar
from argparse import ArgumentParser
import multiprocessing as mp
import json
import traceback
from numpy.lib.npyio import save

from tqdm import tqdm
import numpy as np
import pandas as pd
import swifter
import sparse


def process_df_raw(df_raw, T_delta, N, objid2rank_taxizones, datatype):
    if datatype == 'yellow':
        pickup_datetime_cname, dropoff_datetime_cname = 'tpep_pickup_datetime', 'tpep_dropoff_datetime'
    elif datatype == 'green':
        pickup_datetime_cname, dropoff_datetime_cname = 'lpep_pickup_datetime', 'lpep_dropoff_datetime'
    elif datatype == 'fhv':
        pickup_datetime_cname, dropoff_datetime_cname = 'pickup_datetime', 'dropoff_datetime'

    print('processing...')
    # volume data: [T_slots, N, 2]; flow data: [T_slots, N, N, 2]; 2 means pickup/dropoff
    first_time = datetime.strptime(
        df_raw.iloc[0][pickup_datetime_cname], '%Y-%m-%d %H:%M:%S')
    year, month = first_time.year, first_time.month
    first_day = datetime(year=year, month=month, day=1)
    month_days = calendar.monthrange(year, month)[1]
    T_slots = int(
        np.ceil(timedelta(days=month_days).total_seconds() / T_delta))
    volume_arr_pu_shape = (T_slots, N)
    volume_arr_do_shape = (T_slots, N)
    flow_arr_pu_shape = (T_slots, N, N)
    flow_arr_do_shape = (T_slots, N, N)

    df_raw_part = df_raw[[pickup_datetime_cname, dropoff_datetime_cname, 'PULocationID', 'DOLocationID']]
    df_raw_part.columns = ['pu_t', 'do_t', 'pu_loc', 'do_loc']
    # df_raw_part['pu_loc'] = df_raw_part['pu_loc'] - 1
    # df_raw_part['do_loc'] = df_raw_part['do_loc'] - 1
    df_raw_part['pu_loc'] = df_raw_part['pu_loc'].map(objid2rank_taxizones)
    df_raw_part['do_loc'] = df_raw_part['do_loc'].map(objid2rank_taxizones)
    df_raw_part['pu_t_slot'] = df_raw_part['pu_t'].swifter.apply(lambda x: int(np.floor(
        (datetime.strptime(x, '%Y-%m-%d %H:%M:%S') - first_day).total_seconds() / T_delta)))
    df_raw_part['do_t_slot'] = df_raw_part['do_t'].swifter.apply(lambda x: int(np.floor(
        (datetime.strptime(x, '%Y-%m-%d %H:%M:%S') - first_day).total_seconds() / T_delta)))
    df_raw_part = df_raw_part[['pu_t_slot',
                               'do_t_slot', 'pu_loc', 'do_loc']].dropna()

    df_raw_part_pu = df_raw_part[
        (df_raw_part['pu_t_slot'] >= 0) & (df_raw_part['pu_t_slot'] < T_slots)]
    volume_arr_pu_df = df_raw_part_pu.groupby(
        ['pu_t_slot', 'pu_loc']).count().reset_index()
    volume_arr_pu = sparse.COO(coords=[
        volume_arr_pu_df['pu_t_slot'].to_numpy(
        ), volume_arr_pu_df['pu_loc'].to_numpy()
    ], data=volume_arr_pu_df['do_t_slot'].to_numpy(), shape=volume_arr_pu_shape)
    flow_arr_pu_df = df_raw_part_pu.groupby(
        ['pu_t_slot', 'pu_loc', 'do_loc']).count().reset_index()
    flow_arr_pu = sparse.COO(coords=[
        flow_arr_pu_df['pu_t_slot'].to_numpy(), flow_arr_pu_df['pu_loc'].to_numpy(
        ), flow_arr_pu_df['do_loc'].to_numpy()
    ], data=flow_arr_pu_df['do_t_slot'].to_numpy(), shape=flow_arr_pu_shape)

    df_raw_part_do = df_raw_part[
        (df_raw_part['do_t_slot'] >= 0) & (df_raw_part['do_t_slot'] < T_slots)]
    volume_arr_do_df = df_raw_part_do.groupby(
        ['do_t_slot', 'do_loc']).count().reset_index()
    volume_arr_do = sparse.COO(coords=[
        volume_arr_do_df['do_t_slot'].to_numpy(
        ), volume_arr_do_df['do_loc'].to_numpy()
    ], data=volume_arr_do_df['pu_t_slot'].to_numpy(), shape=volume_arr_do_shape)
    flow_arr_do_df = df_raw_part_do.groupby(
        ['do_t_slot', 'pu_loc', 'do_loc']).count().reset_index()
    flow_arr_do = sparse.COO(coords=[
        flow_arr_do_df['do_t_slot'].to_numpy(), flow_arr_do_df['pu_loc'].to_numpy(
        ), flow_arr_do_df['do_loc'].to_numpy()
    ], data=flow_arr_do_df['pu_t_slot'].to_numpy(), shape=flow_arr_do_shape)

    volume_arr = sparse.stack([volume_arr_pu, volume_arr_do], axis=-1)
    flow_arr = sparse.stack([flow_arr_pu, flow_arr_do], axis=-1)
    return volume_arr, flow_arr


def process_and_save(raw_data_file_path, save_folder, T_delta, N, objid2rank_taxizones, datatype):
    try:
        if type(raw_data_file_path) is tuple:
            assert datatype == 'fhv'
            raw_data_file_path, raw_fhvhv_data_file_path = raw_data_file_path
            raw_data_file_path = Path(raw_data_file_path)
            raw_fhvhv_data_file_path = Path(raw_fhvhv_data_file_path)
            df_raw_fhv = pd.read_csv(raw_data_file_path)
            df_raw_fhv.rename(columns=lambda x: x.lower(), inplace=True)
            df_raw_fhvhv = pd.read_csv(raw_fhvhv_data_file_path)
            df_raw_fhvhv.rename(columns=lambda x: x.lower(), inplace=True)
            df_raw = pd.concat([
                df_raw_fhv[['pickup_datetime', 'dropoff_datetime', 'pulocationid', 'dolocationid']],
                df_raw_fhvhv[['pickup_datetime', 'dropoff_datetime', 'pulocationid', 'dolocationid']]
            ])
            df_raw.rename(columns={'pulocationid': 'PULocationID', 'dolocationid': 'DOLocationID'}, inplace=True)
            df_raw.dropna(inplace=True)
            # print(raw_data_file_path, df_raw.shape)
            # return
            df_raw = df_raw.astype({'pickup_datetime': str, 'dropoff_datetime': str, 'PULocationID': 'int64', 'DOLocationID': 'int64'}, copy=False)
        else:
            raw_data_file_path = Path(raw_data_file_path)
            print('reading...', raw_data_file_path)
            df_raw = pd.read_csv(raw_data_file_path)
            if datatype == 'fhv':
                df_raw.rename(columns=lambda x: x.lower(), inplace=True)
                df_raw = df_raw[['pickup_datetime', 'dropoff_datetime', 'pulocationid', 'dolocationid']]
                df_raw.rename(columns={'pulocationid': 'PULocationID', 'dolocationid': 'DOLocationID'}, inplace=True)
                df_raw.dropna(inplace=True)
                # print(raw_data_file_path, df_raw.shape)
                # return
                df_raw = df_raw.astype({'pickup_datetime': str, 'dropoff_datetime': str, 'PULocationID': 'int64', 'DOLocationID': 'int64'}, copy=False)
        volume_arr, flow_arr = process_df_raw(
            df_raw, T_delta, N, objid2rank_taxizones, datatype)
        save_folder_path = Path(save_folder)
        basename = raw_data_file_path.name.replace('.csv', '')
        sparse.save_npz(save_folder_path.joinpath(
            basename + '_volume.npz'), volume_arr)
        sparse.save_npz(save_folder_path.joinpath(
            basename + '_flow.npz'), flow_arr)
    except Exception as e:
        traceback.print_exc()
        print()
        raise e


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--raw_data_folder', required=True)
    parser.add_argument('--save_folder', required=True)
    parser.add_argument('--region_borough_mapping', default=None,
                        help='read lists of object ids of regions and boroughs from the file if given')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--type', type=str, default='yellow', help='yellow/green/fhv')
    parser.add_argument('--first_ym', type=str, default='2017-01')
    parser.add_argument('--last_ym', type=str, default='2019-12')
    args = parser.parse_args()

    raw_data_folder_path = Path(args.raw_data_folder)
    save_folder_path = Path(args.save_folder)
    monthly_save_folder_path = save_folder_path.joinpath('monthly')
    if not monthly_save_folder_path.exists():
        monthly_save_folder_path.mkdir(exist_ok=True, parents=True)
    num_workers = args.num_workers

    # years = [2017, 2018, 2019]
    # months = list(range(1, 13))
    year_months = []
    fy, fm = tuple(map(int, args.first_ym.split('-')))
    ly, lm = tuple(map(int, args.last_ym.split('-')))
    iy, im = fy, fm
    while (iy <= ly) and (im <= lm):
        year_months.append((iy, im))
        im += 1
        if im > 12:
            iy += 1
            im = 1
    print(year_months)

    # aggregation time window: 30min
    T_delta = timedelta(minutes=30).total_seconds()
    # total number of locations: 265 by default
    if args.region_borough_mapping is None:
        # remove [263, 264] regions with unknown borough [LocationID 264, 265]
        N = 263
        objid2rank_taxizones = {u: u - 1 for u in range(1, N + 1)}
    else:
        region_borough_mapping = pd.read_csv(args.region_borough_mapping)
        N = region_borough_mapping.shape[0]
        objid_taxizones = region_borough_mapping['OBJECTID_taxizones'].tolist()
        objid2rank_taxizones = {v: u for u, v in enumerate(objid_taxizones)}

    if not monthly_save_folder_path.joinpath('{}_tripdata_{:04d}-{:02d}_flow.npz'.format(args.type, year_months[0][0], year_months[0][1])).exists():
        raw_data_file_paths = []
        for year, month in year_months:
            raw_data_file_path = raw_data_folder_path.joinpath(
                '{}_tripdata_{:04d}-{:02d}.csv'.format(args.type, year, month))
            try:
                assert raw_data_file_path.exists()
            except AssertionError:
                raise AssertionError(str(raw_data_file_path))
            if args.type == 'fhv':
                fhvhv_raw_data_file_path  = raw_data_folder_path.joinpath('fhvhv_tripdata_{:04d}-{:02d}.csv'.format(year, month))
                if fhvhv_raw_data_file_path.exists():
                    raw_data_file_paths.append((raw_data_file_path, fhvhv_raw_data_file_path))
                else:
                    raw_data_file_paths.append(raw_data_file_path)
            else:
                raw_data_file_paths.append(raw_data_file_path)

        p = mp.Pool(num_workers)
        for raw_data_file_path in raw_data_file_paths:
            p.apply_async(
                process_and_save, args=(
                    raw_data_file_path, monthly_save_folder_path, T_delta, N, objid2rank_taxizones, args.type)
            )
        p.close()
        p.join()
        # for raw_data_file_path in raw_data_file_paths:
        #     process_and_save(raw_data_file_path, monthly_save_folder_path, T_delta, N, objid2rank_taxizones, args.type)

    if not save_folder_path.joinpath('volume.npz').exists():
        volume_arrs = []
        flow_arrs = []
        for year, month in tqdm(year_months):
            volume_arrs.append(sparse.load_npz(monthly_save_folder_path.joinpath(
                '{}_tripdata_{:04d}-{:02d}_volume.npz'.format(args.type, year, month))))
            flow_arrs.append(sparse.load_npz(monthly_save_folder_path.joinpath(
                '{}_tripdata_{:04d}-{:02d}_flow.npz'.format(args.type, year, month))))
        volume_arr = sparse.concatenate(volume_arrs, axis=0)
        flow_arr = sparse.concatenate(flow_arrs, axis=0)
        sparse.save_npz(save_folder_path.joinpath('volume.npz'), volume_arr)
        sparse.save_npz(save_folder_path.joinpath('flow.npz'), flow_arr)
    else:
        volume_arr = sparse.load_npz(save_folder_path.joinpath('volume.npz'))
        flow_arr = sparse.load_npz(save_folder_path.joinpath('flow.npz'))

    flow_sum_by_t_path = save_folder_path.joinpath('flow_sum_by_t.npy')
    if not flow_sum_by_t_path.exists():
        flow_sum_by_t = flow_arr.sum(axis=0).todense()
        np.save(flow_sum_by_t_path, flow_sum_by_t)
    else:
        flow_sum_by_t = np.load(flow_sum_by_t_path)

    if not save_folder_path.joinpath('volume_valid_region_borough.npz').exists():
        # Graph Construction
        # select region pairs with at least 1 trip on average in 6 hours as edges
        region_edges = np.where(flow_sum_by_t.sum(
            axis=-1) / flow_arr.shape[0] * 12 >= 1)
        region_edges = np.array(list(zip(*region_edges)), dtype=np.int64)
        print('Number of edges among regions: {}'.format(len(region_edges)))

        if args.region_borough_mapping is None:
            taxi_zone_lookup = pd.read_csv(
                'dataset/nyc_taxi/taxi+_zone_lookup.csv')
            # remove [263, 264] regions with unknown borough [LocationID 264, 265]
            taxi_zone_lookup = taxi_zone_lookup[taxi_zone_lookup['Borough'] != 'Unknown']
        else:
            taxi_zone_lookup = region_borough_mapping
        boroughs = sorted(taxi_zone_lookup['Borough'].unique())
        bid2boroughs = {u + N: v for u,
                        v in enumerate(boroughs)}
        boroughs2bid = {v: u + N
                        for u, v in enumerate(boroughs)}
        # with open('dataset/nyc_taxi/bid2boroughs.json', 'w') as f:
        #     json.dump(bid2boroughs, f)

        valid_region_boroughs = taxi_zone_lookup['Borough'].apply(
            lambda x: boroughs2bid[x]).tolist()
        region_borough_edges = []
        for u, v in enumerate(valid_region_boroughs):
            region_borough_edges.append((u, v))
            region_borough_edges.append((v, u))
        # region-borough: bidirectional edges
        region_borough_edges = np.array(region_borough_edges, dtype=np.int64)

        borough_borough_edges = []
        borough_ids = sorted(list(bid2boroughs.keys()))
        for bi in borough_ids:
            for bj in borough_ids:
                if bi != bj:
                    borough_borough_edges.append((bi, bj))
        # fully connected borough borough edges, no self-edge
        borough_borough_edges = np.array(borough_borough_edges, dtype=np.int64)

        all_edges = np.concatenate(
            (region_edges, region_borough_edges, borough_borough_edges), axis=0)
        all_edge_types = np.concatenate(
            (np.repeat(0, len(region_edges)), np.repeat(
                1, len(region_borough_edges)), np.repeat(2, len(borough_borough_edges)))
        )  # 0: region-region; 1: region-borough; 2: borough-borough
        all_node_types = np.concatenate(
            (np.repeat(0, N), np.repeat(1, len(borough_ids)))
        )  # 0: region, 1: borough
        print('region: {}; borough: {}'.format(
            N, len(borough_ids)))
        print('region-region: {}; region-borough: {}; borough-borough: {}'.format(
            len(region_edges), len(region_borough_edges), len(borough_borough_edges)))
        np.savez(save_folder_path.joinpath('edges_valid_region_borough.npz'),
                 edges=all_edges, edge_types=all_edge_types, node_types=all_node_types)

        # aggregate borough volume data by summation
        valid_volume_arr = volume_arr
        borough_volumes = []
        print('Aggregate borough volume by summation...')
        for borough_id in tqdm(borough_ids):
            subregion_ids = taxi_zone_lookup[taxi_zone_lookup['Borough']
                                             == bid2boroughs[borough_id]].index.tolist()
            borough_volume = valid_volume_arr[:, subregion_ids, :].sum(
                axis=1, keepdims=True)
            borough_volumes.append(borough_volume)
        borough_volumes = sparse.concatenate(borough_volumes, axis=1)
        valid_volume_arr_with_borough = sparse.concatenate(
            (valid_volume_arr, borough_volumes), axis=1)
        sparse.save_npz(save_folder_path.joinpath(
            'volume_valid_region_borough.npz'), valid_volume_arr_with_borough)
    else:
        valid_volume_arr_with_borough = sparse.load_npz(
            save_folder_path.joinpath('volume_valid_region_borough.npz'))
        edges_valid_region_borough = np.load(
            save_folder_path.joinpath('edges_valid_region_borough.npz'))
        all_edges, all_edge_types, all_node_types = edges_valid_region_borough[
            'edges'], edges_valid_region_borough['edge_types'], edges_valid_region_borough['node_types']
        temporal_info = {
            'start': datetime(year=fy, month=fm, day=1),
            'delta': timedelta(minutes=30)
        }
        print(temporal_info)
        np.savez(save_folder_path.joinpath('temporal_info.npz'), **temporal_info)
