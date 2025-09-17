import numpy as np
import pandas as pd
import os
import copy
from scipy.ndimage import uniform_filter1d
from FLake.Flake_model import Flake


class IC:
    def __init__(self, ic_path, lake_ids, frac_pop=.75):
        """
        Lake-specific input constants (IC)
        """
        self.df = pd.read_csv(ic_path, index_col=0, header=0).loc[lake_ids].astype(np.float32)

        # Load spring nutrient
        nutrient_df = pd.read_csv('Data/SpringN-SummerChla.csv', header=0, index_col=0)
        for lake_id in lake_ids:
            self.df.loc[lake_id, 'Ini_TP'] = nutrient_df.loc[lake_id, 'Limit_P']
            self.df.loc[lake_id, 'Ini_POP'] = self.df.loc[lake_id, 'Ini_TP'] * frac_pop
            self.df.loc[lake_id, 'Ini_DIP'] = self.df.loc[lake_id, 'Ini_TP'] * (1 - frac_pop)
            self.df.loc[lake_id, 'Sediment_POP'] = self.df.loc[lake_id, 'Ini_TP'] * frac_pop
            self.df.loc[lake_id, 'Sediment_DIP'] = self.df.loc[lake_id, 'Ini_TP'] * (1 - frac_pop)


class SV:
    def __init__(self, nz, sv_names, lake_ids, date_range, icdf, sv_ini_dic):
        """
        Model state variables (SV)
        """
        # For all lake SV ini
        for sv_name in sv_names:
            for inz in range(nz):
                df = pd.DataFrame(index=date_range, columns=lake_ids, dtype=np.float32)
                if 'Ini_{}'.format(sv_name) in icdf.columns:
                    df.iloc[0, :] = icdf.loc[lake_ids, 'Ini_{}'.format(sv_name)].values
                elif sv_name in sv_ini_dic.keys():
                    df.iloc[0, :] = sv_ini_dic[sv_name]

                else:
                    IndexError('SV MUST ASSIGN INITIAL VALUES')
                setattr(self, '{}_{}'.format(sv_name, inz), df)


class ImV:
    def __init__(self, nz, imv_names, lake_ids, date_range):
        """
        For intermediate variables (ImV) during modeling
        """
        for imv_name in imv_names:
            for inz in range(nz):
                df = pd.DataFrame(index=date_range, columns=lake_ids)
                setattr(self, '{}_{}'.format(imv_name, inz), df)


class FlakeOutputs:
    def __init__(self, nz, load_dir, suffix, lake_ids, date_range):
        """
        Load 1D driver variables (from FLake outputs) for modeling
        """
        # load df
        self.lake_ids = lake_ids
        self.nz = nz
        self.dfs = {}
        self.date_range = date_range
        self.first_year = date_range.year.min()
        self.last_year = date_range.year.max()
        self.load_names = [
            'T_ws', 'T_wb', 'T_b1',
            'H_ml', 'H_b1', 'C_t',
            'I_ws',
        ]

        for name in self.load_names:
            df = pd.read_csv(os.path.join(load_dir, '{}{}.csv'.format(suffix, name)), header=0, index_col=0)
            df.index = pd.to_datetime(df.index)
            df = df.interpolate(limit_direction='both')
            self.dfs[name] = df.loc[date_range, lake_ids]

        # load calibrated model
        from FLake.Flake_optimize import FlakeOptimizer
        calibrated_paras_df = pd.read_csv('FlakeInput/Calibrated_Paras_Lake142.csv', index_col=0, header=0).loc[self.lake_ids]
        calibrated_paras_dic = {
            p_name: calibrated_paras_df[p_name].values
            for p_name in calibrated_paras_df.columns
        }
        self.flake_model = FlakeOptimizer.adjust_paras(Flake(lake_ids), use_paras_dic=calibrated_paras_dic)

        # calc profile_arr
        self.depths = self.flake_model.IC.df.loc[self.lake_ids, 'depth_w'].values
        self.box_boundary = np.linspace([0] * len(self.depths), self.depths, nz + 1)    # shape: (nz+1, n_lake)
        self.dz = self.box_boundary[1] - self.box_boundary[0]       # shape: (n_lake, )
        self.box_center = self.box_boundary[:-1] + self.dz * 0.5    # shape: (nz, n_lake)
        # self.box_center[0] = np.where(self.box_center[0] > 2, 2, self.box_center[0])
        self.box_upper = self.box_boundary[:-1]    # shape: (nz, n_lake)
        self.box_bottom = self.box_boundary[1:]    # shape: (nz, n_lake)
        self.opticpar_water_extincoef = self.flake_model.PC.opticpar_water_extincoef

    def calc_discrete_temp(self, date):
        """
        profile_arr: the box center z
        """
        # calc t
        h_ml = self.dfs['H_ml'].loc[date, self.lake_ids].values
        T_ws = self.dfs['T_ws'].loc[date, self.lake_ids].values
        T_wb = self.dfs['T_wb'].loc[date, self.lake_ids].values
        C_t = self.dfs['C_t'].loc[date, self.lake_ids].values

        depth_profile = self.box_boundary[:-1] + self.dz * 0.5
        temp_profile = np.zeros_like(self.box_center, dtype=np.float32)     # (nz, n_lake)
        for il, lake_id in enumerate(self.lake_ids):
            n_ml = np.sum(depth_profile[:, il] <= h_ml[il])
            temp_profile[:n_ml, il] = T_ws[il]
            t_depth = depth_profile[n_ml:, il]
            if len(t_depth) > 0:
                lmd_arr = (t_depth - h_ml[il]) / (self.depths[il] - h_ml[il])
                temp_profile[n_ml:, il] = T_ws[il] - (T_ws[il] - T_wb[il]) * self.shape_function(C_t[il], lmd_arr)
        return temp_profile

    def calc_discrete_radiation(self, date, chla=None, k_ext_chla=0.02):
        """
        Consider light attenuation of chla (in shape [nz, n_lake])
        k_ext_chla: Light attenuation coefficient for chlorophyll in unit: m2/mg (Arh.2005)
        """
        I_ws = self.dfs['I_ws'].loc[date, self.lake_ids].values
        I_surface = I_ws * self.flake_model.PC.opticpar_water_frac

        if chla is None:
            k_tot = np.full((self.nz, len(self.lake_ids)), self.opticpar_water_extincoef, dtype=np.float32)
        else:
            chla = np.maximum(chla, 0)
            k_tot = self.opticpar_water_extincoef + chla * k_ext_chla  # shape: (nz, n_lake)

        I_top = np.zeros((self.nz, len(self.lake_ids)), dtype=np.float32)
        I_bot = np.zeros((self.nz, len(self.lake_ids)), dtype=np.float32)
        I_avg = np.zeros((self.nz, len(self.lake_ids)), dtype=np.float32)
        I_top[0] = I_surface
        I_bot[0] = I_surface * np.exp(-k_tot[0] * self.dz)
        I_avg[0] = (I_top[0] - I_bot[0]) / (k_tot[0] * self.dz)

        for i in range(1, self.nz):
            I_top[i] = I_bot[i - 1]
            I_bot[i] = I_top[i] * np.exp(-k_tot[i] * self.dz)
            I_avg[i] = (I_top[i] - I_bot[i]) / (k_tot[i] * self.dz)
        radiation_profile = I_avg

        return radiation_profile

    def calc_discrete_kz(self, date, temp_profile_crt, temp_profile_lst=None, temp_profile_lst2=None, d2T_dz2_min=0.001, dT_dt_min=0.001):
        """
        Vertical diffusion coefficient        Calc kz use Heat Diffusion Equation
        """
        # (nz, n_lake)
        temp_profile_lst = copy.deepcopy(temp_profile_crt) if temp_profile_lst is None else temp_profile_lst
        temp_profile_lst2 = copy.deepcopy(temp_profile_lst) if temp_profile_lst2 is None else temp_profile_lst2
        temp_profile_fut = self.calc_discrete_temp(date + pd.Timedelta(days=1)) if date + pd.Timedelta(days=1) in self.date_range else copy.deepcopy(temp_profile_crt)
        temp_profile_fut2 = self.calc_discrete_temp(date + pd.Timedelta(days=2)) if date + pd.Timedelta(days=2) in self.date_range else copy.deepcopy(temp_profile_fut)

        # mean Filter
        T_arr = np.zeros(shape=(5, self.nz, len(self.lake_ids)), dtype=np.float32)
        T_arr[0] = temp_profile_lst2
        T_arr[1] = temp_profile_lst
        T_arr[2] = temp_profile_crt
        T_arr[3] = temp_profile_fut
        T_arr[4] = temp_profile_fut2
        T_smooth = uniform_filter1d(T_arr, size=3, axis=1)  # filter in depth
        T_smooth = uniform_filter1d(T_smooth, size=3, axis=0)  # filter in time
        temp_profile_lst = T_smooth[1]
        temp_profile_crt = T_smooth[2]
        temp_profile_fut = T_smooth[3]

        # calc kz
        lst_date = pd.to_datetime(date) + pd.Timedelta(days=-1) if date != self.date_range[0] and date != self.date_range[1] else date
        h_b1 = self.dfs['H_b1'].loc[date, self.lake_ids].values
        T_ws = self.dfs['T_ws'].loc[date, self.lake_ids].values
        T_ws_last = self.dfs['T_ws'].loc[lst_date, self.lake_ids].values
        T_wb = self.dfs['T_wb'].loc[date, self.lake_ids].values
        T_b1 = self.dfs['T_b1'].loc[date, self.lake_ids].values
        depth_profile = self.box_boundary[1:]
        dz = depth_profile[1] - depth_profile[0]
        temp_z_direct = np.where(T_wb <= T_ws, -1, 1)
        temp_t_direct = np.where(T_ws >= T_ws_last, 1, -1)

        dT_dt = (temp_profile_fut - temp_profile_lst) / 2.      # center dif.
        dT_dt = np.where(np.abs(dT_dt) < dT_dt_min, dT_dt_min * temp_t_direct, dT_dt)

        d2T_dz2 = (temp_profile_crt[2:] - 2 * temp_profile_crt[1:-1] + temp_profile_crt[:-2]) / (dz ** 2)       # (nz-2, n_lake)
        d2T_dz2_fill = np.zeros(shape=(self.nz, len(self.lake_ids)), dtype=np.float32)
        d2T_dz2_fill[1:-1] = d2T_dz2
        d2T_dz2_fill[0] = d2T_dz2[1]
        d2T_dz2_fill[-1] = d2T_dz2[-1]
        d2T_dz2 = copy.deepcopy(d2T_dz2_fill)
        d2T_dz2_min = d2T_dz2_min / (self.dz ** 2)
        d2T_dz2 = np.where(np.abs(d2T_dz2) < d2T_dz2_min, d2T_dz2_min * temp_z_direct, d2T_dz2)

        # SAFETY
        kz_min = (self.dz ** 0.3) * 0.01
        kz_max = (self.dz ** 1.0) * 0.5
        kz_profile = np.clip(np.abs(dT_dt / d2T_dz2), kz_min, kz_max)
        return kz_profile

    @staticmethod
    def shape_function(C, lmd_arr):
        """
        Curve function in Flake
        """
        theta = (40 / 3 * C - 20 / 3) * lmd_arr \
                 + (18 - 30 * C) * lmd_arr ** 2 \
                 + (20 * C - 12) * lmd_arr ** 3 \
                 + (5 / 3 - 10 / 3 * C) * lmd_arr ** 4
        return theta
