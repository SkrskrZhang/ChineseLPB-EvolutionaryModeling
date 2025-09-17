import os
import numpy as np
import pandas as pd
import copy
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from model_vars import FlakeOutputs, SV, IC, ImV
from model_paras import Paras
from scaling_relations import ScalingLaws
from Utils.ivp_solver import DoPri45Step, RK2Step
from Utils.lake_data_process import lake_ids


class LakePhytoModel1D:
    def __init__(self,
                 lake_ids,
                 date_range,
                 nz=5,      # unit: m
                 dt=5e-2,   # unit: day
                 ic_path='Data/IC142.csv',
                 flake_results_dir='FlakeInput/SV142_Calibrated',
                 scenario='ssp585',
                 ):
        self.nz = nz
        self.dt = dt
        self.lake_ids = lake_ids
        self.date_range = date_range
        self.future_scenario = scenario

        # --------------------------Load Paras and Input Const and Scaling relations---------------------------
        self.PC = Paras()
        self.IC = IC(ic_path, lake_ids)
        self.SL = ScalingLaws(self.PC)

        # --------------------------Load Variables---------------------------
        self.sv_const_names = [
            # Nutrient
            'POP', 'DIP', 'TP',      # µg P / L or mg P / m3
        ]
        self.sv_vars_names = [
            'P_{}',     # conc. of phytoplankton mm3 / m3 (or ppb) < if water> mg / m3
            # 't1_{}', 't2_{}',     # trait 1: cell volume (log µm^3); trait 2: opti. growth temp. (oC)
            # 'V1_{}', 'V2_{}', 'C12_{}',
            'Pt1_{}', 'Pt2_{}',
            'PV1t1_{}', 'PV2t2_{}', 'PC12t1t2_{}',
        ]
        self.sv_names = copy.deepcopy(self.sv_const_names)
        for p_n in self.PC.p_names:
            self.sv_names.extend([s.format(p_n) for s in self.sv_vars_names])
        self.SV = SV(nz, self.sv_names, lake_ids, date_range[:1], self.IC.df, self.PC.phyto_ini_dic)
        self.phyto_names = ['P_{}', 't1_{}', 't2_{}', 'V1_{}', 'V2_{}', 'C12_{}']
        self.imv_names = []
        self.ImV = ImV(nz, self.imv_names, lake_ids, date_range)

        # --------------------------Load Flake Outputs and v_trans---------------------------
        self.flake_results_dir = flake_results_dir
        self.FO = FlakeOutputs(nz, flake_results_dir, '{}-'.format(self.future_scenario), lake_ids, date_range)
        self.dz = self.FO.dz    # dz for each lake in shape (n_lake, )
        self.d_sediment = (self.IC.df.loc[self.lake_ids, 'Lake_Sediment_Depth'].values + self.dz) * 0.5

        # --------------------------For continuous run---------------------------
        self.dv_dic = {}
        self.chla_dic = {}
        self.log_bar = None
        self.y_output = None

    def continuous_odes(self, t, y):
        """
        RHS of the ODEs
        t: time in day
        y: vars in ODEs, flatten from (n_sv, nz, n_lake)
        NOTE: dt in dy/dt now is self.dt, not unit time
        """

        # --------------------------Reshape last y_output and re-calc y---------------------------
        crt_date = self.date_range[int(t)]


        y_shape = y.reshape(len(self.sv_names), self.nz, len(self.lake_ids))
        y_output = np.zeros_like(y_shape, dtype=np.float32)
        for i_sv, sv_name in enumerate(self.sv_const_names):
            locals()[sv_name] = y_shape[i_sv, :, :]         # shape: (nz, n_lake)
            locals()[sv_name] = np.clip(locals()[sv_name], 1, None)
        for p_n in self.PC.p_names:
            for sv_base_name in self.sv_vars_names:
                sv_name = sv_base_name.format(p_n)
                i_sv = self.sv_names.index(sv_name)
                if sv_base_name in ['P_{}']:
                    temp = y_shape[i_sv, :, :]
                    locals()[sv_name] = np.clip(y_shape[i_sv, :, :], 1, None)
                elif sv_base_name in ['Pt1_{}', 'Pt2_{}']:
                    locals()[sv_name] = np.clip(y_shape[i_sv, :, :], 0, None)
                else:
                    locals()[sv_name] = y_shape[i_sv, :, :]     # shape: (nz, n_lake)


        # Back-calc t1, t2, V1, V2, C12
        for p_n in self.PC.p_names:
            locals()['t1_{}'.format(p_n)] = locals()['Pt1_{}'.format(p_n)] / locals()['P_{}'.format(p_n)]       # # shape: (nz, n_lake)
            locals()['t2_{}'.format(p_n)] = locals()['Pt2_{}'.format(p_n)] / locals()['P_{}'.format(p_n)]
            # SAFETY
            t1_min, t1_max = self.PC.cell_volume_boundary_dic[p_n]
            t2_min, t2_max = self.PC.temp_opt_boundary_dic[p_n]
            locals()['t1_{}'.format(p_n)] = np.clip(locals()['t1_{}'.format(p_n)], t1_min, t1_max)
            locals()['t2_{}'.format(p_n)] = np.clip(locals()['t2_{}'.format(p_n)], t2_min, t2_max)
            locals()['V1_{}'.format(p_n)] = locals()['PV1t1_{}'.format(p_n)] / locals()['P_{}'.format(p_n)] - locals()['t1_{}'.format(p_n)] ** 2
            locals()['V2_{}'.format(p_n)] = locals()['PV2t2_{}'.format(p_n)] / locals()['P_{}'.format(p_n)] - locals()['t2_{}'.format(p_n)] ** 2
            locals()['C12_{}'.format(p_n)] = locals()['PC12t1t2_{}'.format(p_n)] / locals()['P_{}'.format(p_n)] - locals()['t1_{}'.format(p_n)] * locals()['t2_{}'.format(p_n)]
            locals()['V1_{}'.format(p_n)] = np.clip(locals()['V1_{}'.format(p_n)], 1e-3, 9)
            locals()['V2_{}'.format(p_n)] = np.clip(locals()['V2_{}'.format(p_n)], 1e-3, 9)
            locals()['C12_{}'.format(p_n)] = np.clip(locals()['C12_{}'.format(p_n)], -5, 5)


        # Calc Chl-a, which can influence vertical radiance profile
        n_day_radiation_acclimation = 3
        if len(self.dv_dic) >= n_day_radiation_acclimation:
            # Chla : C ratio is variable, depend on past n day radiation acclimation
            n_day_radiation_arr = np.zeros(shape=(n_day_radiation_acclimation, self.nz, len(self.lake_ids)), dtype=np.float32)
            for i_day_radiation_acclimation in range(1, n_day_radiation_acclimation+1):
                _, radiation_profile, _ = self.dv_dic[list(self.dv_dic.keys())[-i_day_radiation_acclimation]]
                n_day_radiation_arr[i_day_radiation_acclimation-1] = radiation_profile
            n_day_radiation_profile_mean = n_day_radiation_arr.mean(axis=0) * 2.1   # W/m2 to µmol quanta/m2/s
            Chla2C, pred_interval = self.SL.irradiance_2_chla_c_ratio(n_day_radiation_profile_mean)
            Chla2C = np.clip(Chla2C, np.log10(0.005), np.log10(0.03))
        else:
            Chla2C = np.log10(0.01)     # initial ratio of Chla:C, do not influence long-term modeling
        Chla_sum = np.zeros(shape=(self.nz, len(self.lake_ids)), dtype=np.float32)
        for p_n in self.PC.p_names:
            C_per_volume = np.power(10, self.SL.cell_volume_2_cell_mass_C(locals()['t1_{}'.format(p_n)], per_x=True))  # Carbon in phyto. [1e-12 g C / µm3] or [mg C / mm3]
            C_conc = locals()['P_{}'.format(p_n)] * C_per_volume                    # Phyto. Carbon conc. in [mg C/mm3] * [mm3/m3] = mg C/m3 or μg/L
            locals()['Chla_{}'.format(p_n)] = C_conc * np.power(10, Chla2C)         # Unit: mg Chla / m3
            Chla_sum += locals()['Chla_{}'.format(p_n)]


        # --------------------------Load driver vars---------------------------
        # temp (K), radiation (W/m2), kz (m2/d) profile, in shape: (nz, n_lake)
        temp_profile, radiation_profile, kz_profile = self.continuous_dv(t, Chla_sum)
        # Convert unit of K to C
        t_p_oC = np.clip(temp_profile - 273.15, 0, None)
        # W/m2 to µmol quanta/m2/s, SR W/m2 = 4.6 µmol quanta/m2/s, PAR = 0.45 SR, so factor is 4.6 * 0.45 = 2.1
        r_p_ppfd = radiation_profile * 2.1
        # kz (nz, n_lake), 0:nz-1 is the kz between nz boxes, the last is the water-bottom interf. kz
        kz_p = kz_profile


        # --------------------------Calc scaling relations---------------------------
        for p_n in self.PC.p_names:
            t1 = locals()['t1_{}'.format(p_n)]
            # log cell volume [µm3] -> log KP [µg P L-1 or mg/m3] -> exp(10,)
            locals()['KP_{}'.format(p_n)] = np.power(10, self.SL.cell_volume_2_KP(t1))
            # log cell volume [µm3] -> log QPmin [1e-12 g P / µm3] or [mg P / mm3] -> exp(10,)
            locals()['QPmin_lst_{}'.format(p_n)] = np.power(10, self.SL.cell_volume_2_QPmin(t1, per_x=True))
            # log cell volume [µm3] -> log Sink [m/d] -> exp(10,)
            locals()['Sink_{}'.format(p_n)] = np.power(10, self.SL.cell_volume_2_sink_rate(t1))


        # ---------------------------Phyto. growth and trait derivatives---------------------------
        for p_n in self.PC.p_names:
            t1, t2, P = locals()['t1_{}'.format(p_n)], locals()['t2_{}'.format(p_n)], locals()['P_{}'.format(p_n)]
            V1, V2, C12 = locals()['V1_{}'.format(p_n)], locals()['V2_{}'.format(p_n)], locals()['C12_{}'.format(p_n)]
            LGF = self.IC.df.loc[self.lake_ids, 'Lake_Grow_Factor'].values      # for lake-specific calibration

            # Calc FN, boundary [0, f(t1)]
            alpha_FN = self.PC.scaling_slope_dic['cell_volume_2_uptake_max_P'] - self.PC.scaling_slope_dic['cell_volume_2_QPmin'] * self.PC.scaling_slope_dic['cell_volume_2_cell_mass_C']
            locals()['FN_{}'.format(p_n)] = locals()['DIP'] / (locals()['DIP'] + locals()['KP_{}'.format(p_n)]) * np.power(10, alpha_FN * t1)

            # Calc FI, boundary [0, 1]
            a_ini, opt_I = self.PC.light_ini_slope_dic[p_n], self.PC.light_opt_I_dic[p_n]
            locals()['FI_{}'.format(p_n)] = a_ini * r_p_ppfd / ((r_p_ppfd / opt_I - 1) ** 2 + a_ini * r_p_ppfd)

            # Calc FT, boundary [0, 1]
            a_gm_Q10, T0, delatT = self.PC.grow_max_Q10_dic[p_n], self.PC.T0, self.PC.deltaT
            locals()['FT_{}'.format(p_n)] = np.clip(np.exp(a_gm_Q10 * (t2 - T0)) * np.exp((t_p_oC - t2) / delatT) * (t2 + delatT - t_p_oC) / delatT, 0, None)

            # Calc growth rate (U) at t1t2
            u_t0 = self.PC.growth_rate_t0_dic[p_n]
            # print('u0', p_n, u_t0)
            locals()['U_{}'.format(p_n)] = u_t0 * LGF * locals()['FT_{}'.format(p_n)] * locals()['FN_{}'.format(p_n)] * locals()['FI_{}'.format(p_n)]

            # Calc loss of morality (LM)
            M0, alpha_m = self.PC.morality_rate_dic[p_n], self.PC.alpha_m
            locals()['LM_{}'.format(p_n)] = M0 * np.exp(alpha_m * (t_p_oC - T0)) * P

            # Calc P sink rate
            locals()['k_sink_P_{}'.format(p_n)] = locals()['Sink_{}'.format(p_n)] * np.exp(0.5 * V1 * ((self.PC.scaling_slope_dic['cell_volume_2_sink_rate'] * self.PC.ln10)**2))      # m/d

            # Calc loss of Graze (LG)
            G0, alpha_m = self.PC.graze_rate_dic[p_n], self.PC.alpha_m
            locals()['LG_{}'.format(p_n)] = G0 * np.exp(alpha_m * (t_p_oC - T0)) * P

            # Calc Net growth rate (A) at t1t2
            locals()['A_{}'.format(p_n)] = locals()['U_{}'.format(p_n)] - (locals()['LM_{}'.format(p_n)] + locals()['LG_{}'.format(p_n)]) / P

            # 1st and 2nd derivatives of FN to t1 (cell volume)
            alpha_KP = self.PC.scaling_slope_dic['cell_volume_2_KP']
            temp = alpha_KP * locals()['KP_{}'.format(p_n)] / (locals()['KP_{}'.format(p_n)] + locals()['DIP'])
            locals()['dFN_dt1_{}'.format(p_n)] = locals()['FN_{}'.format(p_n)] * self.PC.ln10 * (alpha_FN - temp)
            locals()['d2FN_d2t1_{}'.format(p_n)] = self.PC.ln10 * (locals()['dFN_dt1_{}'.format(p_n)] * (alpha_FN - temp) + (temp - alpha_KP) * temp * self.PC.ln10 * locals()['FN_{}'.format(p_n)])

            # 1st and 2nd derivatives of FT to t2 (opti. T)
            temp = a_gm_Q10 - 1 / delatT + 1 / (t2 + delatT - t_p_oC)
            locals()['dFT_dt2_{}'.format(p_n)] = np.where((t2 + delatT - t_p_oC) > 0, locals()['FT_{}'.format(p_n)] * temp, 0)
            locals()['d2FT_d2t2_{}'.format(p_n)] = np.where(
                (t2 + delatT - t_p_oC) > 0,
                locals()['dFT_dt2_{}'.format(p_n)] * temp + locals()['FT_{}'.format(p_n)] * (-1 / (t2 + delatT - t_p_oC) ** 2), 0)

            # 1st and 2nd derivatives of A (same to U) to t1
            temp = u_t0 * LGF * locals()['FI_{}'.format(p_n)]
            locals()['dA_dt1_{}'.format(p_n)] = temp * locals()['FT_{}'.format(p_n)] * locals()['dFN_dt1_{}'.format(p_n)]
            locals()['d2U_d2t1_{}'.format(p_n)] = temp * locals()['FT_{}'.format(p_n)] * locals()['d2FN_d2t1_{}'.format(p_n)]
            locals()['d2A_d2t1_{}'.format(p_n)] = locals()['d2U_d2t1_{}'.format(p_n)]

            # 1st and 2nd derivatives of A (same to U) to t2
            locals()['dA_dt2_{}'.format(p_n)] = temp * locals()['FN_{}'.format(p_n)] * locals()['dFT_dt2_{}'.format(p_n)]
            locals()['d2A_d2t2_{}'.format(p_n)] = temp * locals()['FN_{}'.format(p_n)] * locals()['d2FT_d2t2_{}'.format(p_n)]
            locals()['d2U_d2t2_{}'.format(p_n)] = locals()['d2A_d2t2_{}'.format(p_n)]

            # 2nd derivatives of A (same to U) to t1t2
            locals()['d2A_dt1t2_{}'.format(p_n)] = temp * locals()['dFN_dt1_{}'.format(p_n)] * locals()['dFT_dt2_{}'.format(p_n)]
            locals()['d2U_dt1t2_{}'.format(p_n)] = locals()['d2A_dt1t2_{}'.format(p_n)]

            # Growth rate at group average
            A, U = locals()['A_{}'.format(p_n)], locals()['U_{}'.format(p_n)]
            dA_dt1, d2A_d2t1 = locals()['dA_dt1_{}'.format(p_n)], locals()['d2A_d2t1_{}'.format(p_n)]
            dA_dt2, d2A_d2t2 = locals()['dA_dt2_{}'.format(p_n)], locals()['d2A_d2t2_{}'.format(p_n)]
            d2A_dt1t2 = locals()['d2A_dt1t2_{}'.format(p_n)]
            locals()['Unet_{}'.format(p_n)] = np.clip(U + 0.5 * V1 * d2A_d2t1 + 0.5 * V2 * d2A_d2t2 + C12 * d2A_dt1t2, 0, None)


        # ---------------------------Check to avoid nutrient to negative due to multi-group phyto. uptake---------------------------
        min_dip = 1
        frac_max_uptake = np.clip((locals()['DIP'] - min_dip) / locals()['DIP'], 0., 1.)
        P_uptake_sum = np.zeros(shape=(self.nz, len(self.lake_ids)), dtype=np.float32)
        for p_n in self.PC.p_names:
            P_uptake_sum += locals()['Unet_{}'.format(p_n)] * locals()['P_{}'.format(p_n)] * locals()['QPmin_lst_{}'.format(p_n)]
        frac_U_max = np.where(P_uptake_sum < (locals()['DIP'] * frac_max_uptake), 1, locals()['DIP'] * frac_max_uptake / P_uptake_sum)
        for p_n in self.PC.p_names:
            locals()['Unet_{}'.format(p_n)] = locals()['Unet_{}'.format(p_n)] * frac_U_max
            locals()['U_{}'.format(p_n)] = locals()['U_{}'.format(p_n)] * frac_U_max
            locals()['A_{}'.format(p_n)] = locals()['U_{}'.format(p_n)] - (locals()['LM_{}'.format(p_n)] + locals()['LG_{}'.format(p_n)]) / locals()['P_{}'.format(p_n)]
            locals()['d2A_d2t1_{}'.format(p_n)] = locals()['d2A_d2t1_{}'.format(p_n)] * frac_U_max
            locals()['d2A_d2t2_{}'.format(p_n)] = locals()['d2A_d2t2_{}'.format(p_n)] * frac_U_max
            locals()['d2A_dt1t2_{}'.format(p_n)] = locals()['d2A_dt1t2_{}'.format(p_n)] * frac_U_max
            locals()['d2U_d2t1_{}'.format(p_n)] = locals()['d2U_d2t1_{}'.format(p_n)] * frac_U_max
            locals()['d2U_d2t2_{}'.format(p_n)] = locals()['d2U_d2t2_{}'.format(p_n)] * frac_U_max
            locals()['d2U_dt1t2_{}'.format(p_n)] = locals()['d2U_dt1t2_{}'.format(p_n)] * frac_U_max


        # ---------------------------Update phyto. vars---------------------------
        for p_n in self.PC.p_names:
            t1, t2, P = locals()['t1_{}'.format(p_n)], locals()['t2_{}'.format(p_n)], locals()['P_{}'.format(p_n)]
            V1, V2, C12 = locals()['V1_{}'.format(p_n)], locals()['V2_{}'.format(p_n)], locals()['C12_{}'.format(p_n)]
            A, U = locals()['A_{}'.format(p_n)], locals()['U_{}'.format(p_n)]
            dA_dt1, d2A_d2t1 = locals()['dA_dt1_{}'.format(p_n)], locals()['d2A_d2t1_{}'.format(p_n)]
            dA_dt2, d2A_d2t2 = locals()['dA_dt2_{}'.format(p_n)], locals()['d2A_d2t2_{}'.format(p_n)]
            d2U_d2t1, d2U_d2t2 = locals()['d2U_d2t1_{}'.format(p_n)], locals()['d2U_d2t2_{}'.format(p_n)]
            d2A_dt1t2, d2U_dt1t2 = locals()['d2A_dt1t2_{}'.format(p_n)], locals()['d2U_dt1t2_{}'.format(p_n)]
            t1_min, t1_max = self.PC.cell_volume_boundary_dic[p_n]
            t2_min, t2_max = self.PC.temp_opt_boundary_dic[p_n]
            sink_rate_P = locals()['k_sink_P_{}'.format(p_n)]

            # Time derivative of phyto. biomass
            temp = np.clip(U + 0.5 * V1 * d2A_d2t1 + 0.5 * V2 * d2A_d2t2 + C12 * d2A_dt1t2, 0, None)    # SAFETY
            locals()['dP_dt_{}'.format(p_n)] = P * temp - locals()['LG_{}'.format(p_n)] - locals()['LM_{}'.format(p_n)]

            # SAFETY
            locals()['dP_dt_{}'.format(p_n)] = np.where((locals()['dP_dt_{}'.format(p_n)] + P) < 1, 0, locals()['dP_dt_{}'.format(p_n)])

            dP_x_dt = locals()['dP_dt_{}'.format(p_n)] + self.v_sink(P, sink_rate_P) / self.dt
            sink_rate_P_ad = np.where((dP_x_dt + P).min(axis=0) < 1, 0, 1) * sink_rate_P
            locals()['dP_dt_{}'.format(p_n)] = locals()['dP_dt_{}'.format(p_n)] + self.v_sink(P, sink_rate_P_ad) / self.dt
            diff_sink_t1, diff_sink_V1 = self.v_sink_trait(P, sink_rate_P_ad, t1, V1)

            # Time derivative of phyto. trait distribution
            temp = np.clip(U + 0.5 * V1 * d2U_d2t1 + 0.5 * V2 * d2U_d2t2 + C12 * d2U_dt1t2, 0, None)  # SAFETY
            locals()['dt1_dt_{}'.format(p_n)] = V1 * dA_dt1 + C12 * dA_dt2 + diff_sink_t1
            locals()['dt2_dt_{}'.format(p_n)] = V2 * dA_dt2 + C12 * dA_dt1
            locals()['dV1_dt_{}'.format(p_n)] = V1 ** 2 * d2A_d2t1 + 2 * V1 * C12 * d2A_dt1t2 + C12 ** 2 * d2A_d2t2 + 2 * self.PC.diff_t1 * temp + diff_sink_V1
            locals()['dV2_dt_{}'.format(p_n)] = V2 ** 2 * d2A_d2t2 + 2 * V2 * C12 * d2A_dt1t2 + C12 ** 2 * d2A_d2t1 + 2 * self.PC.diff_t2 * temp
            locals()['dC12_dt_{}'.format(p_n)] = V1 * C12 * d2A_d2t1 + (V1 * V2 + C12 ** 2) * d2A_dt1t2 + V2 * C12 * d2A_d2t2

            # SAFETY
            locals()['dt1_dt_{}'.format(p_n)] = np.where(
                ((t1 + locals()['dt1_dt_{}'.format(p_n)]) < t1_min) | ((t1 + locals()['dt1_dt_{}'.format(p_n)]) > t1_max), 0, locals()['dt1_dt_{}'.format(p_n)])
            locals()['dt2_dt_{}'.format(p_n)] = np.where(
                ((t2 + locals()['dt2_dt_{}'.format(p_n)]) < t2_min) | ((t2 + locals()['dt2_dt_{}'.format(p_n)]) > t2_max), 0, locals()['dt2_dt_{}'.format(p_n)])
            locals()['dV1_dt_{}'.format(p_n)] = np.where(
                ((V1 + locals()['dV1_dt_{}'.format(p_n)]) < 1e-3) | ((V1 + locals()['dV1_dt_{}'.format(p_n)]) > 9), 0, locals()['dV1_dt_{}'.format(p_n)])
            locals()['dV2_dt_{}'.format(p_n)] = np.where(
                ((V2 + locals()['dV2_dt_{}'.format(p_n)]) < 1e-3) | ((V2 + locals()['dV2_dt_{}'.format(p_n)]) > 9), 0, locals()['dV2_dt_{}'.format(p_n)])
            locals()['dC12_dt_{}'.format(p_n)] = np.where(
                ((C12 + locals()['dC12_dt_{}'.format(p_n)]) < -5) | ((C12 + locals()['dC12_dt_{}'.format(p_n)]) > 5), 0, locals()['dC12_dt_{}'.format(p_n)])


        # ---------------------------Update phyto. in vertical diffusion---------------------------
        for p_n in self.PC.p_names:
            t1, t2, P = locals()['t1_{}'.format(p_n)], locals()['t2_{}'.format(p_n)], locals()['P_{}'.format(p_n)]
            V1, V2, C12 = locals()['V1_{}'.format(p_n)], locals()['V2_{}'.format(p_n)], locals()['C12_{}'.format(p_n)]
            dP_dt = locals()['dP_dt_{}'.format(p_n)]
            dt1_dt, dt2_dt = locals()['dt1_dt_{}'.format(p_n)], locals()['dt2_dt_{}'.format(p_n)]
            dV1_dt, dV2_dt = locals()['dV1_dt_{}'.format(p_n)], locals()['dV2_dt_{}'.format(p_n)]
            dC12_dt = locals()['dC12_dt_{}'.format(p_n)]

            # Derivatives under Vertical diffusion of [P, Pt1, Pt2, PV1t1, PV2t2, PC12t1t2]
            locals()['dP_{}_dt'.format(p_n)] = dP_dt * self.dt + self.v_diff(P, kz_p)
            kz_p_ad = np.where((locals()['dP_{}_dt'.format(p_n)] + P).min(axis=0) < 1, 0, 1) * kz_p
            locals()['dP_{}_dt'.format(p_n)] = dP_dt * self.dt + self.v_diff(P, kz_p_ad)
            locals()['dPt1_{}_dt'.format(p_n)] = (P * dt1_dt + t1 * dP_dt) * self.dt + self.v_diff(P * t1, kz_p_ad)
            locals()['dPt2_{}_dt'.format(p_n)] = (P * dt2_dt + t2 * dP_dt) * self.dt + self.v_diff(P * t2, kz_p_ad)
            locals()['dPV1t1_{}_dt'.format(p_n)] = (P * dV1_dt + 2 * P * t1 * dt1_dt + (V1 + t1 ** 2) * dP_dt) * self.dt + self.v_diff(P * (V1 + t1 ** 2), kz_p_ad)
            locals()['dPV2t2_{}_dt'.format(p_n)] = (P * dV2_dt + 2 * P * t2 * dt2_dt + (V2 + t2 ** 2) * dP_dt) * self.dt + self.v_diff(P * (V2 + t2 ** 2), kz_p_ad)
            locals()['dPC12t1t2_{}_dt'.format(p_n)] = (P * dC12_dt + P * t1 * dt2_dt + P * t2 * dt1_dt + (C12 + t1 * t2) * dP_dt) * self.dt + self.v_diff(P * (C12 + t1 * t2), kz_p_ad)

            # Nutrient quota QPmin: [mg P / mm3]        P_uptake: [mg P / m3] = P [mm3 / m3] * Q [mg P / mm3]
            locals()['QPmin_crt_{}'.format(p_n)] = np.power(10, self.SL.cell_volume_2_QPmin(t1 + locals()['dt1_dt_{}'.format(p_n)] * self.dt, per_x=True))      # [mg P / mm3]
            locals()['P_uptake_{}'.format(p_n)] = (P + dP_dt + locals()['LM_{}'.format(p_n)] + locals()['LG_{}'.format(p_n)]) * locals()['QPmin_crt_{}'.format(p_n)] - P * locals()['QPmin_lst_{}'.format(p_n)]
            locals()['P_release_{}'.format(p_n)] = (locals()['LM_{}'.format(p_n)] + locals()['LG_{}'.format(p_n)]) * locals()['QPmin_crt_{}'.format(p_n)]

            # Set to y_output
            for sv_base_name in self.sv_vars_names:
                sv_name = sv_base_name.format(p_n)
                i_sv = self.sv_names.index(sv_name)
                y_output[i_sv] = locals()['d{}_dt'.format(sv_name)]


        # ---------------------------Nutrient dynamics---------------------------
        # Phyto. loss
        P_release_sum = np.zeros(shape=(self.nz, len(self.lake_ids)), dtype=np.float32)  # in unit: mg P / m3
        P_uptake_sum = np.zeros(shape=(self.nz, len(self.lake_ids)), dtype=np.float32)  # in unit: mg P / m3
        d_P_phyto_sum = np.zeros(shape=(self.nz, len(self.lake_ids)), dtype=np.float32)  # in unit: mg P / m3
        for p_n in self.PC.p_names:
            P_uptake_sum += locals()['P_uptake_{}'.format(p_n)]  # P [mm3 / m3] * Q [mg P / mm3] = [mg P / m3]
            P_release_sum += locals()['P_release_{}'.format(p_n)]
            d_P_phyto_sum += locals()['dP_dt_{}'.format(p_n)] * locals()['QPmin_crt_{}'.format(p_n)]
        # POP Mine
        P_mine0, T0, alpha_h = self.PC.P_mine0, self.PC.T0, self.PC.alpha_h
        mine_P = P_mine0 * np.exp(alpha_h * (t_p_oC - T0)) * locals()['POP']
        # POP dynamics
        dPOP_dt = (P_release_sum * (1 - self.PC.frac_release_DIP) - mine_P) * self.dt + self.v_diff(locals()['POP'], kz_p) + self.v_sink(locals()['POP'], self.PC.k_sink_P)
        # SAFETY
        dPOP_dt = np.where((locals()['POP'] + dPOP_dt).min(axis=0) < 1, 0, dPOP_dt)
        # DIP dynamics
        dDIP_dt = (P_release_sum * self.PC.frac_release_DIP + mine_P - P_uptake_sum) * self.dt + self.v_diff(locals()['DIP'], kz_p)
        dDIP_dt = np.where((locals()['DIP'] + dDIP_dt).min(axis=0) < 1, 0, dDIP_dt)
        # TP dynamics
        dTP_dt = dPOP_dt + dDIP_dt + d_P_phyto_sum * self.dt
        # Set to y_output
        for i_sv, sv_name in enumerate(self.sv_const_names):
            y_output[i_sv] = locals()['d{}_dt'.format(sv_name)]


        # ---------------------------Return dSV_dt---------------------------
        return y_output.flatten()

    def v_diff(self, C, kz, add_sediment=False, solve_scheme='Implicit', split=100):
        """
        C: concentration in shape (nz, n_lake)
        kz: Vertical diffusion coefficient in shape (nz, n_lake), the n-1 kz is between boxes and n-th is water-sediment kz
        sink_rate: Vertical sink rate in m/d
        dz: Vertical box height in shape (n_lake, )
        dt: in day

        # Consider water-sediment diffusion, and set sediment conc. const.

        return diff flux in shape (nz, n_lake)
        """
        if add_sediment:
            CC = np.concatenate((C, self.IC.df.loc[self.lake_ids, add_sediment].values.reshape(1, -1)))     # (nz+1, n_lake)
        else:
            CC = C
        nz, n_lake = CC.shape

        # Initialize the inverse of the transport matrix (T)
        T = np.zeros((n_lake, nz, nz))

        if solve_scheme == 'Implicit':
            # Filling the matrix T between boxes
            for i in range(1, nz - 1):
                # Box in water is same and use dz
                T[:, i-1, i] = -kz[i-1, :] * self.dt / self.dz ** 2
                T[:, i+1, i] = -kz[i, :] * self.dt / self.dz ** 2
                T[:, i, i] = 1 - T[:, i-1, i] - T[:, i+1, i]

            # Filling boundary conditions
            T[:, 1, 0] = -kz[0, :] * self.dt / self.dz ** 2
            T[:, 0, 0] = 1 - T[:, 1, 0]
            if add_sediment:
                T[:, nz-2, nz-1] = -kz[nz-2, :] * self.dt / self.d_sediment ** 2
            else:
                T[:, nz-2, nz-1] = -kz[nz-2, :] * self.dt / self.dz ** 2
            T[:, nz-1, nz-1] = 1 - T[:, nz-2, nz-1]

            # Solve for DIFF
            CC_new = np.linalg.solve(T, CC.T.reshape(len(self.lake_ids), self.nz, 1))  # Solving T * C_new = C
            CC_new = CC_new.reshape(len(self.lake_ids), self.nz)

            # DIFF = (CC_new.T - CC) / self.dt       # (nz+1, n_lake) or (nz, n_lake)
            DIFF = (CC_new.T - CC)       # (nz+1, n_lake) or (nz, n_lake)

        return DIFF[:-1] if add_sediment else DIFF

    def v_sink(self, C, k_sink, ratio_max=0.2):
        """
        C: concentration in shape (nz, n_lake)
        dz: Vertical box height in shape (n_lake, )
        k_sink: sink rate (m/d) in shape (nz, n_lake)
        """
        nz, n_lake = C.shape
        if isinstance(k_sink, float):
            k_sink = np.zeros(shape=(nz, n_lake), dtype=np.float32) + k_sink
        k_sink = np.where((k_sink / self.dz) > ratio_max, self.dz * ratio_max, k_sink)        # SAFETY
        k_sink[-1] = 0      # bottom cannot sink

        C_add_boundary = np.concatenate((np.zeros((1, n_lake)), C))     # Add top boundary (nz+1, n_lake)
        k_sink_add_boundary = np.concatenate((np.zeros((1, n_lake)), k_sink))     # Add top boundary
        sink_in = (k_sink_add_boundary[:-1] / self.dz) * C_add_boundary[: -1]
        sink_out = (k_sink_add_boundary[1:] / self.dz) * C_add_boundary[1:]
        Sink = (sink_in - sink_out) * self.dt

        return Sink

    def v_sink_trait(self, P, k_sink_P, t1, V1, ratio_max=0.1):
        """
        Calc phytoplankton trait changes under sinking process
        """
        nz, n_lake = P.shape
        k_sink_P_ad = np.where((k_sink_P / self.dz) > ratio_max, self.dz * ratio_max, k_sink_P)
        sink_P = (k_sink_P_ad / self.dz) * P
        sink_in_P = np.zeros(shape=(nz, n_lake), dtype=np.float32)
        sink_in_P[1:] = sink_P[:-1]
        sink_out_P = np.zeros(shape=(nz, n_lake), dtype=np.float32)
        sink_out_P[:-1] = sink_P[:-1]

        sink_t1 = t1 + self.PC.scaling_slope_dic['cell_volume_2_sink_rate'] * self.PC.ln10 * V1
        sink_in_t1 = np.zeros(shape=(nz, n_lake), dtype=np.float32)
        sink_in_t1[1:] = sink_t1[:-1]
        sink_out_t1 = np.zeros(shape=(nz, n_lake), dtype=np.float32)
        sink_out_t1[:-1] = sink_t1[:-1]

        sink_in_V1 = np.zeros(shape=(nz, n_lake), dtype=np.float32)
        sink_in_V1[1:] = V1[:-1]
        sink_out_V1 = np.zeros(shape=(nz, n_lake), dtype=np.float32)
        sink_out_V1[:-1] = V1[:-1]

        t1_new = (P * t1 - sink_out_P * sink_out_t1 + sink_in_P * sink_in_t1) / (P - sink_out_P + sink_in_P)
        V1_new = (P * (t1**2 + V1) - sink_out_P * (sink_out_t1**2 + sink_out_V1) + sink_in_P * (sink_in_t1**2 + sink_in_V1)) / (P - sink_out_P + sink_in_P) - t1_new ** 2

        diff_t1 = t1_new - t1
        diff_V1 = V1_new - V1
        return diff_t1, diff_V1

    def continuous_dv(self, t, chla=None, max_save=1e4):
        """
        Load model Driver Variables (DV)
        chla in unit mg / mm3 in shape (nz, n_lake)
        Return (temp_profile, rad_profile, kz_profile,...)
        """
        t_int = int(t)
        crt_date = self.date_range[t_int]
        lst_date = crt_date + pd.Timedelta(days=-1)
        lst2_date = crt_date + pd.Timedelta(days=-2)

        # Check did calc driver vars of that day. Load crt date's driver vars and save to dic
        if crt_date not in self.dv_dic.keys():
            temp_profile = self.FO.calc_discrete_temp(crt_date)     # (nz, n_lake)
            radiation_profile = self.FO.calc_discrete_radiation(crt_date, chla, k_ext_chla=self.PC.KEXTchla)
            kz_profile = self.FO.calc_discrete_kz(crt_date, temp_profile, None if lst_date not in self.dv_dic.keys() else self.dv_dic[lst_date][0], None if lst2_date not in self.dv_dic.keys() else self.dv_dic[lst2_date][0])
            self.dv_dic[crt_date] = (temp_profile, radiation_profile, kz_profile, )
            self.chla_dic[crt_date] = copy.deepcopy(chla)
            _ = self.log_bar.update() if self.log_bar is not None else None
            if len(self.dv_dic) > max_save:
                self.dv_dic.pop(crt_date + pd.Timedelta(days=-max_save))
        return self.dv_dic[crt_date]

    def continuous_solve(self, solver='RK45'):
        """
        Y in shape (n_sv, nz, n_lake) and flatten()
        Use local solver with constant time step
        """
        # Load ini. y
        y_ini = self.load_ini_values()

        # Solve model
        n_date = len(self.date_range)
        self.log_bar = tqdm(
            total=n_date, ncols=150,
            desc='RUN continuous Lakes:{}\t:{} to {}'.format(len(self.lake_ids), self.date_range[0].strftime('%Y-%m-%d'), self.date_range[-1].strftime('%Y-%m-%d'))
        )
        steps = np.linspace(self.dt, n_date-1, int((n_date-1) / self.dt))
        y_input = copy.deepcopy(y_ini.flatten())
        log = np.ones(shape=(n_date, len(self.sv_names) * self.nz * len(self.lake_ids)), dtype=np.float32)
        log[0] = copy.deepcopy(y_ini.flatten())

        # Step by step
        i_day = 0
        for i_step, t_step in enumerate(steps):
            if solver == 'RK45':    # RK45
                v4, v = DoPri45Step(self.continuous_odes, t_step-self.dt, y_input, self.dt)
            else:   # RK2
                v = RK2Step(self.continuous_odes, t_step - self.dt, y_input, self.dt)
            y_input += v
            if int(t_step // 1) != i_day:
                i_day += 1
                log[i_day] = copy.deepcopy(self.check_nutrient_balance(y_ini, y_input))
        self.log_bar.close()

        # Log results, sol.y in shape: (n_eval, n_y) -> (n_eval, n_sv, nz, n_lake)
        y_output = log.reshape(log.shape[0], len(self.sv_names), self.nz, len(self.lake_ids))
        self.y_output = y_output
        return 0

    def load_ini_values(self):
        """
        Y in shape (n_sv, nz, n_lake) and flatten()
        """
        # Load ini. y
        y_ini = np.zeros(shape=(len(self.sv_names), self.nz, len(self.lake_ids)), dtype=np.float32)
        for i_sv, sv_name in enumerate(self.sv_names):
            for i_nz in range(self.nz):
                y_ini[i_sv, i_nz] = getattr(self.SV, '{}_{}'.format(sv_name, i_nz)).loc[self.date_range[0], self.lake_ids].values
        return y_ini

    def reset_lake_nutrient(self, tp=None):
        """
        For nutrient scenario.
        """
        if tp is not None:
            print('reset nutrient')
            for i_nz in range(self.nz):
                for sv_name, ratio in {'TP': 1, 'POP': 0.7, 'DIP': 0.3}.items():
                    df = getattr(self.SV, '{}_{}'.format(sv_name, i_nz))
                    df.loc[self.date_range[0], self.lake_ids] = tp * ratio
                    setattr(self.SV, '{}_{}'.format(sv_name, i_nz), df)

    def check_nutrient_balance(self, y_ini, y_input):
        """
        Check nutrient (in phyto. + in water = constant) conservation
        """
        # Ini nutrient
        i_dip, i_pop = self.sv_names.index('DIP'), self.sv_names.index('POP')
        ini_water = y_ini[i_dip] + y_ini[i_pop]
        ini_nutrient_phyto = np.zeros(shape=(self.nz, len(self.lake_ids)), dtype=np.float32)
        for p_name in self.PC.p_names:
            i_P = self.sv_names.index('P_{}'.format(p_name))
            i_Pt1 = self.sv_names.index('Pt1_{}'.format(p_name))
            P = np.clip(y_ini[i_P], 1, None)
            Pt1 = y_ini[i_Pt1]
            t1 = Pt1 / P
            t1 = np.clip(t1, self.PC.cell_volume_boundary_dic[p_name][0], self.PC.cell_volume_boundary_dic[p_name][1])  # (nz, n_lake)
            quota = np.power(10, self.SL.cell_volume_2_QPmin(t1, per_x=True)) * P  # [mg P / mm3]
            ini_nutrient_phyto += quota
        ini_nutrient_all = ini_water + ini_nutrient_phyto
        ini_nutrient_all_lake = ini_nutrient_all.sum(axis=0)    # (n_lake, )

        # nutrient in phyto
        y_input = y_input.reshape(len(self.sv_names), self.nz, len(self.lake_ids))
        nutrient_phyto = np.zeros(shape=(self.nz, len(self.lake_ids)), dtype=np.float32)
        for p_name in self.PC.p_names:
            i_P = self.sv_names.index('P_{}'.format(p_name))
            i_Pt1 = self.sv_names.index('Pt1_{}'.format(p_name))
            P = np.clip(y_input[i_P], 1, None)
            Pt1 = y_input[i_Pt1]
            t1 = Pt1 / P
            t1 = np.clip(t1, self.PC.cell_volume_boundary_dic[p_name][0], self.PC.cell_volume_boundary_dic[p_name][1])      # (nz, n_lake)
            quota = np.power(10, self.SL.cell_volume_2_QPmin(t1, per_x=True)) * P     # [mg P / mm3]
            nutrient_phyto += quota
        dip = np.clip(y_input[i_dip], 1.5, None)
        pop = np.clip(y_input[i_pop], 1.5, None)
        nutrient_water = dip + pop
        nutrient_water_lake = nutrient_water.sum(axis=0)
        nutrient_all = nutrient_phyto + nutrient_water
        nutrient_all_lake = nutrient_all.sum(axis=0)

        ratio = 1 - (nutrient_all_lake - ini_nutrient_all_lake) / nutrient_water_lake
        y_input[i_dip] = dip * ratio
        y_input[i_pop] = pop * ratio
        return y_input.flatten()

    def save_res(self, save_dir='Results', res_i='', suffix='', save_nz=None):
        save_nz = self.nz if save_nz is None else save_nz

        # y_output in (n_eval, n_sv, nz, n_lake)
        y_output = self.y_output
        for p_n in self.PC.p_names:
            i_P = self.sv_names.index('P_{}'.format(p_n))
            i_Pt1, i_Pt2 = self.sv_names.index('Pt1_{}'.format(p_n)), self.sv_names.index('Pt2_{}'.format(p_n))
            i_PV1t1, i_PV2t2 = self.sv_names.index('PV1t1_{}'.format(p_n)), self.sv_names.index('PV2t2_{}'.format(p_n))
            i_PC12t1t2 = self.sv_names.index('PC12t1t2_{}'.format(p_n))
            P = y_output[:, i_P, :, :]  # (n_eval, nz, n_lake)
            Pt1, Pt2 = y_output[:, i_Pt1, :, :], y_output[:, i_Pt2, :, :]
            PV1t1, PV2t2 = y_output[:, i_PV1t1, :, :], y_output[:, i_PV2t2, :, :]
            PC12t1t2 = y_output[:, i_PC12t1t2, :, :]

            # Back-calc
            t1, t2 = Pt1 / P, Pt2 / P
            V1, V2 = PV1t1 / P - t1 ** 2, PV2t2 / P - t2 ** 2
            C12 = PC12t1t2 / P - t1 * t2

            phyto_dic = {
                'P_{}'.format(p_n): P,
                't1_{}'.format(p_n): t1, 't2_{}'.format(p_n): t2,
                'V1_{}'.format(p_n): V1, 'V2_{}'.format(p_n): V2,
                'C12_{}'.format(p_n): C12,
            }
            for i_nz in range(save_nz):
                for phyto_name, data in phyto_dic.items():
                    df = pd.DataFrame(data[:, i_nz, :], index=self.date_range[: y_output.shape[0]], columns=self.lake_ids)
                    df.to_csv(os.path.join(save_dir, 'Model1D_Res{} {}{}{}.csv'.format(res_i, phyto_name, i_nz, suffix)))

        for i_sv, sv_name in enumerate(self.sv_const_names):
            for i_nz in range(save_nz):
                df = pd.DataFrame(y_output[:, i_sv, i_nz, :], index=self.date_range[: y_output.shape[0]], columns=self.lake_ids)
                df.to_csv(os.path.join(save_dir, 'Model1D_Res{} {}{}{}.csv'.format(res_i, sv_name, i_nz, suffix)))

        chl_arr = np.zeros(shape=(len(self.date_range), self.nz, len(self.lake_ids)), dtype=np.float32)
        wt_arr = np.zeros(shape=(len(self.date_range), self.nz, len(self.lake_ids)), dtype=np.float32)
        rd_arr = np.zeros(shape=(len(self.date_range), self.nz, len(self.lake_ids)), dtype=np.float32)
        kz_arr = np.zeros(shape=(len(self.date_range), self.nz, len(self.lake_ids)), dtype=np.float32)
        for i_date, date in enumerate(self.date_range):
            if date in self.chla_dic.keys():
                chl_arr[i_date] = self.chla_dic[date]
            if date in self.dv_dic.keys():
                wt_arr[i_date] = self.dv_dic[date][0]
                rd_arr[i_date] = self.dv_dic[date][1]
                kz_arr[i_date] = self.dv_dic[date][2]

        dic = {
            'Chla': chl_arr,
            'WT': wt_arr,
            'RD': rd_arr,
            'KZ': kz_arr,
        }
        for i_nz in range(save_nz):
            for name, arr in dic.items():
                df = pd.DataFrame(arr[:, i_nz, :], index=self.date_range, columns=self.lake_ids)
                df.to_csv(os.path.join(save_dir, 'Model1D_Res{} {}{}{}.csv'.format(res_i, name, i_nz, suffix)))

    def adjust_paras(self, use_paras_dic):
        """
        For lake-specific calibration
        """
        for full_name, value in use_paras_dic.items():
            if_value = True if full_name[:5] == 'value' else False
            p_name = full_name[8:]
            if full_name[6] == 'P':
                ini_value = copy.deepcopy(getattr(self.PC, '{}_ini'.format(p_name)))
                if str(p_name).endswith('_dic'):
                    for k, v in ini_value.items():
                        ini_value[k] = value if if_value else v * value
                    setattr(self.PC, p_name, ini_value)
                else:   # float or array
                    setattr(self.PC, p_name, value if if_value else ini_value * value)


if __name__ == '__main__':
    t = LakePhytoModel1D(
        lake_ids=lake_ids,
        date_range=pd.date_range('2010-1-1', '2100-12-31', freq='D'),
        nz=10,
        dt=5e-2,
        scenario='ssp245',
    )
    t.continuous_solve()
    t.save_res(res_i=1)
