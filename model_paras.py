import pandas as pd
import numpy as np


lake_ids = ['L{}'.format(i) for i in range(1, 143)]
lake_meta_df = pd.read_csv('Data/Lake142Meta.csv', index_col=0, header=0).loc[lake_ids]
used_lake_ids = lake_meta_df[lake_meta_df['Use'] == 1].index


class Paras:
    # ---------------------Phyto. paras---------------------
    p_names = ['b', 'd', 'g']
    deltaT = 5.         # Width on the right side above the maximum growth rate temperature
    T0 = 20             # Base temperature for temperature-dependence processes
    KEXTchla = 0.02             # Light attenuation coefficient for chlorophyll in unit: m2/mg (Ref. Arh.2005)
    diff_t1 = 0.0003    # trait 1 diffusion coefficient
    diff_t2 = 0.003     # trait 2 diffusion coefficient


    # ---------------------Nutrient paras---------------------
    alpha_h = np.log(2.5) / 10.  # Temperature dependence factor for heterotrophic processes, Q10=2.5
    alpha_m = np.log(2.5) / 10.  # Temperature dependence factor for phyto. death processes, Q10
    P_mine0 = 0.1  # P mine. rate at 20 C
    frac_release_DIP = 0.3  # The percentage of phytoplankton releasing P as DIP
    k_sink_P = .2  # Sink rate of POP, 1.2 (Ref. Gland 2021)
    ln10 = 2.3025851    # constant: np.log(10)


    # ---------------------Phyto. scaling laws---------------------
    """
    Eq: [y] = slope * [x] + intercept
        All input [x] is in log10, return [y] also in log10
        Used unit:
            volume: µm3
            wights: 1e-12 g
            concen: µg/L == mg/m3
    """
    scaling_slope_dic = {
        'cell_volume_2_cell_mass_C': .86,  # 2024 Wickman
        'cell_volume_2_uptake_max_N': .82,  # 2012 Edwards
        'cell_volume_2_uptake_max_P': .94,  # 2012 Edwards
        'cell_volume_2_KN_fresh': .52,  # 2012 Edwards
        'cell_volume_2_KN_ocean': .33,  # 2012 Edwards
        'cell_volume_2_KP': .41,  # 2012 Edwards
        'cell_volume_2_QNmin': .84,  # 2012 Edwards
        'cell_volume_2_QPmin': .96,  # 2012 Edwards
        'cell_volume_2_sink_rate': .433,  # 2019 Durante (m/d)
        'irradiance_2_chla_c_ratio': -0.44050665  # Geider1997 & Zonneveld1998
    }
    scaling_intercept_dic = {
        'cell_volume_2_cell_mass_C': -.5872,  # 2024 Wickman
        'cell_volume_2_uptake_max_N': -.813575,  # 2012 Edwards
        'cell_volume_2_uptake_max_P': -1.27171,  # 2012 Edwards
        'cell_volume_2_KN_fresh': .431,  # 2012 Edwards
        'cell_volume_2_KN_ocean': .304,  # 2012 Edwards
        'cell_volume_2_KP': .1374,  # 2012 Edwards
        'cell_volume_2_QNmin': -1.9,  # 2012 Edwards
        'cell_volume_2_QPmin': -3.1752,  # 2012 Edwards
        'cell_volume_2_sink_rate': -1.769,  # 2019 Durante (m/d)
        'irradiance_2_chla_c_ratio': -0.96376  # Geider1997 & Zonneveld1998
    }


    # ---------------------Phyto. inter-group constants---------------------
    growth_rate_t0_dic = {'b': 1.51, 'd': 2.20, 'g': 2.97, }        # growth rate at 20 oC Kremer2017
    growth_rate_t0_dic_ini = {'b': 1.51, 'd': 2.20, 'g': 2.97, }        # growth rate at 20 oC Kremer2017
    light_ini_slope_dic = {'b': 0.04, 'd': 0.018, 'g': 0.02, }      # 2011Schwaderer
    light_opt_I_dic = {'b': 100, 'd': 180, 'g': 160, }              # 2011Schwaderer, unit: µmol quota/m2/s
    cell_volume_boundary_dic = {'b': [1e-3, 3.5], 'd': [1.5, 7], 'g': [1.0, 4.5], }  # Without negative values      Kremer2017 Litchman2022
    temp_opt_boundary_dic = {'b': [25, 40], 'd': [16, 22], 'g': [20, 26], }
    grow_max_Q10_dic = {'b': 0.0470715, 'd': 0.0474316, 'g': 0.04708183, }    # ini Q10     Kremer2017
    graze_rate_dic = {'b': 0.01, 'd': 0.1, 'g': 0.1, }              # /d
    morality_rate_dic = {'b': 0.05, 'd': 0.1, 'g': 0.1, }


    # ---------------------Model run ini. values (Do not influence long-term results)---------------------
    phyto_ini_dic = {
        'P_b': 100, 'P_d': 100, 'P_g': 100,
        't1_b': 1, 't1_d': 4, 't1_g': 2,
        't2_b': 29, 't2_d': 20, 't2_g': 24,
        'V1_b': 0.05, 'V1_d': 0.05, 'V1_g': 0.05,
        'V2_b': 0.5, 'V2_d': 0.5, 'V2_g': 0.5,
        'C12_b': 0, 'C12_d': 0, 'C12_g': 0,
    }
    for p_n in p_names:
        phyto_ini_dic['Pt1_{}'.format(p_n)] = phyto_ini_dic['P_{}'.format(p_n)] * phyto_ini_dic['t1_{}'.format(p_n)]
        phyto_ini_dic['Pt2_{}'.format(p_n)] = phyto_ini_dic['P_{}'.format(p_n)] * phyto_ini_dic['t2_{}'.format(p_n)]
        phyto_ini_dic['PV1t1_{}'.format(p_n)] = phyto_ini_dic['P_{}'.format(p_n)] * (phyto_ini_dic['V1_{}'.format(p_n)] + phyto_ini_dic['t1_{}'.format(p_n)] ** 2)
        phyto_ini_dic['PV2t2_{}'.format(p_n)] = phyto_ini_dic['P_{}'.format(p_n)] * (phyto_ini_dic['V2_{}'.format(p_n)] + phyto_ini_dic['t2_{}'.format(p_n)] ** 2)
        phyto_ini_dic['PC12t1t2_{}'.format(p_n)] = phyto_ini_dic['P_{}'.format(p_n)] * (phyto_ini_dic['C12_{}'.format(p_n)] + phyto_ini_dic['t1_{}'.format(p_n)] * phyto_ini_dic['t2_{}'.format(p_n)])