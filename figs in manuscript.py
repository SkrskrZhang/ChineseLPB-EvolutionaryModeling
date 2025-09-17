import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress
import statsmodels.formula.api as smf
from scipy import stats
from Utils.lake_data_process import calc_std_n2p_ratio_from_p, spring_limit_nutrient_summer_chl_relationship, split_lake_by_summer_stratification, region_colors, region_names, chl_months


def fig1_limited_nutrient_and_chla_responses():
    lake_meta_df = pd.read_csv('Data/Lake142Meta.csv', index_col=0, header=0)
    lake_nutrient_df = spring_limit_nutrient_summer_chl_relationship().dropna().loc[lake_meta_df['Use'] == 1]
    stratification_ss = split_lake_by_summer_stratification().loc[lake_nutrient_df.index]
    n_plimit = len(lake_nutrient_df[lake_nutrient_df['Limit_Type'] == 'P'])
    n_nlimit = len(lake_nutrient_df) - n_plimit
    print('Plimit', n_plimit, 'Nlimit', n_nlimit)
    print('stratification', len(stratification_ss[stratification_ss == 1]))
    print('chl>12', len(lake_nutrient_df[lake_nutrient_df['Chla'] > 12]))
    lw = 0.5

    # plt1: lake nutrient limit type
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    ax = axes[0]
    plt.sca(ax)
    for p_stratification, marker in enumerate(['o', '^']):
        plt.scatter(
            lake_nutrient_df['TP'][stratification_ss == p_stratification],
            lake_nutrient_df['N2P'][stratification_ss == p_stratification],
            marker=marker,
            s=20,
            linewidths=lw,
            facecolors=[region_colors[int(lake_meta_df.loc[lake_id, 'Region_ID'])] for lake_id in lake_nutrient_df[stratification_ss == p_stratification].index],
            edgecolors='k',
            alpha=1,
            zorder=101,
        )
    tp_arr = np.linspace(0.5, 2.5, 100)
    n2p_arr = calc_std_n2p_ratio_from_p(np.power(10, tp_arr))
    plt.plot(np.power(10, tp_arr), n2p_arr, c='r', lw=lw*2, ls=':', zorder=100, label='Critical ratio: log(y) = -0.19×log(x) + 1.5')
    plt.xlabel('Spring TP ($μg$ $L^{-1}$)', fontsize=9)
    plt.ylabel('Spring TN:TP', fontsize=9)
    plt.xlim(10**0.5, 10**2.5)
    plt.ylim(10**0.5, 10**2.5)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(lw=lw, alpha=0.4)
    for spine in ax.spines.values():
        spine.set_linewidth(lw)
    ax.tick_params(axis='both', which='major', width=lw, length=3, labelsize=8)
    ax.legend(ncol=1, loc='lower left', fontsize=8, frameon=False)

    # plt2: lake nutrient-chl response
    ax = axes[1]
    plt.sca(ax)
    for p_stratification, marker in enumerate(['o', '^']):
        plt.scatter(
            lake_nutrient_df['Limit_P'][stratification_ss == p_stratification],
            lake_nutrient_df['Chla'][stratification_ss == p_stratification],
            marker=marker,
            s=20,
            linewidths=lw,
            facecolors=[region_colors[int(lake_meta_df.loc[lake_id, 'Region_ID'])] for lake_id in lake_nutrient_df[stratification_ss == p_stratification].index],
            edgecolors='k',
            alpha=1,
            zorder=99,
        )
    for region_id in [1, 0, 3, 2, 4]:
        region_color = region_colors[region_id]
        if region_id < 4:
            region_lake_ids = [lake_id for lake_id in lake_nutrient_df.index if int(lake_meta_df.loc[lake_id, 'Region_ID']) == region_id]
        else:
            region_lake_ids = lake_nutrient_df.index
        x = np.log10(lake_nutrient_df.loc[region_lake_ids, 'Limit_P'].values)
        y = np.log10(lake_nutrient_df.loc[region_lake_ids, 'Chla'].values)
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        y_pred = x * slope + intercept
        if p_value < 1e-4:
            p_str = ', $\it{p}$<0.0001'
        elif p_value < 1e-3:
            p_str = ', $\it{p}$<0.001'
        elif p_value < 1e-2:
            p_str = ', $\it{p}$<0.01'
        elif p_value < 5e-2:
            p_str = ', $\it{p}$<0.05'
        elif p_value < 1e-1:
            p_str = ', $\it{p}$<0.1'
        else:
            p_str = ''
        plt.plot(10**x, 10**y_pred, c=region_color, lw=lw*2, label='{} (slope={:.2f}{})'.format(region_names[region_id], slope, p_str), zorder=98)
        print(region_id, slope, std_err, r_value, p_value, np.power(12 * 10 ** -intercept, 1/slope))
        if region_id == 4:
            quantreg_model = smf.quantreg('chl ~ tp', pd.DataFrame({'chl': y, 'tp': x}))
            quantreg_model_q90 = quantreg_model.fit(q=0.95)
            quantreg_model_q90_pred = np.array([x.min(), x.max()]) * quantreg_model_q90.params['tp'] + quantreg_model_q90.params['Intercept']
            plt.plot(10**np.array([x.min(), x.max()]), 10**quantreg_model_q90_pred, c=region_color, lw=lw * 2, label='{} (slope={:.2f})'.format('Q95', quantreg_model_q90.params['tp']), ls='--', zorder=98)
            print(np.std(y - y_pred), np.power(12 * 10 ** -quantreg_model_q90.params['Intercept'], 1/quantreg_model_q90.params['tp']))
    plt.plot([10**0.5, 10**2.5], [12, 12], c='r', lw=lw * 2, label='Alert level: Chl$\it{a}$=12 $μg$ $L^{-1}$', ls=':', zorder=0, alpha=1)
    plt.xlabel('Spring limited nutrient ($μg$ $P$ $equivalent$ $L^{-1}$)', fontsize=9)
    plt.ylabel('Summer surface Chl$\it{a}$ ($μg$ $L^{-1}$)', fontsize=9)
    plt.xlim(10**0.5, 10**2.5)
    plt.ylim(10**0, 10**2.5)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(lw=lw, alpha=0.4)
    for spine in ax.spines.values():
        spine.set_linewidth(lw)
    ax.tick_params(axis='both', which='major', width=lw, length=3, labelsize=8)
    h, l = ax.get_legend_handles_labels()
    l1 = ax.legend(h[:-1], l[:-1], ncol=1, loc='upper left', fontsize=8, frameon=False)
    ax.legend(h[-1:], l[-1:], ncol=1, loc='lower right', fontsize=8, frameon=False)
    ax.add_artist(l1)

    plt.tight_layout()
    plt.savefig(os.path.join('Figs/MS', 'F1 limited_nutrient_and_chla_response.png'), dpi=300)
    # plt.show()


def fig2_responses_future_changes():
    ssp_ls = ['245', '585']
    log_years = [2060, 2100]
    idx = 1
    load_dir = 'Optimize'
    load_dir2 = 'TopSim{}'.format(idx)
    n_top = 1
    nz = 1
    lw = 0.5
    date_range = pd.date_range('2010-1-1', '2100-12-31')
    lake_meta_df = pd.read_csv('Data/Lake142Meta.csv', index_col=0, header=0)

    """
    plt nutrient-chla response changes
    """
    # load surface chl simulations
    for ssp in ssp_ls:
        data_arr = np.zeros(shape=(n_top, nz, len(date_range), len(lake_meta_df.index)), dtype=np.float32)
        used_data_arr = np.zeros(shape=(n_top, len(date_range), len(lake_meta_df.index)), dtype=np.float32)
        for i in range(n_top):
            for j in range(nz):
                data_dir = os.path.join(load_dir, '{}_ssp{}'.format(load_dir2, ssp))
                df = pd.read_csv(os.path.join(data_dir, 'Model1D_Res Chla{}-{}.csv'.format(j, i)), header=0, index_col=0)
                df.index = pd.to_datetime(df.index)
                data_arr[i, j] = df.loc[date_range, lake_meta_df.index].values
            used_data_arr[i] = data_arr[i, 0, :, range(len(lake_meta_df.index))].T

        # group by year-summer
        mean_df = pd.DataFrame(used_data_arr[0], index=date_range, columns=lake_meta_df.index)
        summer_mean_df = mean_df[mean_df.index.month.isin(chl_months)]
        summer_mean_df = summer_mean_df.groupby(summer_mean_df.index.year).mean()
        summer_mean_df = summer_mean_df.where(summer_mean_df > 1, 1)

        # Load simulated Chla to meta_df
        lake_meta_df['Chla2020_sim_{}_{}'.format(ssp, idx)] = summer_mean_df.loc[2018:2022, lake_meta_df.index].mean(axis=0).values
        lake_meta_df.loc[:, 'Chla2020_{}_{}'.format(ssp, idx)] = lake_meta_df['Chla2020_sim_{}_{}'.format(ssp, idx)]
        summer_mean_df = summer_mean_df.where(summer_mean_df > 1, 1)
        for year in log_years:
            if year > 2020:
                lake_meta_df.loc[:, 'Chla{}_{}_{}'.format(year, ssp, idx)] = summer_mean_df.loc[year-2:year+2, lake_meta_df.index].mean(axis=0).values
                lake_meta_df.loc[:, '{}sub2020_{}_{}'.format(year, ssp, idx)] = lake_meta_df['Chla{}_{}_{}'.format(year, ssp, idx)].values - lake_meta_df['Chla2020_{}_{}'.format(ssp, idx)].values

    # Save meta df
    lake_meta_df['N_Id'] = [int(i[1:]) for i in lake_meta_df.index]
    lake_meta_df = lake_meta_df.sort_values('N_Id')
    lake_meta_df.to_csv('Data/Lake142Meta3.csv', encoding='utf_8_sig')

    # plt nutrient-chla response changes
    fig, axes = plt.subplots(len(ssp_ls), len(log_years), figsize=(3.5 * len(log_years), 3.5 * len(ssp_ls)), sharex='col', sharey='row')
    axes = axes.reshape(len(ssp_ls), len(log_years))
    for i_y, year in enumerate(log_years):
        for i_s, ssp in enumerate(ssp_ls):
            ax = axes[i_s, i_y]
            plt.sca(ax)
            col_name = 'Chla{}_{}_{}'.format(year, ssp, idx)
            print(col_name, 'chla>12', len(lake_meta_df[lake_meta_df[col_name] > 12]))
            for p_stratification, marker in enumerate(['o', '^']):
                plt.scatter(
                    lake_meta_df['Limit_P'][lake_meta_df['summer_stratification'] == p_stratification],
                    lake_meta_df[col_name][lake_meta_df['summer_stratification'] == p_stratification],
                    marker=marker,
                    s=20,
                    linewidths=lw,
                    facecolors=[region_colors[int(lake_meta_df.loc[lake_id, 'Region_ID'])] for lake_id in lake_meta_df[lake_meta_df['summer_stratification'] == p_stratification].index],
                    edgecolors='k',
                    alpha=1,
                    zorder=-1,
                )
            for region_id in [1, 0, 3, 2, 4]:
                region_lakes = lake_meta_df[lake_meta_df['Region_ID'] == region_id].index if region_id < 4 else lake_meta_df.index
                x, y = np.log10(lake_meta_df['Limit_P'][region_lakes].values), np.log10(lake_meta_df[col_name][region_lakes].values)
                slope, intercept, r_value, p_value, std_err = linregress(x, y)
                y_pred = x * slope + intercept
                if p_value < 1e-4:
                    p_str = ', $\it{p}$<0.0001'
                elif p_value < 1e-3:
                    p_str = ', $\it{p}$<0.001'
                elif p_value < 1e-2:
                    p_str = ', $\it{p}$<0.01'
                elif p_value < 5e-2:
                    p_str = ', $\it{p}$<0.05'
                elif p_value < 1e-1:
                    p_str = ', $\it{p}$<0.1'
                else:
                    p_str = ''
                plt.plot(10 ** x, 10 ** y_pred, c=region_colors[region_id], lw=lw * 2, label='{} (slope={:.2f}{})'.format(region_names[region_id], max(slope, 0), p_str), zorder=98)
                print(region_id, slope, std_err, r_value, p_value)
            plt.plot([10 ** 0.5, 10 ** 2.5], [12, 12], c='r', lw=lw * 2, ls=':', zorder=-2, alpha=1)
            plt.text(0.05, .03, '{} ({})'.format(year, 'SSP5-8.5' if ssp == '585' else 'SSP2-4.5'), fontsize=10, c='k', va='center', transform=ax.transAxes)
            plt.xscale('log')
            plt.yscale('log')
            plt.ylim(10 ** 0, 10 ** 2.5)
            plt.xlim(10 ** 0.5, 10 ** 2.5)
            ax.legend(ncol=1, fontsize=7, loc='upper left', frameon=False)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
            ax.tick_params(axis='both', which='major', width=0.5, length=3, labelsize=8)
            plt.grid(alpha=0.5, lw=0.4)
            if i_y == 0:
                plt.ylabel('Summer surface Chl$\it{a}$ ($μg$ $L^{-1}$)', fontsize=9)
            if i_s == 1:
                plt.xlabel('Spring limited nutrient ($μg$ $P$ $equivalent$ $L^{-1}$)', fontsize=9)
    plt.tight_layout()
    plt.savefig('Figs/MS/F2 response changes.png', dpi=300)
    # plt.show()


    """
    plt delta chla in different regions, lake types
    """
    lake_meta_df['Nutrient level'] = (lake_meta_df['Limit_P'].values > 30).astype(np.int32)
    lake_meta_df['Nutrient level'] = (lake_meta_df['Limit_P'].values > 30).astype(np.int32)
    lake_meta_df['Ice cover'] = 0
    lake_meta_df['Ice cover'][(lake_meta_df['year_ice_cover_day'] > 0)] = 1
    lake_meta_df['Region'] = -1
    lake_meta_df['Region'][lake_meta_df['Region_ID'] == 1] = 0
    lake_meta_df['Region'][lake_meta_df['Region_ID'] == 0] = 1
    lake_meta_df['Region'][lake_meta_df['Region_ID'] == 3] = 2
    lake_meta_df['Region'][lake_meta_df['Region_ID'] == 2] = 3
    split_dic = {
        'Region': ['YPL', 'EPL', 'IXL', 'NPL'],
        'Nutrient level': ['Nutrient-poor', 'Nutrient-rich'],
        'Stratification': ['Unstratified', 'Stratified'],
        'Ice cover': ['Ice-free', 'Ice-covered'],
    }
    c_y = '2100sub2020_585_{}'.format(idx)

    # box plots
    fig, axes = plt.subplots(1, len(split_dic), figsize=(8, 2.2), sharey='row', gridspec_kw={'width_ratios': [1., 1, 1, 1]})
    for i, (c_split, x_labels) in enumerate(split_dic.items()):
        ax = axes[i]
        plt.sca(ax)
        numbers = []
        values_ls = []
        for i_x in range(len(x_labels)):
            x_sub_df = lake_meta_df[lake_meta_df[c_split] == i_x]
            for i_s, title in enumerate(['Unstratified lakes', 'Stratified lakes']):
                s_sub_df = x_sub_df[x_sub_df['summer_stratification'] == i_s]
                values = s_sub_df[c_y].values
                plt.scatter(
                    np.random.normal(i_x, 0.05, len(values)),
                    values,
                    marker='^' if i_s else 'o',
                    s=15,
                    linewidths=lw,
                    facecolors=[region_colors[int(s_sub_df.loc[lake_id, 'Region_ID'])] for lake_id in s_sub_df.index],
                    edgecolors='k',
                    alpha=0.8,
                    zorder=100,
                )
            plt.boxplot(
                x_sub_df[c_y].values,
                positions=[i_x],
                medianprops={'color': 'k', 'linewidth': lw},
                flierprops={'markeredgewidth': lw, 'markersize': 2},
                widths=lw,
                boxprops={'linewidth': lw, 'color': 'k'},
                whiskerprops={'linewidth': lw, 'color': 'k'},
                capprops={'linewidth': lw, 'color': 'k'},
                manage_ticks=False,
                showfliers=False,
                zorder=99,
            )
            numbers.append(len(x_sub_df))
            values_ls.append(x_sub_df[c_y].values)
        if len(values_ls) == 2:
            t, p_value = stats.ttest_ind(*values_ls, equal_var=False)
            print(c_split, p_value)
            if p_value < 1e-4:
                p_str = '<0.0001'
            elif p_value < 1e-3:
                p_str = '<0.001'
            elif p_value < 5e-3:
                p_str = '<0.005'
            elif p_value < 1e-2:
                p_str = '<0.01'
            elif p_value < 5e-2:
                p_str = '<0.05'
            else:
                p_str = '={:.2f}'.format(p_value)
            if len(p_str) > -1:
                plt.text(0.5, .95, "Welch's $\it{t}$-test ($\it{p}$" + p_str + ')', fontsize=8, c='k', va='center', ha='center', transform=ax.transAxes)
        plt.xticks(range(len(x_labels)), ['{}\n(n={})'.format(x_label, n) for x_label, n in zip(x_labels, numbers)], fontsize=9)
        if i == 0:
            plt.ylabel('Δ Summer surface Chl$\it{a}$ ($μg$ $L^{-1}$)', fontsize=8)
        for spine in ax.spines.values():
            spine.set_linewidth(lw)
        ax.tick_params(axis='both', which='major', width=lw, length=3, labelsize=8)
        plt.grid(alpha=0.4, lw=0.3)
        plt.ylim(0, 30)
        plt.yticks([0, 10, 20, 30])
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)
    plt.savefig('Figs/MS/F2 delta chla in different lakes.png', dpi=300)
    # plt.show()


def fig3_nutrient_thresholds_future_changes():
    lw = 0.5
    lake_meta_df = pd.read_csv('Data/Lake142Meta3.csv', index_col=0, header=0)
    lake_meta_df = lake_meta_df[lake_meta_df['Use'] == 1]
    lake_meta_df['Stratification'] = lake_meta_df['summer_stratification'].values
    lake_meta_df['Nutrient level'] = (lake_meta_df['Limit_P'].values > 30).astype(np.int32)
    lake_meta_df['Ice cover'] = 0
    lake_meta_df['Ice cover'][(lake_meta_df['year_ice_day_2020_585'] > 0)] = 1
    lake_meta_df['Region'] = -1
    lake_meta_df['Region'][lake_meta_df['Region_ID'] == 1] = 0
    lake_meta_df['Region'][lake_meta_df['Region_ID'] == 0] = 1
    lake_meta_df['Region'][lake_meta_df['Region_ID'] == 3] = 2
    lake_meta_df['Region'][lake_meta_df['Region_ID'] == 2] = 3
    split_dic = {
        'Region': ['YPL', 'EPL', 'IXL', 'NPL'],
    }
    year_target_dic = {
        2020: 'g',
        2060245: 'orange',
        2060585: 'tomato',
        2100585: 'darkred'
    }

    fig, axes = plt.subplots(len(split_dic), len(year_target_dic), figsize=(8, 7), sharey='row')
    for i, (c_split, x_labels) in enumerate(split_dic.items()):
        for j, c_y in enumerate(year_target_dic.keys()):
            ax = axes[i, j]
            plt.sca(ax)
            numbers = []
            values_ls = []
            for i_x in range(len(x_labels)):
                x_sub_df = lake_meta_df[lake_meta_df[c_split] == i_x]
                for i_s, title in enumerate(['Unstratified lakes', 'Stratified lakes']):
                    s_sub_df = x_sub_df[x_sub_df['summer_stratification'] == i_s]
                    values = s_sub_df[c_y].values
                    plt.scatter(
                        np.random.normal(i_x, 0.05, len(values)),
                        values,
                        marker='^' if i_s else 'o',
                        s=15,
                        linewidths=lw,
                        facecolors=year_target_dic[int(c_y[13:17])] if i > 0 else [region_colors[int(s_sub_df.loc[lake_id, 'Region_ID'])] for lake_id in s_sub_df.index],
                        edgecolors='k',
                        alpha=0.8,
                        zorder=100,
                    )
                region_values = x_sub_df[c_y].values
                print(c_y, c_split, i_x, region_values.mean(), np.median(region_values))
                plt.boxplot(
                    region_values,
                    positions=[i_x],
                    medianprops={'color': 'k', 'linewidth': lw},
                    meanprops={'color': 'r', 'linewidth': lw},
                    flierprops={'markeredgewidth': lw, 'markersize': 2},
                    widths=lw,
                    boxprops={'linewidth': lw, 'color': 'k'},
                    whiskerprops={'linewidth': lw, 'color': 'k'},
                    capprops={'linewidth': lw, 'color': 'k'},
                    manage_ticks=False,
                    showfliers=False,
                    zorder=99,
                )
                numbers.append(len(x_sub_df))
                values_ls.append(x_sub_df[c_y].values)
            s = '{}\n({})'.format(c_y[:4], 'SSP5-8.5' if c_y[-3:] == 585 else 'SSP2-4.5') if c_y[:4] != '2020' else 'Current'
            s = s.replace('5-8.5', '2-4.5').replace('2040', '2060') if '2040' in s else s
            plt.text(0.98, .075 if len(values_ls) > 2 else 0.05, s if len(values_ls) > 2 else s.replace('\n', ''), fontsize=8, c='k', va='center', ha='right', transform=ax.transAxes)
            plt.xticks(range(len(x_labels)), ['{}'.format(x_label) for x_label, n in zip(x_labels, numbers)], fontsize=8)
            if j == 0:
                plt.ylabel('Nutrient threshold\n($μg$ $P$ $equivalent$ $L^{-1}$)', fontsize=8)
            for spine in ax.spines.values():
                spine.set_linewidth(lw)
            ax.tick_params(axis='both', which='major', width=lw, length=4, labelsize=8)
            ax.tick_params(axis='both', which='minor', width=lw, length=2, labelsize=8)
            plt.grid(alpha=0.4, lw=0.3, which='both')
            plt.yscale('log')
            if len(values_ls) > 2:
                plt.ylim(9, 110)
            else:
                plt.ylim(7, 150)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)
    plt.savefig('Figs/MS/F3 nutrient thresholds in different warms.png', dpi=300)
    # plt.show()


if __name__ == '__main__':
    fig1_limited_nutrient_and_chla_responses()
    fig2_responses_future_changes()
    fig3_nutrient_thresholds_future_changes()
