import os
import pandas as pd
from SALib.sample import latin
import gc
from Utils.mutil_processing import mutil_process
from model import LakePhytoModel1D
from model_paras import used_lake_ids


class ModelOptimizer:
    def __init__(self, lake_ids, date_range, future_scenario):
        self.lake_ids = lake_ids
        self.date_range = date_range
        self.future_scenario = future_scenario
        self.paras_dic = {
            'scale_P_growth_rate_t0_dic': [0.01, 1],    # lamda 1
            'scale_P_morality_rate_dic': [0.8, 1.2],    # lamda 2
        }

    def optimize_latin(self, n_sample=50, n_process=8, save_dir='F:/lgf', random_seed=10010):
        _ = None if os.path.exists(save_dir) else os.mkdir(save_dir)

        problem = {
            'num_vars': len(self.paras_dic),
            'names': self.paras_dic.keys(),
            'bounds': list(self.paras_dic.values()),
        }
        samples = latin.sample(problem=problem, N=n_sample, seed=random_seed)
        latin_df = pd.DataFrame(data=samples, columns=problem['names'])

        multi_paras_dic_ls = []
        for i in range(n_sample):
            path = os.path.join(save_dir, 'Model1D_Res{} {}{}{}.csv'.format('', 'Chla', 0, '-{}'.format(i)))
            if os.path.exists(path):
                print('Have run {}'.format(i))
            else:
                multi_paras_dic_ls.append(
                    {
                        'use_paras_dic': dict(zip(latin_df.columns, latin_df.loc[i].values)),
                        'idx': i,
                        'save_dir': save_dir,
                        'lake_ids': self.lake_ids,
                        'date_range': self.date_range,
                        'future_scenario': self.future_scenario,
                    }
                )
        mutil_process(
            function=run_model_multi,
            kwargs_ls=multi_paras_dic_ls,
            n_process=n_process,
        )
        latin_df.to_csv(os.path.join(save_dir, '00-Optimize Samples.csv'))


def run_model_multi(kwargs):
    use_paras_dic = kwargs['use_paras_dic']
    idx = kwargs['idx']
    save_dir = kwargs['save_dir']
    lake_ids = kwargs['lake_ids']
    date_range = kwargs['date_range']
    future_scenario = 'ssp585' if 'future_scenario' not in kwargs.keys() else kwargs['future_scenario']
    ini_tp = None if 'ini_tp' not in kwargs.keys() else kwargs['ini_tp']
    dt = 5e-2 if 'dt' not in kwargs.keys() else kwargs['dt']
    nz = 10 if 'nz' not in kwargs.keys() else kwargs['nz']
    save_nz = 1 if 'save_nz' not in kwargs.keys() else kwargs['save_nz']

    model1d = LakePhytoModel1D(lake_ids=lake_ids, date_range=date_range, scenario=future_scenario, dt=dt, nz=nz)
    model1d.adjust_paras(use_paras_dic)
    model1d.reset_lake_nutrient(ini_tp)
    model1d.continuous_solve(solver='RK45')
    model1d.save_res(save_dir=save_dir, suffix='-{}'.format(idx), save_nz=save_nz)
    gc.collect()


if __name__ == '__main__':
    mo = ModelOptimizer(
        lake_ids=used_lake_ids,
        date_range=pd.date_range('2000-1-1', '2025-1-1', freq='D'),
        future_scenario='ssp245'
    )
    mo.optimize_latin(n_sample=100, n_process=10, save_dir='Calibration')
