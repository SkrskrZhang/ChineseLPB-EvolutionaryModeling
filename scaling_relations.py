import numpy as np


"""
All input x is in log10, return y also in log10
Used units:
    volume: log µm3
    wights: log 1e-12 g
    concen: µg/L or mg/m3
"""


class ScalingLaws:
    def __init__(self, PC):
        self.PC = PC


    def cell_volume_2_cell_mass_C(self, x, per_x=False):
        """
        2024 Wickman
            log(cell mass [1e-12 g C / cell]) = 0.86 * log(cell volume [(µm3]) - 0.5872
        if per_x:
            log(cell mass [1e-12 g C / µm3]) = (slope - 1) * log(cell volume [(µm3]) + intercept
        """
        s = self.PC.scaling_slope_dic['cell_volume_2_cell_mass_C']
        i = self.PC.scaling_intercept_dic['cell_volume_2_cell_mass_C']
        s = s - 1 if per_x else s
        return x * s + i


    def cell_volume_2_uptake_max_N(self, x):
        """
        2012 Edwards
            log(max N uptake [1e-12 g N cell-1]) = slope * log(cell volume [(µm3]) + intercept
        """
        s = self.PC.scaling_slope_dic['cell_volume_2_uptake_max_N']
        i = self.PC.scaling_intercept_dic['cell_volume_2_uptake_max_N']
        return x * s + i


    def cell_volume_2_uptake_max_P(self, x):
        """
        2012 Edwards
            log(max P uptake [1e-12 g P cell-1]) = slope * log(cell volume [(µm3]) + intercept
        """
        s = self.PC.scaling_slope_dic['cell_volume_2_uptake_max_P']
        i = self.PC.scaling_intercept_dic['cell_volume_2_uptake_max_P']
        return x * s + i


    def cell_volume_2_KN(self, x, c='fresh'):
        """
        2012 Edwards
            log(KN [µg N L-1]) = slope * log(cell volume [(µm3]) + intercept
        """
        s = self.PC.scaling_slope_dic['cell_volume_2_KN_{}'.format(c)]
        i = self.PC.scaling_intercept_dic['cell_volume_2_KN_{}'.format(c)]
        return x * s + i


    def cell_volume_2_KP(self, x):
        """
        2012 Edwards
            log(KP [µg P L-1]) = slope * log(cell volume [(µm3]) + intercept
        """
        s = self.PC.scaling_slope_dic['cell_volume_2_KP']
        i = self.PC.scaling_intercept_dic['cell_volume_2_KP']
        return x * s + i


    def cell_volume_2_QNmin(self, x, per_x=False):
        """
        2012 Edwards
            log(QNmin [1e-12 g N cell-1]) = slope * log(cell volume [(µm3]) + intercept
        if per_x:
            log(QNmin [1e-12 g N / µm3]) = (slope - 1) * log(cell volume [(µm3]) + intercept
        """
        s = self.PC.scaling_slope_dic['cell_volume_2_QNmin']
        s = s - 1 if per_x else s
        i = self.PC.scaling_intercept_dic['cell_volume_2_QNmin']
        return x * s + i


    def cell_volume_2_QPmin(self, x, per_x=False):
        """
        2012 Edwards
            log(QPmin [1e-12 g P cell-1]) = slope * log(cell volume [(µm3]) + intercept
        """
        s = self.PC.scaling_slope_dic['cell_volume_2_QPmin']
        s = s - 1 if per_x else s
        i = self.PC.scaling_intercept_dic['cell_volume_2_QPmin']
        return x * s + i


    def cell_volume_2_dry_mass(self, x, r=False):
        """
        Ref. Kremer 2017
        dry mass [1e-12 g] = volume ^ 0.99 [µm3] * 0.47 * 1e-12
        """
        if r:
            return (x - np.log10(0.47 * 1e-12)) / 0.99
        else:
            return 0.99 * x + np.log10(0.47 * 1e-12)


    def cell_volume_2_sink_rate(self, x):
        """
        2012 2019 Durante (log m/d)
        """
        s = self.PC.scaling_slope_dic['cell_volume_2_sink_rate']
        i = self.PC.scaling_intercept_dic['cell_volume_2_sink_rate']
        return x * s + i


    def irradiance_2_chla_c_ratio(self, x, x_min=10, p=0.05):
        """
        Ref 1997.Geider 1998.Zon.
        x is irradiance (μmol/m2/s), need log
        log (Chla:C) = -0.44050665 * log(x) -0.96376
        """
        x_new = np.clip(x, x_min, None)
        s = self.PC.scaling_slope_dic['irradiance_2_chla_c_ratio']
        i = self.PC.scaling_intercept_dic['irradiance_2_chla_c_ratio']
        Chla2C = np.log10(x_new) * s + i
        prediction_interval = None
        return Chla2C, prediction_interval


