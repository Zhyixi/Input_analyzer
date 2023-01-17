# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from Stats.Stats import Fit_Density
from scipy import stats
import matplotlib.pyplot as plt

if __name__ == "__main__":
    #%%前期設定
    #建立物件與常用變數
    fd = Fit_Density()
    file_path = "直棒各站生產資料202001_202104_0617.xlsx"  # 檔案路徑
    sheet_name = "401直棒退火爐"  # # # 活頁簿名稱'401直棒退火爐'403解捲站
    if_filter=True # 是否濾雜訊
    fit_col = "WIP時間"  # 分析欄位
    group_conditioin=["下站別"] #建立分組條件
    group_conditioin_str=""
    for i,v in enumerate(group_conditioin):
        if i ==0:
            group_conditioin_str+=v
        else:
            group_conditioin_str += "_" + v
    #%%讀取檔案並分組
    df=pd.read_excel(file_path,sheet_name=sheet_name)
    df_group=dict(list(df.groupby(group_conditioin)))
    #%%分析
    result={group_conditioin_str:[],"分析參數":[],"樣本數":[],"樣本數(filtered)":[],"平均數":[],"平均數(filtered)":[],"標準差":[],"標準差(filtered)":[],"建議分布":[],"建議分布(filtered)":[],"建議參數":[],"建議參數(filtered)":[],"統計量":[],"統計量(filtered)":[],"p-value":[],"p-value(filtered)":[],"均方誤差":[],"均方誤差(filtered)":[]} #儲存分析結果
    for _group,_data in df_group.items():
        result[group_conditioin_str].append(_group)
        result["分析參數"].append(fit_col)
        #%%某些組別不適合做FIT
        if _group=='-'or _data.shape[0]<10:
            result["樣本數"].append(_data.shape[0])
            result["樣本數(filtered)"].append(_data.shape[0])
            result["建議分布"].append("-")
            result["建議參數"].append("-")
            result["統計量"].append("-")
            result["p-value"].append("-")
            result["均方誤差"].append("-")
            result["平均數"].append("-")
            result["標準差"].append("-")
            result["建議分布(filtered)"].append("-")
            result["建議參數(filtered)"].append("-")
            result["統計量(filtered)"].append("-")
            result["p-value(filtered)"].append("-")
            result["均方誤差(filtered)"].append("-")
            result["平均數(filtered)"].append("-")
            result["標準差(filtered)"].append("-")
            continue
        else:
            data=_data[[fit_col]]
            data_filter=data[(np.abs(stats.zscore(data)) < 1).all(axis=1)]
            data_filter = data_filter.values.flatten()
            data_raw = data.values.flatten()
            #%%畫圖
            #subplot1
            params_raw = fd.Fit_distributions(data=data_raw)
            #TEST
            # for i in fd.continuous_dist_names:
            #     a = fd.get_parameters_name(i)
            #     print(i, a)
            #     print("==============================")
            # exit()
            #TEST
            best_dist_raw, best_statistic_raw, best_p_raw, best_dist_params_raw, dist_results_raw = fd.get_best_distribution(data=data_raw,
                                                                                                         params=params_raw)
            title_raw = '組別:{}_分析欄位:{}_最佳分布:{}_樣本數:{}'.format(_group, fit_col, best_dist_raw, data_raw.shape[0])
            f_raw = Fitter(data_raw, distributions=[best_dist_raw])
            f_raw.fit()
            plt.subplot(211)
            f_raw.hist()
            f_raw.plot_pdf(names=best_dist_raw)
            plt.title(title_raw)
            # %%放入分析數據
            result["樣本數"].append(data_raw.shape[0])
            result["平均數"].append(np.std(data_raw))
            result["標準差"].append(np.mean(data_raw))
            result["建議分布"].append(best_dist_raw)
            # %%建立建議參數
            d = {}
            parameter_names = fd.get_parameters_name(best_dist_raw)
            for name_, paras_ in zip(parameter_names, best_dist_params_raw):
                d[name_] = paras_
            # %%
            result["建議參數"].append(d)
            result["統計量"].append(best_statistic_raw)
            result["p-value"].append(best_p_raw)
            result["均方誤差"].append(dist_results_raw[best_dist_raw]['mse'])
            # subplot2
            params_filter = fd.Fit_distributions(data=data_filter)
            best_dist_filter, best_statistic_filter, best_p_filter, best_dist_params_filter, dist_results_filter = fd.get_best_distribution(data=data_filter,
                                                                                                         params=params_filter)
            title_filter = '組別:{}_分析欄位:{}_最佳分布:{}_樣本數:{}'.format(_group, fit_col, best_dist_filter, data_filter.shape[0])
            f_filter = Fitter(data_filter, distributions=[best_dist_filter])
            f_filter.fit()
            plt.subplot(212)
            f_filter.hist()
            f_filter.plot_pdf(names=best_dist_filter)
            plt.title(title_filter)
            # %%放入分析數據
            result["樣本數(filtered)"].append(data_filter.shape[0])
            result["平均數(filtered)"].append(np.std(data_filter))
            result["標準差(filtered)"].append(np.mean(data_filter))
            result["建議分布(filtered)"].append(best_dist_filter)
            # %%建立建議參數
            d = {}
            parameter_names = fd.get_parameters_name(best_dist_filter)
            for name_, paras_ in zip(parameter_names, best_dist_params_filter):
                d[name_] = paras_
            # %%
            result["建議參數(filtered)"].append(d)
            result["統計量(filtered)"].append(best_statistic_filter)
            result["p-value(filtered)"].append(best_p_filter)
            result["均方誤差(filtered)"].append(dist_results_filter[best_dist_filter]['mse'])
            #%%%%%%%%%%%%%%%%%%%%
            path = "{}.png".format(title_raw.replace(':', "_"))
            plt.grid(True)
            # plt.tight_layout()
            plt.subplots_adjust(left=0.125,
                                bottom=0.1,
                                right=0.9,
                                top=0.9,
                                wspace=0.2,
                                hspace=0.35)
            plt.xlabel('{} '.format(fit_col), size=12)
            plt.ylabel('{}'.format("Probability"), size=12)
            plt.savefig(path)
            plt.close()
    result_df=pd.DataFrame(data=result)
    result_df.to_excel("{}_density_suggestion.xlsx".format(sheet_name))