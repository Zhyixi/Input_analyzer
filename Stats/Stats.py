import logging
import sys
import threading
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab
import scipy.stats
import tensorflow as tf
import tensorflow_probability as tfp
from numpy import random
from scipy import stats
from scipy.optimize import curve_fit
from scipy.stats import entropy as kl_div
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

# matplotlib.use('Agg')
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 09:14:34 2021
@author: stats
"""
np.random.seed(0)
tf.random.set_seed(0)
plt.style.use("seaborn-whitegrid")
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

logging.getLogger(__name__) #內建變數  返回執行程式名稱，若為本程式則顯示__main__
__all__ = ['get_common_distributions', 'get_distributions', 'Fit_Density']

def get_distributions():
    # 若fit 方法存在於scipy.stats的屬性中
    # 則儲存起來
    distributions = []
    for this in dir(scipy.stats):
        if "fit" in eval("dir(scipy.stats." + this + ")"):
            distributions.append(this)
    return distributions

def get_common_distributions():
    # 取出常用的分布
    distributions = get_distributions()
    # to avoid error due to changes in scipy
    common = ['cauchy', 'chi2', 'expon', 'exponpow', 'gamma',
         'lognorm', 'norm', 'powerlaw', 'rayleigh', 'uniform']
    common = [x for x in common if x in distributions]
    return common

class Fit_Density(object):
    # """擬合數據到已知的分布Fit a data sample to known distributions
    #
    # 通常用來找出可能產生數據集的潛在分佈的一種簡單方法是將數據的直方圖與已知分佈（例如正態）的 PDF（概率分佈函數）進行比較。
    # A naive approach often performed to figure out the undelying distribution that
    # could have generated a data set, is to compare the histogram of the data with
    # a PDF (probability distribution function) of a known distribution (e.g., normal).
    #然而，分佈的參數是未知的，並且有很多分佈。因此，一種將許多分佈擬合到數據的自動方法將是很有用的，這就是這裡實現的。
    # Yet, the parameters of the distribution are not known and there are lots of
    # distributions. Therefore, an automatic way to fit many distributions to the data
    # would be useful, which is what is implemented here.
    #給定一個數據樣本，我們使用 SciPy 的“擬合”方法來提取最適合數據的分佈參數。
    # 我們對所有可用的發行版重複此操作。最後，我們提供了一個總結，以便人們可以看到這些分佈的擬合質量
    # Given a data sample, we use the `fit` method of SciPy to extract the parameters
    # of that distribution that best fit the data. We repeat this for all available distributions.
    # Finally, we provide a summary so that one can see the quality of the fit for those distributions
    #這裡是一個我們從GAMMA分布產生的樣本
    # Here is an example where we generate a sample from a gamma distribution.
    # First, we create a data sample following a Gamma distribution
    # from scipy import stats
    # data = stats.gamma.rvs(2, loc=1.5, scale=2, size=20000)
    # We then create the Fitter object
    # import fitter
    # f = fitter.Fitter(data)
    # just a trick to use only 10 distributions instead of 80 to speed up the fitting
    # f.distributions = f.distributions[0:10] + ['gamma']
    # fit and plot
    # f.fit()
    # f.summary()
    #             sumsquare_error
    #     gamma          0.000095
    #     beta           0.000179
    #     chi            0.012247
    #     cauchy         0.044443
    #     anglit         0.051672
    #     [5 rows x 1 columns]
    #
    # Once the data has been fitted, the :meth:`summary` metod returns a sorted dataframe where the
    #
    # Looping over the 80 distributions in SciPy could takes some times so you can overwrite the
    # :attr:`distributions` with a subset if you want. In order to reload all distributions,
    # call :meth:`load_all_distributions`.
    #
    # Some distributions do not converge when fitting. There is a timeout of 10 seconds after which
    # the fitting procedure is cancelled. You can change this :attr:`timeout` attribute if needed.
    #
    # If the histogram of the data has outlier of very long tails, you may want to increase the
    # :attr:`bins` binning or to ignore data below or above a certain range. This can be achieved
    # by setting the :attr:`xmin` and :attr:`xmax` attributes. If you set xmin, you can come back to
    # the original data by setting xmin to None (same for xmax) or just recreate an instance.
    # """

    def __init__(self, data, xmin=None, xmax=None, bins=100,
                 distributions=None, timeout=300,
                 density=True,sigma=0.05):
        """.. rubric:: Constructor

        :param list data: a numpy array or a list
        :param float xmin: if None, use the data minimum value, otherwise histogram and
            fits will be cut
        :param float xmax: if None, use the data maximum value, otherwise histogram and
            fits will be cut
        :param int bins: numbers of bins to be used for the cumulative histogram. This has
            an impact on the quality of the fit.
        :param list distributions: give a list of distributions to look at. If none, use
            all scipy distributions that have a fit method. If you want to use
            only one distribution and know its name, you may provide a string (e.g.
            'gamma'). Finally, you may set to 'common' to  include only common
            distributions, which are: cauchy, chi2, expon, exponpow, gamma,
                 lognorm, norm, powerlaw, irayleigh, uniform.
        :param timeout: max time for a given distribution. If timeout is
            reached, the distribution is skipped.

        .. versionchanged:: 1.2.1 remove verbose argument, replacedb by logging module.
        .. versionchanged:: 1.0.8 increase timeout from 10 to 30 seconds.
        """
        self.sigma=sigma
        self.num_epoch=1000
        self.timeout = timeout
        # 使用者輸入
        self._data = None
        # 因為需與密度函術進行比較，所以必須為True
        self._density = True
        #: list of distributions to test
        self.distributions = distributions
        if self.distributions == None:
            self._load_continuous_distributions()
            # self._load_all_distributions()
        elif self.distributions == "common":
            self.distributions = get_common_distributions()
        elif isinstance(distributions, str):
            self.distributions = [distributions]

        self.bins = bins
        if (data.dtype == object):
            self.scaler=preprocessing.LabelEncoder()
            self._alldata=self.scaler.fit_transform(data)
        else:
            self._alldata = np.array(data)

        if xmin == None:
            self._xmin = self._alldata.min()
        else:
            self._xmin = xmin
        if xmax == None:
            self._xmax = self._alldata.max()
        else:
            self._xmax = xmax

        self._trim_data()
        self._update_data_pdf() #建立pdf的x,y
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
        # Other attributes
        self._init()

    def _init(self):
        self.fitted_param = {}
        self.fitted_pdf = {}
        self._fitted_errors = {}
        self._aic = {}
        self._bic = {}
        self._kldiv = {}

    def _update_data_pdf(self):
        # 直方圖使用 N+1 個值返回 X。 因此，我們將 X 輸出重新排列為僅 N
        # histogram retuns X with N+1 values. So, we rearrange the X output into only N
        self.y, self.x = np.histogram(
            self._data, bins=self.bins, density=self._density)
        # 取兩點中間值
        self.x= [(this + self.x[i + 1]) / 2. for i,this in enumerate(self.x[0:-1])]
        self.step=np.mean(np.diff(self.x))


    def _trim_data(self):
        #篩選指定範圍大小的數據
        self._data = self._alldata[np.logical_and(self._alldata >= self._xmin, self._alldata <= self._xmax)]

    def _get_xmin(self):
        return self._xmin

    def _set_xmin(self, value):
        if value == None:
            value = self._alldata.min()
        elif value < self._alldata.min():
            value = self._alldata.min()
        self._xmin = value
        self._trim_data()
        self._update_data_pdf()
    xmin = property(_get_xmin, _set_xmin,
                    doc="consider only data above xmin. reset if None")

    def _get_xmax(self):
        return self._xmax

    def _set_xmax(self, value):
        if value == None:
            value = self._alldata.max()
        elif value > self._alldata.max():
            value = self._alldata.max()
        self._xmax = value
        self._trim_data()
        self._update_data_pdf()
    xmax = property(_get_xmax, _set_xmax,
                    doc="consider only data below xmax. reset if None ")

    def _load_all_distributions(self):
        """Replace the :attr:`distributions` attribute with all scipy distributions"""
        self.distributions = get_distributions()

    def _load_continuous_distributions(self):
        self.distributions = ['beta', 'erlang', 'expon', 'gamma', 'weibull_min', 'johnsonsu', 'lognorm',
                                          'norm', 'triang', 'uniform','poisson'] #, 't', 'alpha'

    def hist(self):
        _ = pylab.hist(self._data, bins=self.bins, density=self._density)
        pylab.grid(True)

    def fit(self, amp=1, progress=False):
        r"""Loop over distributions and find best parameter to fit the data for each

        When a distribution is fitted onto the data, we populate a set of
        dataframes:

            - :attr:`df_errors`  :sum of the square errors between the data and the fitted
              distribution i.e., :math:`\sum_i \left( Y_i - pdf(X_i) \right)^2`
            - :attr:`fitted_param` : the parameters that best fit the data
            - :attr:`fitted_pdf` : the PDF generated with the parameters that best fit the data

        Indices of the dataframes contains the name of the distribution.

        """
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        from easydev import Progress
        N = len(self.distributions)
        pb = Progress(N)
        for i, distribution in enumerate(self.distributions):
            if distribution == 'poisson' and self._data.dtype!=np.float:
                param = self.__fit_poisson(data=self._data)
                pdf_fitted = stats.poisson.pmf(self.x, *param)

            try:
                # need a subprocess to check time it takes. If too long, skip it
                dist = eval("scipy.stats." + distribution)
                # TODO here, dist.fit may take a while or just hang forever
                # with some distributions. So, I thought to use signal module
                # to catch the error when signal takes too long. It did not work
                # presumably because another try/exception is inside the
                # fit function, so I used threading with a recipe from stackoverflow
                # See timed_run function above

                if distribution != 'poisson':
                    param = self._timed_run(dist.fit, distribution, args=self._data)
                    # with signal, does not work. maybe because another expection is caught
                    # hoping the order returned by fit is the same as in pdf
                    pdf_fitted = dist.pdf(self.x, *param)

                params_name = self.__get_parameters_and_name(parameter=param,
                                                             name=self.__get_parameters_name(distribution))

                self.fitted_param[distribution] = params_name
                self.fitted_pdf[distribution] = pdf_fitted

                # calculate error
                sq_error = np.sum((self.fitted_pdf[distribution] - self.y) ** 2)
                if distribution == 'poisson':
                    logLik = np.sum(dist.logpmf(self.x, *param))
                else:
                    # calcualte information criteria
                    logLik = np.sum(dist.logpdf(self.x, *param))
                k = len(param[:])
                n = len(self._data)
                aic = 2 * k - 2 * logLik
                bic = n * np.log(sq_error / n) + k * np.log(n)

                # calcualte kullback leibler divergence
                kullback_leibler = kl_div(
                    self.fitted_pdf[distribution], self.y)

                logging.info("Fitted {} distribution with error={})".format(
                    distribution, sq_error))

                # compute some errors now
                self._fitted_errors[distribution] = sq_error
                self._aic[distribution] = aic
                self._bic[distribution] = bic
                self._kldiv[distribution] = kullback_leibler
            except Exception as err: #pragma: no cover
                logging.warning("SKIPPED {} distribution (taking more than {} seconds)".format(distribution,
                                                                                         self.timeout))
                # if we cannot compute the error, set it to large values
                self._fitted_errors[distribution] = np.inf
                self._aic[distribution] = np.inf
                self._bic[distribution] = np.inf
                self._kldiv[distribution] = np.inf
            if progress:
                pb.animate(i+1)
        self.df_errors = pd.DataFrame({'sumsquare_error': self._fitted_errors,
                                       'aic': self._aic,
                                       'bic': self._bic,
                                       'kl_div': self._kldiv})

    def plot_pdf(self, names=None, Nbest=5, lw=2, method="sumsquare_error"):
        """Plots Probability density functions of the distributions

        :param str,list names: names can be a single distribution name, or a list
            of distribution names, or kept as None, in which case, the first Nbest
            distribution will be taken (default to best 5)


        """
        assert Nbest > 0 # 限制不能小於0
        if Nbest > len(self.distributions):
            Nbest = len(self.distributions)

        if isinstance(names, list): #判斷names是否為list物件
            for name in names:
                pylab.plot(self.x, self.fitted_pdf[name], lw=lw, label=name,c='r')
        elif names:#表示只是單一輸入
            pylab.plot(self.x, self.fitted_pdf[names], lw=lw, label=names,c='r')
        else:
            try:
                names = self.df_errors.sort_values(by=method).index[0:Nbest]
            except Exception:
                names = self.df_errors.sort(method).index[0:Nbest]

            for name in names:
                if name in self.fitted_pdf.keys():
                    pylab.plot(
                        self.x, self.fitted_pdf[name], lw=lw, label=name)
                else: #pragma: no cover
                    raise ValueError("%s was not fitted. no parameters available" % name)
                    # logger.warning("%s was not fitted. no parameters available" % name)
        pylab.grid(True)
        pylab.legend()

    def get_best(self, method='sumsquare_error',N=1):
        """Return best fitted distribution and its parameters
        a dictionary with one key (the distribution name) and its parameters
        """
        # self.df should be sorted, so then us take the first one as the best
        score=pd.DataFrame(self.get_test_score()).T.sort_values(["p-value"],ascending=False)
        if score[score["p-value"] > self.sigma].shape[0]==0:
            pass
        else:
            score = score[score["p-value"] > self.sigma]
        loss=self.df_errors.sort_values(method)
        res=pd.merge(score, loss, left_index=True, right_index=True)
        # p<0.05 拒絕HO；表示不為此分布
        # name = self.df_errors.sort_values(method).iloc[0].name
        names = list(res.sort_values([method,'p-value']).iloc[0:N].index)
        res={}
        for name_ in names:
            params = self.fitted_param[name_]
            res[name_]= params
        return res

    def summary(self, Nbest=20, lw=2, plot=True, method="sumsquare_error"):
        """
            Plots the distribution of the data and Nbest distribution
        """
        if plot:
            pylab.clf()
            self.hist()
            self.plot_pdf(Nbest=Nbest, lw=lw, method=method)
            pylab.grid(True)

        Nbest = min(Nbest, len(self.distributions))
        try:
            names = self.df_errors.sort_values(
                by=method).index[0:Nbest]
        except: #pragma: no cover
            names = self.df_errors.sort(method).index[0:Nbest]
        return self.df_errors.loc[names]

    def __get_parameters_and_name(self,parameter,name):
        res={}
        for pra_name,para in zip(name,parameter):
            res[pra_name]=para
        return res

    def _timed_run(self, func, distribution, args=(), kwargs={},  default=None):
        """This function will spawn a thread and run the given function
        using the args, kwargs and return the given default value if the
        timeout is exceeded.

        http://stackoverflow.com/questions/492519/timeout-on-a-python-function-call
        """
        class InterruptableThread(threading.Thread):
            def __init__(self):
                threading.Thread.__init__(self)
                self.result = default
                self.exc_info = (None, None, None)

            def run(self):
                try:
                    self.result = func(args, **kwargs)
                except Exception as err: #pragma: no cover
                    self.exc_info = sys.exc_info()

            def suicide(self): # pragma: no cover
                raise RuntimeError('Stop has been called')

        it = InterruptableThread()
        it.start()
        started_at = datetime.now()
        it.join(self.timeout)
        ended_at = datetime.now()
        diff = ended_at - started_at

        if it.exc_info[0] is not None:  #pragma: no cover ;  if there were any exceptions
            a, b, c = it.exc_info
            raise Exception(a, b, c)  # communicate that to caller

        if it.is_alive(): #pragma: no cover
            it.suicide()
            raise RuntimeError
        else:
            return it.result

    def __get_parameters_name(self,distribution):
        """List parameters for scipy.stats.distribution.
        # Arguments
            distribution: a string or scipy.stats distribution object.
        # Returns
            A list of distribution parameter strings.
        """
        if isinstance(distribution, str):
            distribution = getattr(stats, distribution)
        if distribution.shapes:
            parameters = [name.strip() for name in distribution.shapes.split(',')]
        else:
            parameters = []
        if distribution.name in stats._discrete_distns._distn_names:
            parameters += ['loc']
        elif distribution.name in stats._continuous_distns._distn_names:
            parameters += ['loc', 'scale']
        else:
            sys.exit("Distribution name not found in discrete or continuous lists.")
        return parameters

    def nll1(self, y_true, y_pred):
        """ Negative log likelihood. """
        # keras.losses.binary_crossentropy give the mean
        # over the last axis. we require the sum
        return tf.keras.backend.sum(tf.keras.backend.binary_crossentropy(y_true, y_pred), axis=-1)

    def fit_poisson(self, data, vb=0):
        x,y=self.x,self.y
        # %% Setting the Tensor to input model
        X = tf.constant(x, dtype=tf.float64)
        y = tf.constant(y, dtype=tf.float64)
        # %% Weight for training
        lam_ = tf.Variable(initial_value=1, dtype=tf.float64)
        variables = [lam_]
        for e in range(self.num_epoch):
            # 使用tf.GradientTape()記錄損失函數的梯度資訊
            with tf.GradientTape() as tape:
                # y_pred = amp_ * tfp.distributions.Normal(loc=loc_, scale=scale_).prob(X) + bias_
                y_pred = tfp.distributions.Poisson(rate=lam_).prob(X)
                loss = self.nll1(y_true=y, y_pred=y_pred)
            # TensorFlow自動計算損失函數關於自變數（模型參數）的梯度
            grads = tape.gradient(loss, variables)
            print("===grads:{},loss:{}....{:3f}%===".format(grads, loss, 100 * e / self.num_epoch))
            if vb == 1:
                print("===grads:{},loss:{}....{:3f}%===".format(grads, loss, 100 * e / self.num_epoch))
            # TensorFlow自動根據梯度更新參數
            self.optimizer.apply_gradients(grads_and_vars=zip(grads, variables))
        lam_pred = variables[0].numpy()
        mse = mean_squared_error(y_true=y, y_pred=y_pred)
        if vb == 1:
            plt.bar(x=x, height=y, width=1, color='b', alpha=0.3)
            plt.scatter(x=x, y=y_pred)
            plt.show()
            print("======Mean square error:{}".format(mse))

        return np.array([lam_pred])

    def __fit_poisson(self,data):
        # get poisson deviated random numbers
        bins = np.arange(np.max(data) + 1) - 0.5
        # self.y, b, patches = plt.hist(data, bins=bins, density=True, label='Data')
        self.y, bin_edges=np.histogram(data,bins=bins,density=True)
        self.x = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        def fit_function(k, lamb):
            '''poisson function, parameter lamb is the fit parameter'''
            return stats.poisson.pmf(k, lamb)
        try:
            parameters, cov_matrix = curve_fit(fit_function, self.x, self.y,maxfev=500)
        except:
            parameters=np.array([1.0])
        return parameters

    def Density_Plot(self, data, N=1,dist_name=None, vb=0, path="TEST.png",sub_title="sub_title"):
        # %% 依據每個欄位畫出機率密度的值方圖或數量
        col_name=list(data.columns)[0]

        dist_name = list(self.get_best(N=N).keys())
        if dist_name[0]=="poisson":
            dist_name = list(self.get_best(N=1).keys())
        plt.bar(self.x, self.y, self.step)
        for idx,dist_name_ in enumerate(dist_name):
            parameters_dic = self.fitted_param[dist_name_]
            params = np.array([])
            for k, v in parameters_dic.items():
                params = np.append(arr=params, values=v)
            if idx==0:
                if dist_name_ == 'poisson':
                    plt.plot(self.x, getattr(stats,dist_name_).pmf(self.x,*params,),'r', label=dist_name_)
                else:

                    plt.plot(self.x, getattr(stats, dist_name_).pdf(self.x, *params), 'r', label=dist_name_)
            else:
                if dist_name_ == 'poisson':
                    plt.plot(self.x, getattr(stats, dist_name_).pmf(self.x, *params),  label=dist_name_)
                else:

                    plt.plot(self.x, getattr(stats, dist_name_).pdf(self.x, *params),  label=dist_name_)
        plt.grid(True)
        plt.title('Histogram of {}_in_{}_with_{}'.format(col_name,sub_title,dist_name))
        plt.xlabel('{} '.format(col_name), size=12)
        plt.ylabel('{}'.format("Probability"), size=12)
        plt.legend()
        plt.savefig(path)
        if vb == 1:
            print("plt.show()")
            plt.show()
        plt.close()
        return

    def get_test_score(self, dist_names=None, vb=0):
        dist_names_ = dist_names
        dist_names=self.distributions
        params=self.fitted_param
        dist_results = {}
        for dist_name in dist_names:
            try:
                params_ = params[dist_name]  #
                param=np.array([])
                for k,v in params_.items():
                    param=np.append(arr=param,values=v)
            except KeyError:
                print("沒有這個分布:{}".format(dist_name))
                continue
            if dist_name=='poisson':

                try:
                    s, p = stats.chisquare(f_obs=self.y, f_exp=self.fitted_pdf[dist_name],
                                       ddof=len(self.y) - 1 - len(param))
                except:
                    s, p = stats.kstest(self._data, dist_name, args=(*param,), mode="auto")  # loc,scale, *arg
            else:
                s, p = stats.kstest(self._data, dist_name, args=(*param,), mode="auto")  # loc,scale, *arg
            loss=self.get_mse()[dist_name]
            dist_results[dist_name] = {"statistic": s, "p-value": p,'mse':loss}
        dist_results_ = {}
        if dist_names_ != None:
            for key in dist_names_:
                dist_results_[key]=dist_results[key]
        else:
            dist_results_ = dist_results
        return dist_results_

    def get_mse(self):
        loss={}
        df=self.df_errors[["sumsquare_error"]]
        for i,row in df.iterrows():
            loss[i]=row.values[0]
        return loss

    def get_simio_expression(self):
        self.fitted_param_simio={}
        for dist_name,paras in self.fitted_param.items():
            if dist_name=='beta':
                self.fitted_param_simio[dist_name]="{:.3f}+{:.3f}*Beta({:.3f},{:.3f})".format(paras['loc'],paras['scale'],paras['b'],paras['a'])
            elif dist_name=='erlang':
                self.fitted_param_simio[dist_name] = "{}+Erlang({},{})".format(paras['loc'],paras['scale'],paras['a'])
            elif dist_name=='expon':
                self.fitted_param_simio[dist_name]="{}+Exponential({})".format(paras['loc'],paras['scale'])
            elif dist_name=='gamma':
                self.fitted_param_simio[dist_name] = "{}+Gamma({},{})".format(paras['loc'],paras['a'],paras['scale'])
            elif dist_name=='weibull_min':
                self.fitted_param_simio[dist_name] = "{}+Weibull({},{})".format(paras['loc'],paras['c'],paras['scale'])
            elif dist_name=='johnsonsu':
                self.fitted_param_simio[dist_name] = "JohnsonSU({},{},{},{})".format(paras['a'],paras['b'],paras['loc'],paras['scale'])
            elif dist_name=='lognorm': #!!!
                self.fitted_param_simio[dist_name] = "{}+LogNormal({},{})".format(paras['loc'],np.log(paras['scale']),paras['s'])
            elif dist_name=='norm':
                self.fitted_param_simio[dist_name] = "Normal({},{})".format(paras['loc'],np.log(paras['scale']))
            elif dist_name=='triang': #!!!!
                self.fitted_param_simio[dist_name] = "Triangular({},{},{})".format(paras['loc'],stats.mode(a=self._data,axis=0)[0][0],np.max(self._data))
            elif dist_name=='uniform':
                self.fitted_param_simio[dist_name] = "Uniform({},{})".format(paras['loc'],paras['loc']+paras['scale'])
            elif dist_name=='poisson': #!!!
                self.fitted_param_simio[dist_name] = "Poisson({})".format(paras['mu'])

if __name__ == '__main__':
    # ['beta', 'erlang', 'expon', 'gamma', 'weibull_min', 'johnsonsu', 'lognorm',
    #  'norm', 'triang', 'uniform', 't', 'alpha']
    test_columns="norm"
    np.random.seed(0)
    a,b,c,s,mu,loc,scale=1,2,1,4,5,6,8
    beta = stats.beta.rvs(a=a, b=b, loc=loc, scale=scale,size=1000)
    erlang = stats.erlang.rvs(a=a, loc=5, scale=scale, size=1000)
    expon = stats.expon.rvs( loc=loc, scale=scale,size=1000)
    gamma = stats.gamma.rvs(a=a, loc=loc, scale=scale,size=1000)
    weibull_min = stats.weibull_min.rvs(c=c, loc=scale, scale=1, size=1000)
    johnsonsu=stats.johnsonsu.rvs(a=a,b=b,loc=scale, scale=6, size=1000)
    lognorm = stats.lognorm.rvs(s=s, loc=100, scale=scale, size=1000)
    norm=stats.norm.rvs(loc=loc, scale=scale, size=1000)
    triang = stats.triang.rvs(c=c, loc=loc, scale=scale, size=1000)
    uniform = stats.uniform.rvs(loc=loc, scale=scale, size=1000)
    poisson = stats.poisson.rvs(mu=mu, loc=scale, size=1000)
    data=pd.DataFrame({'beta':beta,"erlang":erlang,'expon':expon,'gamma':gamma,'weibull_min':weibull_min,'johnsonsu':johnsonsu,'lognorm':lognorm,'norm':norm,'triang':triang,'uniform':uniform,'poisson':poisson})
    # data.to_excel("")
    data=pd.read_excel("鋼捲資料202104.xlsx")
    for test_columns in ["機台"]:
        data_=data[[test_columns]].values
        try:
            np.savetxt('{}.txt'.format(test_columns), data_)
        except:
            pass
        fitter=Fit_Density(data=data_,distributions=None)
        fitter.fit()
        mse=fitter.get_mse()

        all_parameters=fitter.fitted_param
        best=fitter.get_best()
        # test_result=fitter.get_test_score(dist_names=None, vb=0)#[test_columns]
        fitter.Density_Plot(data[[test_columns]],vb=1)
