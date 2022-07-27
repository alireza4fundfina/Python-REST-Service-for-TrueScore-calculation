from __future__ import division
from flask import request
import sys
import datetime as dt
import joblib
import warnings
import json
from scipy.stats import skew, kurtosis
import numpy as np
import pandas as pd
from flask import Flask
import datetime

app = Flask(__name__)
warnings.filterwarnings('ignore')

libfilePath = "C:/Users/alire/PythonService_Fundfina/"
delinquent_model = joblib.load(libfilePath+'delinquent_model.joblib')
default_model = joblib.load(libfilePath+'default_model.joblib')
writeoff_model = joblib.load(libfilePath+'writeoff_model.joblib')


@app.route("/getTrueScore",methods=['POST'])
def getTrueScore():
    # End Library imports

    # Function for engineering features

    start=datetime.datetime.now()
    print(start)
    def feature_engineering(df):
        """ Function for engineering features

            Parameters: 
            df: dataframe

            Returns:
            df: dataframe with added statistics columns 'trans_count_<statistic>', 
                    'trans_vol_<statistic>' and 'ValuePerCount_<statistic>', 
                    and other columns 'AvgNumIrregPaymAcrossLoans', 
                    'WorstDelinqOrdinalAcrossLoans' and 'DaysFrmOnbrdToLnStrt';
                    and removed columns 'activation_date', 'tvol_Month[01-12]' 
                    'tcnt_Month[01-12]', 'delinquencyString', 'loanStartDate', 
                    and 'onboardedDate'

        """

        Dict_DelinqStatusToOrdinalValue = {'X': 0, 'P': 1, 'L': 2, 'Q': 4, 'D': 8, 'W': 16}

        df['AvgNumIrregPaymAcrossLoans'] = 0.0
        df['WorstDelinqOrdinalAcrossLoans'] = 0.0

        for ii in range(len(df)):

            if (len(df['delinquencyString'].iloc[ii]) > 1*5):

                TotalNumIrregPaymAcrossLoans = 0
                WorstDelinqOrdinalAcrossLoans = Dict_DelinqStatusToOrdinalValue['X']
                if (len(df['delinquencyString'].iloc[ii]) > 10*5):
                    LenStr = 10*5
                else:
                    LenStr = int(len(df['delinquencyString'].iloc[ii]))
                for jj in range(LenStr // 5):
                    TotalNumIrregPaymAcrossLoans += int((df['delinquencyString'].iloc[ii])[(jj*5+2):(jj*5+4)])
                    if (Dict_DelinqStatusToOrdinalValue[((df['delinquencyString'].iloc[ii])[jj*5+4])] > WorstDelinqOrdinalAcrossLoans):
                        WorstDelinqOrdinalAcrossLoans = Dict_DelinqStatusToOrdinalValue[((df['delinquencyString'].iloc[ii])[jj*5+4])]
                # end of for jj in range(LenStr // 5)
                NumLoans = LenStr // 5
                if (len(df['delinquencyString'].iloc[ii]) > 10*5):
                    if (len(df['delinquencyString'].iloc[ii]) > 10*5 + 90*6):
                        LenStr = 10*5 + 90*6
                    else:
                        LenStr = int(len(df['delinquencyString'].iloc[ii]))
                    for jj in range(10, (10 + ((LenStr-10*5) // 6))):
                        TotalNumIrregPaymAcrossLoans += int((df['delinquencyString'].iloc[ii])[(10*5+(jj-10)*6+3):(10*5+(jj-10)*6+5)])
                        if (Dict_DelinqStatusToOrdinalValue[((df['delinquencyString'].iloc[ii])[10*5+(jj-10)*6+5])] > WorstDelinqOrdinalAcrossLoans):
                            WorstDelinqOrdinalAcrossLoans = Dict_DelinqStatusToOrdinalValue[((df['delinquencyString'].iloc[ii])[10*5+(jj-10)*6+5])]
                    # end of for jj in range(10, (10 + (LenStr-10*5) // 6))
                    NumLoans = 10 + ((LenStr-10*5) // 6)
                # end of if (len(df['delinquencyString'].iloc[ii]) > 10*5)
                df['AvgNumIrregPaymAcrossLoans'].iloc[ii] = TotalNumIrregPaymAcrossLoans / NumLoans
                df['WorstDelinqOrdinalAcrossLoans'].iloc[ii] = np.copy(WorstDelinqOrdinalAcrossLoans)

            else:  # of if (len(df['delinquencyString'].iloc[ii]) > 1*5)

                df['AvgNumIrregPaymAcrossLoans'].iloc[ii] = -9999.0
                df['WorstDelinqOrdinalAcrossLoans'].iloc[ii] = -9999.0

            # end of if (len(df['delinquencyString'].iloc[ii]) > 1*5)

        # end of for ii in range(len(df))

        df['loanStartDate'] = pd.to_datetime(df['loanStartDate'], format='%d-%m-%Y')
        df['onboardedDate'] = pd.to_datetime(df['onboardedDate'], format='%d-%m-%Y')

        df['DaysFrmOnbrdToLnStrt'] = [0]*len(df)
        for index in range(0, len(df)):  # looping through rows
            df['DaysFrmOnbrdToLnStrt'][index] = (df['loanStartDate'][index] - df['onboardedDate'][index]).days

        df['DaysFrmOnbrdToLnStrt'] = df['DaysFrmOnbrdToLnStrt'] * 1.0

        # For df, create new ValuePerCount_Month[01-12] columns that are initialized to 0s   
        for i in range(1, 12+1):  # loops from 1 to 12 (i.e., # of months) to assign list of 0s to created each column
            if i <= 9:            
                df['ValuePerCount_Month0%d' % i] = [0]*len(df)  # len(df) is number of rows in dataframe
            else:            
                df['ValuePerCount_Month%d' % i] = [0]*len(df)   # len(df) is number of rows in dataframe
            # end of if i<=9
        # end of for i in range(1,13)

        # Subfunction to calculate ValuePerCount values for each row and month (row number is the argument 'index')
        def calculate_ValuePerCount(index):
            """Subfunction to calculate ValuePerCount values for each row and past N months 

                Parameters: 
                index (int): row number of dataframe

                Returns: 
                df: dataframe with added columns ValuePerCount_Month[01-12]

            """
            
            start_C1 = df.columns.get_loc("tcnt_Month01")      # get column number of tcnt of immediate past Month (i.e., month # 1)        
            start_V1 = df.columns.get_loc("tvol_Month01")      # get column number of tvol of immediate past Month (i.e., month # 1)
            start_VPC1 = df.columns.get_loc("ValuePerCount_Month01")      # get column number of ValuePerCount of immediate past Month (i.e., month # 1)
            
            for i in range(1, 12+1):   # loops from 1 to 12 (i.e., # of months) to assign list of 0s into each column

                if df.iloc[index, start_C1] == 0:  # if count is 0, then set ValuePerCount to 0
                    df.iloc[index, start_VPC1] = 0
                else:
                    df.iloc[index, start_VPC1] = np.true_divide(df.iloc[index, start_V1], df.iloc[index, start_C1])
                # end of if df.iloc[index,start_C1]==0

                # Following lines move ahead by 1 column to column number for next month
                start_C1 += 1    # for tcnt
                start_V1 += 1    # for tvol
                start_VPC1 += 1    # for ValuePerCount

            # end of for i in range(1,13)

        # end of def calculate_ValuePerCount(index)

        for index in range(0, len(df)):  # for each row of dataframe
            calculate_ValuePerCount(index)

        # Start of calculation of mean, std, skew, kurtosis using Month01 to Month12 data for every feature

        # get columns numbers of past 1st and Nth (i.e., 12th) months for count, vol and ValuePerCount
        index_count_month_1 = df.columns.get_loc("tcnt_Month01")
        index_count_month_12 = df.columns.get_loc("tcnt_Month12")
        index_vol_month_1 = df.columns.get_loc("tvol_Month01")
        index_vol_month_12 = df.columns.get_loc("tvol_Month12")
        index_vpc_month_1 = df.columns.get_loc("ValuePerCount_Month01")
        index_vpc_month_12 = df.columns.get_loc("ValuePerCount_Month12")

        # get arrays for past N (i.e., 12) months and dataframe rows for count, vol and ValuePerCount
        #   (rows is dataframe rows, columns is the past N columns of dataframe)
        nparray_of_count = df.values[:, index_count_month_1:index_count_month_12+1]
        nparray_of_vol = df.values[:, index_vol_month_1:index_vol_month_12+1]
        nparray_of_vpc = df.values[:, index_vpc_month_1:index_vpc_month_12+1]

        # calculate stats for count, vol and ValuePerCount (each result, e.g, mean, forms a new column of the dataframe)
        df['trans_count_mean'] = df.iloc[:,index_count_month_1:index_count_month_12+1].mean(axis=1)
        df['trans_count_std'] = df.iloc[:,index_count_month_1:index_count_month_12+1].std(axis=1, ddof=0)
        df['trans_count_skews'] = df.iloc[:,index_count_month_1:index_count_month_12+1].skew(axis=1)
        df['trans_count_kurtosis'] = df.iloc[:,index_count_month_1:index_count_month_12+1].kurtosis(axis=1)
        df['trans_vol_mean'] = df.iloc[:,index_vol_month_1:index_vol_month_12+1].mean(axis=1)
        df['trans_vol_std'] = df.iloc[:,index_vol_month_1:index_vol_month_12+1].std(axis=1, ddof=0)
        df['trans_vol_skews'] = df.iloc[:,index_vol_month_1:index_vol_month_12+1].skew(axis=1)
        df['trans_vol_kurtosis'] = df.iloc[:,index_vol_month_1:index_vol_month_12+1].kurtosis(axis=1)
        df['ValuePerCount_mean'] = df.iloc[:,index_vpc_month_1:index_vpc_month_12+1].mean(axis=1)
        df['ValuePerCount_std'] = df.iloc[:,index_vpc_month_1:index_vpc_month_12+1].std(axis=1, ddof=0)
        df['ValuePerCount_skews'] = df.iloc[:,index_vpc_month_1:index_vpc_month_12+1].skew(axis=1)
        df['ValuePerCount_kurtosis'] = df.iloc[:,index_vpc_month_1:index_vpc_month_12+1].kurtosis(axis=1)

        # End of calculation of mean, std, skew, kurtosis using Month01 to Month12 data for every feature

        # Begin TRUEscore_v1.1 calculation

        #  begin initialising variables
        today1 = pd.Timestamp(dt.date.today())
        maxMonthsForTRUEscorev1p1 = 18
        weights = pd.DataFrame(index = ['trans_vol_mean_12m', 'trans_vol_std_12m',\
                                    'cnt_cv_inv_12m', 'gaps_busn', 'vintage'],
                            columns = ['model1'],\
                            data=[0.03, 0.00175, 0.000275, -0.0165, 100])
        #  end initialising variables

        #  begin creating TRUEscore_v1.1 data

        df_ForTRUEscorev1p1 = pd.DataFrame(columns = ['partnerMerchantCode', 'trans_vol_mean_12m',\
                                                    'trans_vol_std_12m', 'cnt_cv_inv_12m', \
                                                    'gaps_busn', 'vintage', 'TRUEscore'])

        df_ForTRUEscorev1p1['partnerMerchantCode'] = df['partnerMerchantCode']
        df_ForTRUEscorev1p1['trans_vol_mean_12m'] = df['trans_vol_mean']
        df_ForTRUEscorev1p1['trans_vol_std_12m'] = df['trans_vol_std']

        df_ForTRUEscorev1p1['cnt_cv_inv_12m'] = df['trans_count_mean'] /\
            df['trans_count_std'] *\
            df_ForTRUEscorev1p1['trans_vol_mean_12m']

        df_ForTRUEscorev1p1['gaps_busn'] = (df.iloc[:, index_vol_month_1:index_vol_month_12+1] == 0).sum(axis=1) *\
            df_ForTRUEscorev1p1['trans_vol_mean_12m']/12.0

        df_ForTRUEscorev1p1['vintage'] = (today1 - pd.to_datetime(df['activation_date'])).dt.days/30
        df_ForTRUEscorev1p1['vintage'] = df_ForTRUEscorev1p1['vintage'].where(df_ForTRUEscorev1p1['vintage'] <= maxMonthsForTRUEscorev1p1,
                                                                            maxMonthsForTRUEscorev1p1)

        df_ForTRUEscorev1p1['raw'] = df_ForTRUEscorev1p1[df_ForTRUEscorev1p1.columns[1:6]].dot(weights)
        df_ForTRUEscorev1p1['raw2'] = df_ForTRUEscorev1p1['raw'].where(df_ForTRUEscorev1p1['raw'] >= 1, 1)
        df_ForTRUEscorev1p1['TRUEscore'] = round(np.log(df_ForTRUEscorev1p1['raw2'])/np.log(1.9455)*100)-933.0
        df_ForTRUEscorev1p1['TRUEscore'] = df_ForTRUEscorev1p1['TRUEscore'].where(df_ForTRUEscorev1p1['TRUEscore'] >= 150, 150)
        df_ForTRUEscorev1p1['TRUEscore'] = df_ForTRUEscorev1p1['TRUEscore'].where(df_ForTRUEscorev1p1['TRUEscore'] <= 950, 950)

        #  end creating TRUEscore_v1.1 data

        # End TRUEscore_v1.1 calculation

        # start of removing df columns that are not required
        remove_cols = ['delinquencyString', 'activation_date',
                    'loanStartDate', 'onboardedDate',
                    'tvol_Month01', 'tvol_Month02', 'tvol_Month03',
                    'tvol_Month04', 'tvol_Month05', 'tvol_Month06',
                    'tvol_Month07', 'tvol_Month08', 'tvol_Month09',
                    'tvol_Month10', 'tvol_Month11', 'tvol_Month12',
                    'tcnt_Month01', 'tcnt_Month02', 'tcnt_Month03',
                    'tcnt_Month04', 'tcnt_Month05', 'tcnt_Month06',
                    'tcnt_Month07', 'tcnt_Month08', 'tcnt_Month09',
                    'tcnt_Month10', 'tcnt_Month11', 'tcnt_Month12',
                    'ValuePerCount_Month01', 'ValuePerCount_Month02',
                    'ValuePerCount_Month03', 'ValuePerCount_Month04',
                    'ValuePerCount_Month05', 'ValuePerCount_Month06',
                    'ValuePerCount_Month07', 'ValuePerCount_Month08',
                    'ValuePerCount_Month09', 'ValuePerCount_Month10',
                    'ValuePerCount_Month11', 'ValuePerCount_Month12']
        df.drop(columns=remove_cols, axis=1, errors='ignore', inplace=True)
        # end of removing df columns that are not required

        return df, df_ForTRUEscorev1p1

    # end of def feature_engineering(df)

    # Function for scaling numerical variable to a range using max_abs scaling


    def scaling_function(df, FlagsDict, Dict_PastMaxes, Dict_MaxesAtLastBinEdge):
        """Function for scaling numerical variable to a range using max_abs scaling

        Parameters: 
            df: dataframe
            FlagsDict (dictionary): (Empty) For storing lists of flags, specifically 
            by adding in this function the column of flags indicating whether any 
            pre-scaled df values are greater than max_abs

        Returns:
            df: max_abs-scaled df
            FlagsDict (dictionary): FlagsDict modified by adding the column of flags 
            indicating whether any pre-scaled df values are greater than max_abs 

        """

        # list of columns for which values are checked
        list_var = ['ValuePerCount_mean',
                    'ValuePerCount_std'	,
                    'ValuePerCount_skews'	,
                    'ValuePerCount_kurtosis'	,
                    'trans_count_mean'	,
                    'trans_count_std'	,
                    'trans_count_skews'	,
                    'trans_count_kurtosis'	,
                    'trans_vol_mean'	,
                    'trans_vol_std'	,
                    'trans_vol_skews'	,
                    'trans_vol_kurtosis']

        list_max = [Dict_PastMaxes[this_list_var] + 0.000001 for this_list_var in list_var]

        for j in range(len(list_var)):  # for each statistic column
            FlagsDict[list_var[j]+'_outofscalerange'] = []
            for i in range(len(df)):  # for each row of dataframe
                if (df[list_var[j]][i] > list_max[j]):  # if beyond its past max value
                    FlagsDict[list_var[j]+'_outofscalerange'].append(1)
                else:
                    FlagsDict[list_var[j]+'_outofscalerange'].append(0)
                # end of if (df[list_var[j]][i] > list_max[j])            
                df[list_var[j]][i] = df[list_var[j]][i] / list_max[j] # Scale each value in these list_var columns
            # end of for i in range(len(df))
        # end of for j in range(len(list_var))

        list_var = ['AvgNumIrregPaymAcrossLoans',
                    'DaysFrmOnbrdToLnStrt']

        list_PastMax = [Dict_PastMaxes[this_list_var] + 0.000001 for this_list_var in list_var]
        list_MaxAtLastBinEdge = [ Dict_MaxesAtLastBinEdge[this_list_var] for this_list_var in list_var]

        for j in range(len(list_var)):  # for each column
            FlagsDict[list_var[j]+'_PastMaxCrossed'] = []
            for i in range(len(df)):  # for each row of dataframe
                if (df[list_var[j]][i] > list_PastMax[j]):  # if beyond its past max value
                    FlagsDict[list_var[j]+'_PastMaxCrossed'].append(1)
                else:
                    FlagsDict[list_var[j]+'_PastMaxCrossed'].append(0)
                # end of if (df[list_var[j]][i] > list_PastMax[j])
                if (df[list_var[j]][i] > (-9999+1)):                
                    df[list_var[j]][i] = df[list_var[j]][i] / list_MaxAtLastBinEdge[j] # Scale each value in these list_var columns
            # end of for i in range(len(df))
        # end of for j in range(len(list_var))

        list_var = ['WorstDelinqOrdinalAcrossLoans']
        list_max = 16 + 0.000001

        for j in range(len(list_var)):  # for each column
            for i in range(len(df)):  # for each row of dataframe
                if (df[list_var[j]][i] > (-9999+1)):                
                    df[list_var[j]][i] = df[list_var[j]][i] / list_max # Scale each value in these list_var columns
            # end of for i in range(len(df))
        # end of for j in range(len(list_var))

        return df, FlagsDict

    # end of def scaling_function(df, FlagsDict, Dict_PastMaxes, Dict_MaxesAtLastBinEdge)

    # Function for Bucketing Numerical variables into 10 buckets


    def discretization_function(df, FlagsDict):
        """Function for Bucketing Numerical variables into 10 buckets

        Parameters: 
            df: dataframe
            FlagsDict (dictionary): Stores lists of flags, specifically by 
                    adding in this function the column of flags indicating whether any
            statistics are beyond the range from 1st to last (i.e., extreme) bins

        Returns: 
            df: df modified so that 'ValuePerCount_<statistic>', 
                    'trans_count_<statistic>' and 'trans_vol_<statistic>' 
                            column values are set to bin numbers instead of original values     
            FlagsDict (dictionary): FlagsDict modified by adding in this function 
            the column of flags indicating whether any statistics are beyond the
            range from 1st to last (i.e., extreme) bins 

        """

        FinalNumBins_Dict = {'ValuePerCount_mean': 9,
                            'ValuePerCount_std': 9,
                            'ValuePerCount_skews': 10,
                            'ValuePerCount_kurtosis': 10,
                            'trans_count_mean': 9,
                            'trans_count_std': 9,
                            'trans_count_skews': 10,
                            'trans_count_kurtosis': 9,
                            'trans_vol_mean': 9,
                            'trans_vol_std': 9,
                            'trans_vol_skews': 9,
                            'trans_vol_kurtosis': 9,
                            'AvgNumIrregPaymAcrossLoans': 3,
                            'DaysFrmOnbrdToLnStrt': 6,
                            'WorstDelinqOrdinalAcrossLoans': 7}

        Bin_Range_Dict = {'ValuePerCount_mean': [[-1e-06, 0.007720719591194402],
                                                [0.007720719591194402, 0.018166034792321973],
                                                [0.018166034792321973, 0.026191699019101095],
                                                [0.026191699019101095, 0.034270044985184066],
                                                [0.034270044985184066, 0.041896766663706855],
                                                [0.041896766663706855, 0.04952743788647816],
                                                [0.04952743788647816, 0.05826017920936626],
                                                [0.05826017920936626, 0.07056183882956182],
                                                [0.07056183882956182, 1.0]],
                        'ValuePerCount_std': [[-1e-06, 0.006742924232862974],
                                                [0.006742924232862974, 0.01200068875117684],
                                                [0.01200068875117684, 0.01771753453726564],
                                                [0.01771753453726564, 0.023439246823797468],
                                                [0.023439246823797468, 0.027738106340057522],
                                                [0.027738106340057522, 0.032932981826894546],
                                                [0.032932981826894546, 0.03894870387374006],
                                                [0.03894870387374006, 0.04901395836119603],
                                                [0.04901395836119603, 1.0]],
                        'ValuePerCount_skews': [[-0.9878022268413098, -0.36814474381072537],
                                                [-0.36814474381072537, -0.22220076588553433],
                                                [-0.22220076588553433, -0.11554326602020794],
                                                [-0.11554326602020794, -0.047269707931170066],
                                                [-0.047269707931170066, 0.0],
                                                [0.0, 0.004107939033581731],
                                                [0.004107939033581731, 0.09560826924171471],
                                                [0.09560826924171471, 0.19741767956558795],
                                                [0.19741767956558795, 0.396369861681814],
                                                [0.396369861681814, 1.0]],
                        'ValuePerCount_kurtosis': [[-0.20360271310841613, -0.1796676897950772],
                                                    [-0.1796676897950772, -0.13814818632534934],
                                                    [-0.13814818632534934, -0.12018069276253564],
                                                    [-0.12018069276253564, -0.06074645575571366],
                                                    [-0.06074645575571366, -0.025197875744849857],
                                                    [-0.025197875744849857, 0.0],
                                                    [0.0, 0.005744529256275419],
                                                    [0.005744529256275419, 0.12753888610786668],
                                                    [0.12753888610786668, 0.31663367041367574],
                                                    [0.31663367041367574, 1.0]],
                        'trans_count_mean': [[-1e-06, 0.0032826299646843446],
                                            [0.0032826299646843446, 0.005815019720920383],
                                            [0.005815019720920383, 0.00809996182945768],
                                            [0.00809996182945768, 0.010542849245036725],
                                            [0.010542849245036725, 0.013933408502866267],
                                            [0.013933408502866267, 0.018387466161443575],
                                            [0.018387466161443575, 0.025962698581951466],
                                            [0.025962698581951466, 0.046936080409239626],
                                            [0.046936080409239626, 1.0]],
                        'trans_count_std': [[-1e-06, 0.0033903860414464556],
                                            [0.0033903860414464556, 0.0061393516597672585],
                                            [0.0061393516597672585, 0.00896412515415251],
                                            [0.00896412515415251, 0.01207150705910986],
                                            [0.01207150705910986, 0.015885086827367733],
                                            [0.015885086827367733, 0.02138974834526246],
                                            [0.02138974834526246, 0.031117392468865246],
                                            [0.031117392468865246, 0.05920697014938556],
                                            [0.05920697014938556, 1.0]],
                        'trans_count_skews': [[-0.8739898121346914, -0.21113323872251286],
                                                [-0.21113323872251286, -0.10859103751153659],
                                                [-0.10859103751153659, -0.03155434384584324],
                                                [-0.03155434384584324, 0.0],
                                                [0.0, 0.0005950057880918635],
                                                [0.0005950057880918635, 0.08140398978984092],
                                                [0.08140398978984092, 0.1563088750336633],
                                                [0.1563088750336633, 0.24918307627456288],
                                                [0.24918307627456288, 0.4126226220901422],
                                                [0.4126226220901422, 1.0]],
                        'trans_count_kurtosis': [[-0.20169601087113737, -0.15413633487868647],
                                                [-0.15413633487868647, -0.13060452664845731],
                                                [-0.13060452664845731, -0.10361777612033926],
                                                [-0.10361777612033926, -0.07165778431434981],
                                                [-0.07165778431434981, -0.037887756696885606],
                                                [-0.037887756696885606, 0.0],
                                                [0.0, 0.04853962172984235],
                                                [0.04853962172984235, 0.19266536592666153],
                                                [0.19266536592666153, 1.0]],
                        'trans_vol_mean': [[-1e-06, 0.004010596548511582],
                                            [0.004010596548511582, 0.006851788673398725],
                                            [0.006851788673398725, 0.009538571748050286],
                                            [0.009538571748050286, 0.012688348775543853],
                                            [0.012688348775543853, 0.01691051054129622],
                                            [0.01691051054129622, 0.023236418259572546],
                                            [0.023236418259572546, 0.034638788900024954],
                                            [0.034638788900024954, 0.06409003267024199],
                                            [0.06409003267024199, 1.0]],
                        'trans_vol_std': [[-1e-06, 0.0036923175240220945],
                                            [0.0036923175240220945, 0.006816097999046159],
                                            [0.006816097999046159, 0.009890414904658954],
                                            [0.009890414904658954, 0.013317979973051154],
                                            [0.013317979973051154, 0.01707884257492206],
                                            [0.01707884257492206, 0.0249002038953258],
                                            [0.0249002038953258, 0.03649522585101739],
                                            [0.03649522585101739, 0.0669543755410302],
                                            [0.0669543755410302, 1.0]],
                        'trans_vol_skews': [[-0.8286630492430477, -0.17273101297294682],
                                            [-0.17273101297294682, -0.0763705805666481],
                                            [-0.0763705805666481, 0.0],
                                            [0.0, 0.03238457760398729],
                                            [0.03238457760398729, 0.10954965753195291],
                                            [0.10954965753195291, 0.18514646248136296],
                                            [0.18514646248136296, 0.28532171736014705],
                                            [0.28532171736014705, 0.45053823533686654],
                                            [0.45053823533686654, 1.0]],
                        'trans_vol_kurtosis': [[-0.20155038349562832, -0.14812967796312898],
                                                [-0.14812967796312898, -0.12460635779610382],
                                                [-0.12460635779610382, -0.09432883149397484],
                                                [-0.09432883149397484, -0.0636596649582743],
                                                [-0.0636596649582743, -0.02647993436577496],
                                                [-0.02647993436577496, 0.0],
                                                [0.0, 0.06209183261643961],
                                                [0.06209183261643961, 0.2113465316746568],
                                                [0.2113465316746568, 1.0]],
                        'AvgNumIrregPaymAcrossLoans': [[-10000, -1e-06],
                                                        [-1e-06, 0.052500000000000005],
                                                        [0.052500000000000005, 1.0]],
                        'DaysFrmOnbrdToLnStrt': [[-10000, -1e-06],
                                                [-1e-06, 0.08039430449069],
                                                [0.08039430449069, 0.22179627601314347],
                                                [0.22179627601314347, 0.33263964950711966],
                                                [0.33263964950711966, 0.5783132530120482],
                                                [0.5783132530120482, 1.0]],
                        'WorstDelinqOrdinalAcrossLoans': [[-9999-1, -0.000001],
                                                            [-0.000001, 0.5/16.0],
                                                            [0.5/16.0, 1.5/16.0],
                                                            [1.5/16.0, 3.0/16.0],
                                                            [3.0/16.0, 6.0/16.0],
                                                            [6.0/16.0, 12.0/16.0],
                                                            [12.0/16.0, 1.0]]}

        # list of continuous-value columns to bin
        Col_to_binning = ['ValuePerCount_mean', 'ValuePerCount_std',
                        'ValuePerCount_skews', 'ValuePerCount_kurtosis',
                        'trans_count_mean', 'trans_count_std',
                        'trans_count_skews', 'trans_count_kurtosis',
                        'trans_vol_mean', 'trans_vol_std',
                        'trans_vol_skews', 'trans_vol_kurtosis']

        for col in Col_to_binning:  # for each column that is to be binned
            FlagsDict[col + '_outofbinrange'] = []
            for i in range(len(df)):  # for each row of dataframe
                if ((df[col][i] <= Bin_Range_Dict[col][0][0]) or
                        (df[col][i] > Bin_Range_Dict[col][FinalNumBins_Dict[col]-1][1])):  # if beyond
                        # the range from 1st to last (i.e., extreme) bins
                    FlagsDict[col+'_outofbinrange'].append(1)
                else:
                    FlagsDict[col+'_outofbinrange'].append(0)
                # end of if ( (df[col][i] <= Bin_Range_Dict[col][0][0]) or...
            # end of for i in range(len(df))
            # set appropriate bin number
            df.loc[(df[col] <= Bin_Range_Dict[col][0][0]), col] = 0
            df.loc[(df[col] > Bin_Range_Dict[col][FinalNumBins_Dict[col]-1][1]), col] = FinalNumBins_Dict[col]-1
            for i in range(0, FinalNumBins_Dict[col]):
                df.loc[(df[col] > Bin_Range_Dict[col][FinalNumBins_Dict[col]-1-i][0]) & (df[col] <= Bin_Range_Dict[col][FinalNumBins_Dict[col]-1-i][1]), col] = FinalNumBins_Dict[col]-1-i
            # end of for i in range(0,FinalNumBins_Dict[col])
        # end of for col in Col_to_binning

        # list of continuous-value columns to bin
        Col_to_binning = ['AvgNumIrregPaymAcrossLoans', 'DaysFrmOnbrdToLnStrt',
                        'WorstDelinqOrdinalAcrossLoans']
        for col in Col_to_binning:  # for each column that is to be binned
            # set appropriate bin number
            df.loc[(df[col] <= Bin_Range_Dict[col][0][0]), col] = 0
            df.loc[(df[col] > Bin_Range_Dict[col][FinalNumBins_Dict[col]-1][1]), col] = FinalNumBins_Dict[col]-1
            for i in range(0, FinalNumBins_Dict[col]):
                df.loc[(df[col] > Bin_Range_Dict[col][FinalNumBins_Dict[col]-1-i][0]) & (df[col] <= Bin_Range_Dict[col][FinalNumBins_Dict[col]-1-i][1]), col] = FinalNumBins_Dict[col]-1-i
            # end of for i in range(0,FinalNumBins_Dict[col])
        # end of for col in Col_to_binning

        return df, FlagsDict, FinalNumBins_Dict

    # end of def discretization_function(df, FlagsDict)

    # Function to created encoded dataframe from which inputs will be extracted


    def data_encoding(df, FlagsDict, Dict_PastMaxes, Dict_MaxesAtLastBinEdge):
        """Function to created encoded dataframe from which inputs will be extracted

        Parameters: 
            df: dataframe of engineered features obtained through the 
            feature_engineering function
            FlagsDict (dictionary): Stores lists of flags, specifically by 
                    adding in this function the column of flags indicating whether 
            any out-of-range, out-of-statistics-range, or new unknown 
            values exist 

        Returns: 
            df_enc: encoded dataframe (one-hot, sequential, and 'Both') 
            corresponding to df
            FlagsDict (dictionary): FlagsDict modified by adding in this function 
            the column of flags indicating whether any out-of-range, 
            out-of-statistics-range, or new unknown values exist  

        """

        list_KnownEnterprisePartnerValues = ['BAN', 'CHA', 'EAS', 'EKO',
                                            'INS', 'MIN', 'RRS', 'SUG']  # enterprise partners
        list_KnownChildrenValues = ['Yes', 'un', 'No']
        list_KnownMaritalStatusValues = ['Married', 'un', 'Single']
        list_KnownGenderValues = ['Male', 'un', 'Female']
        list_KnownWhatsAppValues = ['Yes', 'un', 'No']
        list_KnownPurchasedInOneYearValues = ['Yes', 'un', 'No']
        list_KnownRecommendedValues = ['Yes', 'un', 'No']
        list_KnownHomeOwnershipTypeValues = ['Both', 'Own', 'Rent', 'un']
        list_KnownVehicleValues = ['2-Wheeler', '4-Wheeler', 'Both', 'No', 'un']
        list_KnownDegreeValues = ['Graduation', 'MBA', 'MCA',
                                'Other', 'Post Graduation', 'un']
        list_KnownJobTypeValues = ['Both', 'No', 'Private', 'Public', 'un']
        list_KnownStateValues = ['AN', 'AP', 'AR', 'AS', 'BR', 'CH', 'CT', 'DH', 'DL',
                                'GA', 'GJ', 'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA',
                                'LD', 'MH', 'ML', 'MN', 'MP', 'MZ', 'NL', 'OR', 'PB',
                                'PY', 'RJ', 'SK', 'TG', 'TN', 'TR', 'UP', 'UT', 'WB']

        FlagsDict['NewUnknown_enterprisePartner'] = []
        FlagsDict['NewUnknown_children'] = []
        FlagsDict['NewUnknown_maritalStatus'] = []
        FlagsDict['NewUnknown_gender'] = []
        FlagsDict['NewUnknown_whatsApp'] = []
        FlagsDict['NewUnknown_purchasedInOneYear'] = []
        FlagsDict['NewUnknown_recommended'] = []
        FlagsDict['NewUnknown_homeOwnershipType'] = []
        FlagsDict['NewUnknown_vehicle'] = []
        FlagsDict['NewUnknown_degree'] = []
        FlagsDict['NewUnknown_jobType'] = []
        FlagsDict['NewUnknown_state'] = []

        # Subfunction to replace NewUnknown value with relevant value
        def ReplaceNewUnknownValues(ThisDf, ThisCol, list_KnownValuesInCol, SubstitutionValue, ThisFlagsDict):
            """Subfunction to replace NewUnknown value with relevant value

            Parameters: 
                ThisDf: dataframe
                ThisCol (string): column name whose NewUnknown values are to be replaced
                list_KnownValuesInCol: list of known values in this column
                SubstitutionValue: Value which replaces the NewUnknown values
                ThisFlagsDict (dictionary): Stores lists of flags, specifically by 
                        adding in this subfunction the column of flags associated with ThisCol

            Returns:
                ThisDf: Modified dataframe 
                ThisFlagsDict (dictionary): ThisFlagsDict modified by adding in this 
                subfunction the column of flags associated with ThisCol 

            """

            for i in range(len(ThisDf)):  # looping through each row
                if (ThisDf[ThisCol][i] not in list_KnownValuesInCol):
                    ThisFlagsDict['NewUnknown_' + ThisCol].append(1)
                    ThisDf.loc[i, ThisCol] = SubstitutionValue
                else:
                    ThisFlagsDict['NewUnknown_' + ThisCol].append(0)
                # end of if (ThisDf[ThisCol][i] not in list_KnownValuesInCol)
            # end of for i in range(len(ThisDf))

            return ThisDf, ThisFlagsDict

        # end of def ReplaceNewUnknownValues(ThisDf, ThisCol, list_KnownValuesInCol, SubstitutionValue, ThisFlagsDict)

        df, FlagsDict = ReplaceNewUnknownValues(df, 'enterprisePartner', list_KnownEnterprisePartnerValues, 'UNK', FlagsDict)  # NOTE 'UNK',...
        # ...not 'un'
        df, FlagsDict = ReplaceNewUnknownValues(df, 'children', list_KnownChildrenValues, 'un', FlagsDict)
        df, FlagsDict = ReplaceNewUnknownValues(df, 'maritalStatus', list_KnownMaritalStatusValues, 'un', FlagsDict)
        df, FlagsDict = ReplaceNewUnknownValues(df, 'gender', list_KnownGenderValues, 'un', FlagsDict)
        df, FlagsDict = ReplaceNewUnknownValues(df, 'whatsApp', list_KnownWhatsAppValues, 'un', FlagsDict)
        df, FlagsDict = ReplaceNewUnknownValues(df, 'purchasedInOneYear', list_KnownPurchasedInOneYearValues, 'un', FlagsDict)
        df, FlagsDict = ReplaceNewUnknownValues(df, 'recommended', list_KnownRecommendedValues, 'un', FlagsDict)
        df, FlagsDict = ReplaceNewUnknownValues(df, 'homeOwnershipType', list_KnownHomeOwnershipTypeValues, 'un', FlagsDict)
        df, FlagsDict = ReplaceNewUnknownValues(df, 'vehicle', list_KnownVehicleValues, 'un', FlagsDict) 
        df, FlagsDict = ReplaceNewUnknownValues(df, 'degree', list_KnownDegreeValues, 'UNK', FlagsDict) # SY: NOTE list_KnownDegreeValues...
        # ...already has an 'un' value besides this 'UNK' value assigned
        df, FlagsDict = ReplaceNewUnknownValues(df, 'jobType', list_KnownJobTypeValues, 'un', FlagsDict)
        df, FlagsDict = ReplaceNewUnknownValues(df, 'state', list_KnownStateValues, 'UNK', FlagsDict)

        # lists of specific categorical (to one-hot-encode), and numerical (continuous) features
        list_cat = ['homeOwnershipType', 'vehicle', 'degree',
                    'jobType', 'enterprisePartner', 'state']
        list_num = ['ValuePerCount_mean', 'ValuePerCount_std', 'ValuePerCount_skews', 'ValuePerCount_kurtosis',
                    'trans_count_mean', 'trans_count_std', 'trans_count_skews', 'trans_count_kurtosis',
                    'trans_vol_mean', 'trans_vol_std', 'trans_vol_skews', 'trans_vol_kurtosis',
                    'WorstDelinqOrdinalAcrossLoans', 'AvgNumIrregPaymAcrossLoans', 'DaysFrmOnbrdToLnStrt']

        df_scaled, FlagsDict = scaling_function(df, FlagsDict, Dict_PastMaxes, Dict_MaxesAtLastBinEdge)
        # scale continuous variables and assign to new dataframe df_scaled,
        # and flag filled in case of any relevant df values

        
        df_enc = pd.get_dummies(columns=list_cat, data=df_scaled) # one-hot encode specific categorical columns
        # (list_cat) of df_scaled and assign resultant dataframe to df_enc

        # list of column names of all possible enterprisePartners, degrees, states, homeOwnershipTypes,
        #   vehicles, and jobTypes (i.e., list_cat columns) to test if present in dataframe
        # SY: Both 'degree_un' and 'degree_UNK' present?
        add_dummy = ['enterprisePartner_BAN', 'enterprisePartner_CHA', 'enterprisePartner_EAS',
                    'enterprisePartner_EKO', 'enterprisePartner_INS', 'enterprisePartner_MIN',
                    'enterprisePartner_RRS', 'enterprisePartner_SUG', 'enterprisePartner_UNK',
                    'degree_Graduation', 'degree_MBA', 'degree_MCA', 'degree_Other',
                    'degree_Post Graduation', 'degree_un', 'degree_UNK',
                    'state_AN', 'state_AP', 'state_AR', 'state_AS', 'state_BR',
                    'state_CH', 'state_CT', 'state_DH', 'state_DL', 'state_GA',
                    'state_GJ', 'state_HP', 'state_HR', 'state_JH', 'state_JK',
                    'state_KA', 'state_KL', 'state_LA', 'state_LD', 'state_MH',
                    'state_ML', 'state_MN', 'state_MP', 'state_MZ', 'state_NL',
                    'state_OR', 'state_PB', 'state_PY', 'state_RJ', 'state_SK',
                    'state_TG', 'state_TN', 'state_TR', 'state_UP', 'state_UT',
                    'state_WB', 'state_UNK',
                    'homeOwnershipType_Both', 'homeOwnershipType_Own',
                    'homeOwnershipType_Rent', 'homeOwnershipType_un',
                    'vehicle_2-Wheeler', 'vehicle_4-Wheeler',	'vehicle_Both',
                    'vehicle_No', 'vehicle_un',
                    'jobType_Both', 'jobType_No', 'jobType_Private',
                    'jobType_Public', 'jobType_un']

        # if any column from above add_dummy list is missing, then add it as 0-initialized one
        for col in add_dummy:  # loop through all possible column names
            if col not in df_enc.columns:  # if column not in df_enc dataframe
                df_enc[col] = 0  # create that column as a 0-initialized one
            # end of if col not in df_enc.columns
        # end of for col in add_dummy

    #    #following line is list of categorical columns complementary to list_cat
    #    un_list = ['recommended', 'purchasedInOneYear', 'children', 'maritalStatus',
    #               'gender', 'whatsApp', 'productType']
    #    #following line is list of categorical columns containing 'both' as a value
    #    both_list = ['vehicle', 'homeOwnershipType', 'jobType']

        # Begin encoding categorical columns that are complementary to list_cat

        # Map from Yes/un/no to 2/1/0
        un_dict = {'Yes': 2, 'un': 1, 'No': 0}
        df_enc['recommended'] = df_scaled.recommended.map(un_dict)
        df_enc['purchasedInOneYear'] = df_scaled.purchasedInOneYear.map(un_dict)
        df_enc['children'] = df_scaled.children.map(un_dict)
        df_enc['whatsApp'] = df_scaled.whatsApp.map(un_dict)

        # Map to 2/1/0
        df_enc['maritalStatus'] = df_scaled.maritalStatus.map({'Married': 2, 'un': 1, 'Single': 0})
        df_enc['gender'] = df_scaled.gender.map({'Male': 2, 'un': 1, 'Female': 0})

        # Map to 1/0
        df_enc['productType'] = df_scaled.productType.map({'Term Loan': 1, 'Daily Loan': 0})

        # End encoding categorical columns that are complementary to list_cat

        # List of categorical columns containing 'both' as a value: ['vehicle',
        #  'homeOwnershipType', 'jobType']

        # Begin assigning 1 to the relevant two column values for 'Both' value-containing columns

        df_enc.loc[df_enc['vehicle_Both'] == 1, 'vehicle_2-Wheeler'] = 1
        df_enc.loc[df_enc['vehicle_Both'] == 1, 'vehicle_4-Wheeler'] = 1

        df_enc.loc[df_enc['homeOwnershipType_Both'] == 1, 'homeOwnershipType_Own'] = 1
        df_enc.loc[df_enc['homeOwnershipType_Both'] == 1, 'homeOwnershipType_Rent'] = 1

        df_enc.loc[df_enc['jobType_Both'] == 1, 'jobType_Private'] = 1
        df_enc.loc[df_enc['jobType_Both'] == 1, 'jobType_Public'] = 1

        # End assigning 1 to the relevant two column values for 'Both' value-containing columns

        # Now remove those 'Both' value-containing columns
        df_enc = df_enc.drop(columns=['vehicle_Both', 'homeOwnershipType_Both', 'jobType_Both'], errors='ignore')

        # following line discretizes continuous variables, note that the FlagsDict parameter is
        #   that which was output from the scaling_function call further above
        df_bin, FlagsDict, FinalNumBins_Dict = discretization_function(df_enc, FlagsDict)

        # following line converts those continuous-variable discretizations into one-hot encodings
        df_enc = pd.get_dummies(data=df_bin, columns=list_num)

        # list of all continuous-variable-derived column names to test whether present in dataframe
        add_dummy = []
        # list of continuous-value columns to bin
        Cols_encoded = ['ValuePerCount_mean', 'ValuePerCount_std',
                        'ValuePerCount_skews', 'ValuePerCount_kurtosis',
                        'trans_count_mean', 'trans_count_std',
                        'trans_count_skews', 'trans_count_kurtosis',
                        'trans_vol_mean', 'trans_vol_std',
                        'trans_vol_skews', 'trans_vol_kurtosis',
                        'AvgNumIrregPaymAcrossLoans', 'DaysFrmOnbrdToLnStrt',
                        'WorstDelinqOrdinalAcrossLoans']
        for Col_encoded in Cols_encoded:
            for ii in range(FinalNumBins_Dict[Col_encoded]):
                add_dummy.append(Col_encoded + '_' + format(int(round(float(ii))), '01') + '.0')
        # end of for Col_encoded in Cols_encoded

        # if any column from above add_dummy list is missing, then add it as 0-initialized one
        for col in add_dummy:  # loop through all possible column names
            if col not in df_enc.columns:  # if column not in df_enc dataframe
                df_enc[col] = 0  # create that column as a 0-initialized one
            # end of if col not in df_enc.columns
        # end of for col in add_dummy

        return df_enc, FlagsDict

    # end of def data_encoding(df, FlagsDict, Dict_PastMaxes, Dict_MaxesAtLastBinEdge)

    # Function for credit scoring system, model probabilities and error strings


    def credit_scoring_model(ThisX, ThisDf, delinquent_model, default_model, writeoff_model, ThisFlagsDict):

        p_delinquent = delinquent_model.predict_proba(ThisX)
        p_default = default_model.predict_proba(ThisX)
        p_writeoff = writeoff_model.predict_proba(ThisX)

        ThisDf['p_delinquent'] = 0.0
        ThisDf['p_default'] = 0.0
        ThisDf['p_writeoff'] = 0.0
        for i in range(ThisX.shape[0]):
            ThisDf['p_delinquent'][i] = p_delinquent[i][1]
            ThisDf['p_default'][i] = p_default[i][1]
            ThisDf['p_writeoff'][i] = p_writeoff[i][1]
        # end of for i in range(ThisX.shape[0])

        ThisDf['p_total'] = (1 * ThisDf['p_delinquent'] + 3 * ThisDf['p_default'] + 6 * ThisDf['p_writeoff'])/10

        ThisDf['TRUEscore_v2.5'] = 0.0

        ThisDf.loc[                               ThisDf['p_total'] <= 0.035 , 'TRUEscore_v2.5'] = 1000 - 200 *  ThisDf.loc[                               ThisDf['p_total'] <= 0.035 , 'p_total']          / 0.035
        ThisDf.loc[(ThisDf['p_total'] > 0.035) & (ThisDf['p_total'] <= 0.1  ), 'TRUEscore_v2.5'] =  800 - 200 * (ThisDf.loc[(ThisDf['p_total'] > 0.035) & (ThisDf['p_total'] <= 0.1  ), 'p_total'] - 0.035) / (0.1 - 0.035)
        ThisDf.loc[(ThisDf['p_total'] > 0.1  ) & (ThisDf['p_total'] <= 0.3  ), 'TRUEscore_v2.5'] =  600 - 200 * (ThisDf.loc[(ThisDf['p_total'] > 0.1  ) & (ThisDf['p_total'] <= 0.3  ), 'p_total'] - 0.1  ) / (0.3 - 0.1  )
        ThisDf.loc[(ThisDf['p_total'] > 0.3  ) & (ThisDf['p_total'] <= 0.6  ), 'TRUEscore_v2.5'] =  400 - 200 * (ThisDf.loc[(ThisDf['p_total'] > 0.3  ) & (ThisDf['p_total'] <= 0.6  ), 'p_total'] - 0.3  ) / (0.6 - 0.3  )
        ThisDf.loc[ ThisDf['p_total'] > 0.6                                  , 'TRUEscore_v2.5'] =  200 - 200 * (ThisDf.loc[ ThisDf['p_total'] > 0.6                                  , 'p_total'] - 0.6  ) / (1   - 0.6  )

        ThisDf['TRUEscore_v2.5'] = round(ThisDf['TRUEscore_v2.5'])

        ThisDf['ErrorString'] = ''
        for i in range(ThisX.shape[0]):

            ThisDf['ErrorString'][i] = ThisDf['ErrorString'][i] + format(ThisFlagsDict['ValuePerCount_mean_outofscalerange'][i], '01')
            ThisDf['ErrorString'][i] = ThisDf['ErrorString'][i] + format(ThisFlagsDict['ValuePerCount_std_outofscalerange'][i], '01')
            ThisDf['ErrorString'][i] = ThisDf['ErrorString'][i] + format(ThisFlagsDict['ValuePerCount_skews_outofscalerange'][i], '01')
            ThisDf['ErrorString'][i] = ThisDf['ErrorString'][i] + format(ThisFlagsDict['ValuePerCount_kurtosis_outofscalerange'][i], '01')
            ThisDf['ErrorString'][i] = ThisDf['ErrorString'][i] + format(ThisFlagsDict['trans_count_mean_outofscalerange'][i], '01')
            ThisDf['ErrorString'][i] = ThisDf['ErrorString'][i] + format(ThisFlagsDict['trans_count_std_outofscalerange'][i], '01')
            ThisDf['ErrorString'][i] = ThisDf['ErrorString'][i] + format(ThisFlagsDict['trans_count_skews_outofscalerange'][i], '01')
            ThisDf['ErrorString'][i] = ThisDf['ErrorString'][i] + format(ThisFlagsDict['trans_count_kurtosis_outofscalerange'][i], '01')
            ThisDf['ErrorString'][i] = ThisDf['ErrorString'][i] + format(ThisFlagsDict['trans_vol_mean_outofscalerange'][i], '01')
            ThisDf['ErrorString'][i] = ThisDf['ErrorString'][i] + format(ThisFlagsDict['trans_vol_std_outofscalerange'][i], '01')
            ThisDf['ErrorString'][i] = ThisDf['ErrorString'][i] + format(ThisFlagsDict['trans_vol_skews_outofscalerange'][i], '01')
            ThisDf['ErrorString'][i] = ThisDf['ErrorString'][i] + format(ThisFlagsDict['trans_vol_kurtosis_outofscalerange'][i], '01')

            ThisDf['ErrorString'][i] = ThisDf['ErrorString'][i] + format(ThisFlagsDict['ValuePerCount_mean_outofbinrange'][i], '01')
            ThisDf['ErrorString'][i] = ThisDf['ErrorString'][i] + format(ThisFlagsDict['ValuePerCount_std_outofbinrange'][i], '01')
            ThisDf['ErrorString'][i] = ThisDf['ErrorString'][i] + format(ThisFlagsDict['ValuePerCount_skews_outofbinrange'][i], '01')
            ThisDf['ErrorString'][i] = ThisDf['ErrorString'][i] + format(ThisFlagsDict['ValuePerCount_kurtosis_outofbinrange'][i], '01')
            ThisDf['ErrorString'][i] = ThisDf['ErrorString'][i] + format(ThisFlagsDict['trans_count_mean_outofbinrange'][i], '01')
            ThisDf['ErrorString'][i] = ThisDf['ErrorString'][i] + format(ThisFlagsDict['trans_count_std_outofbinrange'][i], '01')
            ThisDf['ErrorString'][i] = ThisDf['ErrorString'][i] + format(ThisFlagsDict['trans_count_skews_outofbinrange'][i], '01')
            ThisDf['ErrorString'][i] = ThisDf['ErrorString'][i] + format(ThisFlagsDict['trans_count_kurtosis_outofbinrange'][i], '01')
            ThisDf['ErrorString'][i] = ThisDf['ErrorString'][i] + format(ThisFlagsDict['trans_vol_mean_outofbinrange'][i], '01')
            ThisDf['ErrorString'][i] = ThisDf['ErrorString'][i] + format(ThisFlagsDict['trans_vol_std_outofbinrange'][i], '01')
            ThisDf['ErrorString'][i] = ThisDf['ErrorString'][i] + format(ThisFlagsDict['trans_vol_skews_outofbinrange'][i], '01')
            ThisDf['ErrorString'][i] = ThisDf['ErrorString'][i] + format(ThisFlagsDict['trans_vol_kurtosis_outofbinrange'][i], '01')

            ThisDf['ErrorString'][i] = ThisDf['ErrorString'][i] + format(ThisFlagsDict['NewUnknown_enterprisePartner'][i], '01')
            ThisDf['ErrorString'][i] = ThisDf['ErrorString'][i] + format(ThisFlagsDict['NewUnknown_children'][i], '01')
            ThisDf['ErrorString'][i] = ThisDf['ErrorString'][i] + format(ThisFlagsDict['NewUnknown_maritalStatus'][i], '01')
            ThisDf['ErrorString'][i] = ThisDf['ErrorString'][i] + format(ThisFlagsDict['NewUnknown_gender'][i], '01')
            ThisDf['ErrorString'][i] = ThisDf['ErrorString'][i] + format(ThisFlagsDict['NewUnknown_whatsApp'][i], '01')
            ThisDf['ErrorString'][i] = ThisDf['ErrorString'][i] + format(ThisFlagsDict['NewUnknown_purchasedInOneYear'][i], '01')
            ThisDf['ErrorString'][i] = ThisDf['ErrorString'][i] + format(ThisFlagsDict['NewUnknown_recommended'][i], '01')
            ThisDf['ErrorString'][i] = ThisDf['ErrorString'][i] + format(ThisFlagsDict['NewUnknown_homeOwnershipType'][i], '01')
            ThisDf['ErrorString'][i] = ThisDf['ErrorString'][i] + format(ThisFlagsDict['NewUnknown_vehicle'][i], '01')
            ThisDf['ErrorString'][i] = ThisDf['ErrorString'][i] + format(ThisFlagsDict['NewUnknown_degree'][i], '01')
            ThisDf['ErrorString'][i] = ThisDf['ErrorString'][i] + format(ThisFlagsDict['NewUnknown_jobType'][i], '01')
            ThisDf['ErrorString'][i] = ThisDf['ErrorString'][i] + format(ThisFlagsDict['NewUnknown_state'][i], '01')

            ThisDf['ErrorString'][i] = ThisDf['ErrorString'][i] + format(ThisFlagsDict['AvgNumIrregPaymAcrossLoans_PastMaxCrossed'][i], '01')
            ThisDf['ErrorString'][i] = ThisDf['ErrorString'][i] + format(ThisFlagsDict['DaysFrmOnbrdToLnStrt_PastMaxCrossed'][i], '01')

        # end of for i in range(ThisX.shape[0])

        return ThisDf

    # end of def credit_scoring_model(ThisX, ThisDf, delinquent_model, default_model, writeoff_model, ThisFlagsDict)

    # df = pd.read_json('test.json')
    # lines = sys.stdin
    # aa = json.load(lines)
    # df = pd.DataFrame(aa)
    # fl=open('test.json')
    # flString=json.dumps(fl)
    # aa=json.loads(json.dumps(request.get_json()))
    # print(request.get_json())
    a=json.loads(json.dumps(request.get_json()))
    print(len(a))
    df = pd.DataFrame(json.loads(json.dumps(request.get_json())))
    # df = pd.DataFrame(json.loads(fl.read()))

    df.replace({'enterprisePartner': {'Pay': 'MIN', 
                                    'Eko': 'EKO',
                                    'MPay': 'CHA',
                                    'PEassy': 'RRS',
                                    'PWorld': 'SUG',
                                    'ENAC': 'ENA',
                                    'BNKIT': 'BAN',
                                    'INMD': 'INS'}},
            inplace=True)

    df, df_ForTRUEscorev1p1 = feature_engineering(df)
    # This adds to df the statistics columns
    #  'trans_count_<statistic>', 'trans_vol_<statistic>' and
    #  'ValuePerCount_<statistic>', and other columns
    #  'AvgNumIrregPaymAcrossLoans', 'WorstDelinqOrdinalAcrossLoans'
    #  and 'DaysFrmOnbrdToLnStrt',
    #  and removes columns 'activation_date', 'tvol_Month[01-12]',
    #  'tcnt_Month[01-12]', 'delinquencyString', 'loanStartDate',
    #  and 'onboardedDate'
    #  Also, this creates a new dataframe df_ForTRUEscorev1p1 containing the
    #  TRUEscore calculation of v1.1

    Dict_PastMaxes = {'trans_count_mean': 18993.916666666668,
                    'trans_count_std': 12197.399318661703,
                    'trans_count_skews': 3.464101615137763,
                    'trans_count_kurtosis': 12.00000000000004,
                    'trans_vol_mean': 38505614.505833335,
                    'trans_vol_std': 28111541.323776342,
                    'trans_vol_skews': 3.464101615137758,
                    'trans_vol_kurtosis': 12.000000000000016,
                    'ValuePerCount_mean': 44959.60315605713,
                    'ValuePerCount_std': 34802.0916842266,
                    'ValuePerCount_skews': 3.4641016151377575,
                    'ValuePerCount_kurtosis': 12.000000000000016,
                    'AvgNumIrregPaymAcrossLoans': 78.0,
                    'DaysFrmOnbrdToLnStrt': 3525.0}
    Dict_MaxesAtLastBinEdge = {'AvgNumIrregPaymAcrossLoans': 80.,
                            'DaysFrmOnbrdToLnStrt': 365*5 + 1.0}

    FlagsDict = {}
    df_enc, FlagsDict = data_encoding(df, FlagsDict, Dict_PastMaxes, Dict_MaxesAtLastBinEdge)
    # gives encoded dataframe (one-hot, sequential, and 'Both' columns) corresponding to df,
    # and FlagsDict indicating whether any out-of-range,  out-of-statistics-range,
    # or new unknown values exist

    X = df_enc.drop(columns=['pan_no', 'partnerMerchantCode', 'merchantType'], axis=1)

    X = X.sort_index(axis=1)  # sort per alphabetical order of column names?

    #print(len(list(X.columns)))
    #print(list(X.columns))
    # NOW, list(X.columns) should give the following order of 194 columns:
    #  ['AvgNumIrregPaymAcrossLoans_0.0', 'AvgNumIrregPaymAcrossLoans_1.0', 'AvgNumIrregPaymAcrossLoans_2.0',
    #   'DaysFrmOnbrdToLnStrt_0.0', 'DaysFrmOnbrdToLnStrt_1.0', 'DaysFrmOnbrdToLnStrt_2.0',
    #   'DaysFrmOnbrdToLnStrt_3.0', 'DaysFrmOnbrdToLnStrt_4.0', 'DaysFrmOnbrdToLnStrt_5.0',
    #   'ValuePerCount_kurtosis_0.0', 'ValuePerCount_kurtosis_1.0', 'ValuePerCount_kurtosis_2.0',
    #   'ValuePerCount_kurtosis_3.0', 'ValuePerCount_kurtosis_4.0', 'ValuePerCount_kurtosis_5.0',
    #   'ValuePerCount_kurtosis_6.0', 'ValuePerCount_kurtosis_7.0', 'ValuePerCount_kurtosis_8.0',
    #   'ValuePerCount_kurtosis_9.0',
    #   'ValuePerCount_mean_0.0', 'ValuePerCount_mean_1.0', 'ValuePerCount_mean_2.0',
    #   'ValuePerCount_mean_3.0', 'ValuePerCount_mean_4.0', 'ValuePerCount_mean_5.0',
    #   'ValuePerCount_mean_6.0', 'ValuePerCount_mean_7.0', 'ValuePerCount_mean_8.0',
    #   'ValuePerCount_skews_0.0', 'ValuePerCount_skews_1.0', 'ValuePerCount_skews_2.0',
    #   'ValuePerCount_skews_3.0', 'ValuePerCount_skews_4.0', 'ValuePerCount_skews_5.0',
    #   'ValuePerCount_skews_6.0', 'ValuePerCount_skews_7.0', 'ValuePerCount_skews_8.0',
    #   'ValuePerCount_skews_9.0',
    #   'ValuePerCount_std_0.0', 'ValuePerCount_std_1.0', 'ValuePerCount_std_2.0',
    #   'ValuePerCount_std_3.0', 'ValuePerCount_std_4.0', 'ValuePerCount_std_5.0',
    #   'ValuePerCount_std_6.0', 'ValuePerCount_std_7.0', 'ValuePerCount_std_8.0',
    #   'WorstDelinqOrdinalAcrossLoans_0.0', 'WorstDelinqOrdinalAcrossLoans_1.0', 'WorstDelinqOrdinalAcrossLoans_2.0',
    #   'WorstDelinqOrdinalAcrossLoans_3.0', 'WorstDelinqOrdinalAcrossLoans_4.0', 'WorstDelinqOrdinalAcrossLoans_5.0',
    #   'WorstDelinqOrdinalAcrossLoans_6.0',
    #   'children',
    #   'degree_Graduation', 'degree_MBA', 'degree_MCA', 'degree_Other',
    #   'degree_Post Graduation', 'degree_UNK', 'degree_un',
    #   'enterprisePartner_BAN', 'enterprisePartner_CHA', 'enterprisePartner_EAS', 'enterprisePartner_EKO',
    #   'enterprisePartner_INS', 'enterprisePartner_MIN', 'enterprisePartner_RRS', 'enterprisePartner_SUG',
    #   'enterprisePartner_UNK',
    #   'gender',
    #   'homeOwnershipType_Own', 'homeOwnershipType_Rent', 'homeOwnershipType_un',
    #   'jobType_No', 'jobType_Private', 'jobType_Public', 'jobType_un',
    #   'maritalStatus', 'productType', 'purchasedInOneYear', 'recommended',
    #   'state_AN', 'state_AP', 'state_AR', 'state_AS', 'state_BR', 'state_CH', 'state_CT', 'state_DH',
    #   'state_DL', 'state_GA', 'state_GJ', 'state_HP', 'state_HR', 'state_JH', 'state_JK', 'state_KA',
    #   'state_KL', 'state_LA', 'state_LD', 'state_MH', 'state_ML', 'state_MN', 'state_MP', 'state_MZ',
    #   'state_NL', 'state_OR', 'state_PB', 'state_PY', 'state_RJ', 'state_SK', 'state_TG', 'state_TN',
    #   'state_TR', 'state_UNK', 'state_UP', 'state_UT', 'state_WB',
    #   'trans_count_kurtosis_0.0', 'trans_count_kurtosis_1.0', 'trans_count_kurtosis_2.0',
    #   'trans_count_kurtosis_3.0', 'trans_count_kurtosis_4.0', 'trans_count_kurtosis_5.0',
    #   'trans_count_kurtosis_6.0', 'trans_count_kurtosis_7.0', 'trans_count_kurtosis_8.0',
    #   'trans_count_mean_0.0', 'trans_count_mean_1.0', 'trans_count_mean_2.0', 'trans_count_mean_3.0',
    #   'trans_count_mean_4.0', 'trans_count_mean_5.0', 'trans_count_mean_6.0', 'trans_count_mean_7.0',
    #   'trans_count_mean_8.0',
    #   'trans_count_skews_0.0', 'trans_count_skews_1.0', 'trans_count_skews_2.0', 'trans_count_skews_3.0',
    #   'trans_count_skews_4.0', 'trans_count_skews_5.0', 'trans_count_skews_6.0', 'trans_count_skews_7.0',
    #   'trans_count_skews_8.0', 'trans_count_skews_9.0',
    #   'trans_count_std_0.0', 'trans_count_std_1.0', 'trans_count_std_2.0', 'trans_count_std_3.0',
    #   'trans_count_std_4.0', 'trans_count_std_5.0', 'trans_count_std_6.0', 'trans_count_std_7.0',
    #   'trans_count_std_8.0',
    #   'trans_vol_kurtosis_0.0', 'trans_vol_kurtosis_1.0', 'trans_vol_kurtosis_2.0', 'trans_vol_kurtosis_3.0',
    #   'trans_vol_kurtosis_4.0', 'trans_vol_kurtosis_5.0', 'trans_vol_kurtosis_6.0', 'trans_vol_kurtosis_7.0',
    #   'trans_vol_kurtosis_8.0',
    #   'trans_vol_mean_0.0', 'trans_vol_mean_1.0', 'trans_vol_mean_2.0', 'trans_vol_mean_3.0',
    #   'trans_vol_mean_4.0', 'trans_vol_mean_5.0', 'trans_vol_mean_6.0', 'trans_vol_mean_7.0',
    #   'trans_vol_mean_8.0',
    #   'trans_vol_skews_0.0', 'trans_vol_skews_1.0', 'trans_vol_skews_2.0', 'trans_vol_skews_3.0',
    #   'trans_vol_skews_4.0', 'trans_vol_skews_5.0', 'trans_vol_skews_6.0', 'trans_vol_skews_7.0',
    #   'trans_vol_skews_8.0',
    #   'trans_vol_std_0.0', 'trans_vol_std_1.0', 'trans_vol_std_2.0', 'trans_vol_std_3.0',
    #   'trans_vol_std_4.0', 'trans_vol_std_5.0', 'trans_vol_std_6.0', 'trans_vol_std_7.0',
    #   'trans_vol_std_8.0',
    #   'vehicle_2-Wheeler', 'vehicle_4-Wheeler', 'vehicle_No', 'vehicle_un', 'whatsApp']

    df_output = df_enc[['pan_no', 'partnerMerchantCode', 'merchantType']].copy()
    

    df_output = credit_scoring_model(X, df_output, delinquent_model, default_model, writeoff_model, FlagsDict)

    df_output['oldTRUEscore'] = df_ForTRUEscorev1p1['TRUEscore']
    df_output['amount'] = df_ForTRUEscorev1p1['raw2']

    output = df_output.to_json(orient='records')[1:-1].replace('},{', '} {')

    # print(output)
    end=datetime.datetime.now()
    print(end)
    time_diff=end-start
    print(time_diff)
    print(time_diff.total_seconds())
    # print(output)
    return output