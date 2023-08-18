import pandas_market_calendars as mcal
import pandas as pd 
import numpy as np
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error
from scipy.stats import ttest_1samp
from scipy.stats import norm
import string
import importlib.resources as pkg_resources


def open_example_data():
    data_path = pkg_resources.open_text('event_study_toolkit', 'data_panel.csv').name
    data =  pd.read_csv(data_path, index_col=False)
    data['date'] = data['date'].astype('datetime64[ns]')
    return data


def open_example_events():
    data_path = pkg_resources.open_text('event_study_toolkit', 'event_panel.csv').name
    events = pd.read_csv(data_path, index_col=False)
    events['EVT_DATE'] = events['EVT_DATE'].astype('datetime64[ns]')
    return events

class eventstudy:
    """
    The eventstudy class performs calculations and statistical operations designed to assist researchers 
    when conducting an event study

    Attributes:
    -----------
    estperiod: integer
        The estimation period of an eventstudy
    gap: integer
        The gap between the end of the estimation and the start of the event
    start: integer
        The start of an event period
    end: integer
        The end of an event period
    events: Pandas Dataframe
        The event data that includes the security identifier, the groups the security belongs to, as well as, the event dates
    cal: pandas_market_calendars object
        Creates the desired trading calendar with days the market is open for the requested exchange
    data: Pandas Dataframe
        The security data that includes the security identifier, as well as various other information (i.e., return) for a particular trading day
    dataFrame: Pandas Dataframe
        Events, calendars, and data in one panel/frame
    
    """
    def __init__(self, estperiod, gap, start, end, data, events, unique_id = 'permno', calType = 'NYSE', groups = None):
        """
        Constructs a new instance of the eventstudy class
        
        Parameters:
        -----------
        estperiod: integer
            The estimation period of an eventstudy
        gap: integer
            The gap between the end of the estimation and the start of the event
        start: integer
            The start of an event period
        end: integer
            The end of an event period
        data: Pandas Dataframe
            The security data that includes the security identifier, as well as various other information (i.e., return) for a particular trading day
        events: Pandas Dataframe
            The event data that includes the security identifier, the groups the security belongs to, as well as, the event dates
        calType: string
            Type of trading calendar that is required to be used for the desired eventstudy, default is 'NYSE'
        unique_id: string
            Unique identifier for a security, default is 'permno'
        group: list
            List of strings where each string describes a column representing a group
        """
        if estperiod < 0:
            raise ValueError('The estperiod/Estimation Period is not valid, must be positive')
        self.estperiod = estperiod
        self.gap = gap
        self.start = start
        self.end = end
        self.events = events
        self.unique_id = unique_id
        self.cal = self.getTradingCalendar(calType)
        self.data = data
        self.dataFrame = self.getDataFrame()
        if groups != None:
            checker = 0
            for group in groups:
                if group not in self.dataFrame.columns:
                    checker += 1
            
            if checker == 0:
                self.groups = groups
            else:
                raise ValueError("The provided group(s) do not exist in inputted data")    


    def getTradingCalendar(self, calType):
        """
        Retrieves trading calendar

        Returns:
        --------
        Pandas dataframe
            The trading calendar created from generateTradingCalendar()
        """

        typelist = mcal.get_calendar_names()
        if calType not in typelist:
            raise ValueError(calType + " not in list of acceptable market calendars. See list for all available calendars: \n" + str(typelist))
            #print("See list for all available calendars: \n" + str(typelist))
        
        # object of type <pandas_market_calendars.exchange_calendar_nyse.NYSEExchangeCalendar at 0x7f591827ae50>, nyse in case of calType = NYSE        
        calObj = mcal.get_calendar(calType) 
        
        calendar = calObj.schedule(start_date='1900-01-01', end_date='2023-07-10') # returns pandas.core.frame.DataFrame
        # default index are the dates, make new column using index named 'date'
        calendar['date'] = calendar.index 
        #generate the trading calendar and return it 
        return self.generateTradingCalendar(calendar)
    
    def generateTradingCalendar(self, calendar):
        """
        Retrieves trading calendar

        Returns:
        --------
        Pandas dataframe
            The trading calendar
        """
        # estimation period has the ability to start at the beginning of the calendar until the there are not enough days to run an event study based on the given parameters
        estStart = calendar['date'][0:(len(calendar.index) - 1) - (self.estperiod + self.gap - self.start + self.end)] 
        # estimation period has the ability to end from the beginning of the calendar + the est_period until there are still enough days for the gap and event period to occur
        estEnd = calendar['date'][self.estperiod - 1:len(calendar.index) - (self.gap - self.start + self.end + 1)]
        # event can start after the estperiod and gap have occurred until there are enough days for an event to go to completion 
        evtStart = calendar['date'][self.estperiod+self.gap:len(calendar.index) - (self.start + self.end)]
        # event date ...
        evtDate = calendar['date'][self.estperiod + self.gap - self.start:len(calendar.index) - self.end - 1]
        # event end ...
        evtEnd = calendar['date'][self.estperiod + self.gap - self.start + self.end:len(calendar.index) - 1]

        estStart.reset_index(drop=True, inplace=True)
        estEnd.reset_index(drop=True, inplace=True)
        evtStart.reset_index(drop=True, inplace=True)
        evtDate.reset_index(drop=True, inplace=True)
        evtEnd.reset_index(drop=True, inplace=True)

        tradingCalendar = pd.concat([estStart,estEnd,evtStart,evtDate,evtEnd],axis=1)
        tradingCalendar.columns = ['EST_START','EST_END','EVT_START','EVT_DATE','EVT_END']
        tradingCalendar = tradingCalendar.dropna()
        return tradingCalendar
    
    def getDataFrame(self):
        """
        Creates dataframe using events, security data, and calendar

        Returns:
        --------
        Pandas dataframe
            The dataframe with events, security data, and calendar
        """
        part1 = pd.merge(self.data, self.events, on = self.unique_id) # merge security data and event data on permno
        part1['EVT_DATE'] = part1['EVT_DATE'].astype('datetime64[ns]') # make EVT_DATE variable of type datetime64[ns]  
        self.cal['EVT_DATE'] = self.cal['EVT_DATE'].astype('datetime64[ns]') # make EVT_DATE variable of type datetime64[ns] 
        part2 = pd.merge(part1, self.cal, on = 'EVT_DATE') # merge security data and events with trading calendar on EVT_DATE
        return part2
    
    def createEstChunk(self):
        """
        Create chunks for estimation period, one 'chunk' for each permno

        Returns:
        --------
        Pandas DataFrameGroupBy
            The DataFrameGroupBy where the number of group members is equal to the number of permnos 
        """
        est_returns = self.dataFrame[(self.dataFrame['date'] >= self.dataFrame['EST_START']) & (self.dataFrame['date'] <= self.dataFrame['EST_END'])] #create data chunks using dates fall between and include the start and end of the estimation period 
        return est_returns.groupby(self.unique_id)
    
    def createEvtChunk(self):
        """
        Create chunks for event period, one 'chunk' for each permno

        Returns:
        --------
        Pandas DataFrameGroupBy
            The DataFrameGroupBy where the number of group members is equal to the number of permnos 
        """
        evt_returns = self.dataFrame[(self.dataFrame['date'] >= self.dataFrame['EVT_START']) & (self.dataFrame['date'] <= self.dataFrame['EVT_END'])] #create data chunks using dates fall between and include the start and end of the event period 
        return evt_returns.groupby(self.unique_id)
    
    def checkStandardModelType(self, modelType):
        """
        Checks to see if inputted model is a standard model type, either 'market', 'famafrench', or 'capm'
        and then check if data has the appropriate column names

        Returns:
        --------
        Boolean
            True if model is a standard model with the expected column names, False if not
        """
        if modelType == 'market':
            if all([item in self.dataFrame.columns for item in ['ret_dlst_adj','vwretd']]): # checks to see if ['ret_dlst_adj','vwretd'] are column names that exist in the created dataFrame class attribute
                return True
            else:
                return False
        elif modelType == 'famafrench':
            if all([item in self.dataFrame.columns for item in ['ret_dlst_adj','mktrf_h15','smb','hml']]):
                return True
            else:
                return False
        elif modelType == 'capm':
            if all([item in self.dataFrame.columns for item in ['ret_dlst_adj','mktrf_h15','rf_h15']]):
                return True
            else:
                return False
        else:
            return False 
    
    def checkCustomModelType(self, modelType):
        """
        Parses ols string to get columns and returns true if provided columns exist in the data frame

        Returns:
        --------
        Boolean
            True if ols custom string is formatted correctly and column names in ols string exist in dataFrame attribute
            Otherwise, false
        """
        col_names = [] # place holder for resulting column names
        no_spaces = modelType.translate({ord(c): None for c in string.whitespace}) # Removes spaces in modelType string parameter
        response_var = no_spaces.partition("~")[0] # response variable is on the left side of the tilde (variable being predicted)
        regressors = no_spaces.partition("~")[2] # regressor(s) are on the right side of the tilde (variables which will receive coefficients in order to predict the response variable) 
        col_names.append(response_var) # append response var to colomn names variable
        col_names = col_names + (regressors.split("+")) # add regressor(s) to column names variable after removal of tilde and addition signs in preparation for condition checking
        if all([item in self.dataFrame.columns for item in col_names]): # check to see if these columns exist within the dataFrame attribute
            return True
        else:
            return False

    def runModel(self, modelType):
        """
        Uses ordinarary least squares function from statsmodels.formula.api to get model parameters

        Returns:
        --------
        Pandas Dataframe
            Parameters (coefficients and intercept) of provided ols model for each permno (security)

        """
        est_chunks = self.createEstChunk() # retrieve estimation chunks to create model
        

        if modelType == 'market':
            try:
                est_params = est_chunks.apply(lambda x: smf.ols('ret_dlst_adj ~ vwretd', data=x).fit().params) # run market model, catch error and print out error message with instructions on how to proceed in case of failure
            except:
                print("To use the 'market' model, make sure that the data has columns named ret_dlst_adj and vwretd")
                print("If not, please rename the columns or enter a string like so: 'ret_dlst_adj ~ vwretd'")

        elif modelType == 'famafrench':
            try:
                est_params = est_chunks.apply(lambda x: smf.ols('ret_dlst_adj~mktrf_h15+smb+hml', data=x).fit().params)
            except:
                print("To use the 'famafrench' model, make sure that the data has columns named ret_dlst_adj, smb, hml and mktrf_h15")
                print("If not, please rename the columns or enter a string like so: 'ret_dlst_adj~mktrf_h15+smb+hml'")

        elif modelType == 'capm':
            try:
                est_params = est_chunks.apply(lambda x: smf.ols('ret_dlst_adj~rf_h15+mktrf_h15', data=x).fit().params)
            except:
                print("To use the 'capm' model, make sure that the data has columns named ret_dlst_adj, rf_h15 and mktrf_h15")
                print("If not, please rename the columns or enter a custom model like so: ret_dlst_adj~rf_h15+mktrf_h15")

        else:

            if self.checkCustomModelType(modelType) == True:
                try:
                    est_params = est_chunks.apply(lambda x: smf.ols(modelType, data=x).fit().params)
                except:
                    print("Make sure your columns in: '" + modelType + "' exist")
            else:
                print("Make sure your columns in: '" + modelType + "' exist")
        
        return est_params
    
    def fitModel(self, modelType):
        """
        Uses ordinarary least squares function from statsmodels.formula.api to get fit object

        Returns:
        --------
        Fit object from statsmodels.formula.api
            Fit object of provided ols model for each permno (security)

        """
        est_chunks = self.createEstChunk() # retrieve estimation chunks to create model

        if modelType == 'market':
            try:
                est_fit = est_chunks.apply(lambda x: smf.ols('ret_dlst_adj ~ vwretd', data=x).fit()) # fit market model, catch error and print out error message with instructions on how to proceed in case of failure
            except:
                print("To use the 'market' model, make sure that the data has columns named ret_dlst_adj and vwretd")
                print("If not, please rename the columns or enter a string like so: 'ret_dlst_adj ~ vwretd'")

        elif modelType == 'famafrench':
            try:
                est_fit = est_chunks.apply(lambda x: smf.ols('ret_dlst_adj~mktrf_h15+smb+hml', data=x).fit())
            except:
                print("To use the 'famafrench' model, make sure that the data has columns named ret_dlst_adj, smb, hml and mktrf_h15")
                print("If not, please rename the columns or enter a string like so: 'ret_dlst_adj~mktrf_h15+smb+hml'")

        elif modelType == 'capm':
            try:
                est_fit = est_chunks.apply(lambda x: smf.ols('ret_dlst_adj~rf_h15+mktrf_h15', data=x).fit())
            except:
                print("To use the 'capm' model, make sure that the data has columns named ret_dlst_adj, rf_h15 and mktrf_h15")
                print("If not, please rename the columns or enter a custom model like so: ret_dlst_adj~rf_h15+mktrf_h15")
                est_fit = 0
        else:

            if self.checkCustomModelType(modelType) == True:
                try:
                    est_fit = est_chunks.apply(lambda x: smf.ols(modelType, data=x).fit())
                except:
                    print("Make sure your columns in: '" + modelType + "' exist")

            else:
                print("Make sure your columns in: '" + modelType + "' exist")

        return est_fit
    
    def getResiduals(self, x, est_params, est_chunk):
        """
        Calculates residuals using the smf fitobject (est_params) for each permno's estimation chunk (est_chunk) where x is the permno

        Returns:
        --------
        Pandas dataframe
            Dataframe includes permno, group data (CCAR), date, n (number of residuals or length of estimation period), k (number of parameters in ols), and residual
        """
        p = est_params[x] # get parameters of model fit object
        d = est_chunk.get_group(x) # get requisite group from est_chunk
        df = pd.DataFrame({
            self.unique_id: int(x), # unique security identifier
            # 'CCAR': d['CCAR'], # group identifier, neeed to add flexibility to this
            'date': d['date'], # date value for data
            'n': len(p.resid), # number of residuals (equal to length of estimation period)
            'k': len(p.params), # number of regressors 
            'e': p.resid # residual or error, difference between the estimated return and the actual return during the "training" (or estimation period) of the model 
        })
        for group in self.groups:
            df[group] = d[group]
        return df
    
    def getModelErrors(self, modelType):
        """
        Calculates sigmas (root mean squared error) and model errors 

        Returns:
        --------
        2 Pandas dataframes
            sigmas is a select grouping of model errors, e_sign (1 if error/residual is positive, 0 if negative), sigma (rmse), sar (standardized abnormal return)
        """
        est_chunks = self.createEstChunk() # retrieve estimation chunks 
        est_fit = self.fitModel(modelType) # get fit objects of model 
        model_errors = pd.concat([self.getResiduals(name, est_fit, est_chunks) for name in est_fit.keys()]) # get residuals for all securities
        model_errors['model'] = modelType # add model info to dataframe
        model_errors['e_sign'] = np.where(model_errors['e'] > 0, 1, 0) # create column for sign of residual/error, 0 if negative, 1 if positive
        # root mean squared error
        # A way to measure the performance of a regression model
        # Measures the average difference between values predicted by a model and the actual values, provides an estimation of how well the model is able to predict the target values (accuracy)
        model_errors['sigma'] = model_errors.groupby(self.unique_id)['e'].transform(lambda x: np.sqrt((x**2).sum() / (len(x) - len(est_fit[x.name].params)))) # root mean squared error calculation
        model_errors['sar'] = model_errors['e'] / model_errors['sigma'] # ???????????? 
        group_columns = [self.unique_id, 'sigma'] + self.groups + ['sar']
        sigmas = model_errors.groupby(group_columns)['e_sign'].mean().reset_index().rename(columns={'e_sign': 'e_est_pos'}) # create sigmas dataframe, GRP FLEXIBILITY
        return sigmas, model_errors

    def predictReturn(self, x, est_params, evt_chunk):
        """
        Helper function for getPredictedReturns(), uses the fitted ordinary least squares model with data from the ols model to calculate
        what the expected/predicted return is for the event. There is a predicted return for each day in the event period

        Returns:
        --------
        Pandas dataframe
            Pandas dataframe with added column for predicted return

        """
        model = est_params[x] # get parameters of model fit object
        evt = evt_chunk.get_group(x).copy()  # get requisite group from evt_chunk
        if not evt.empty:
            r = model.predict(evt) # predict values during event period
            evt.loc[:, 'predicted_return'] = r.values  # new column for predicted returns
            return evt
        else:
            return None
    
    def getPredictedReturns(self, modelType):
        """
        Aggragated predicted returns for all permnos in the event study

        Returns:
        --------
        Pandas dataframe
            Entire dataFrame with added column for the predicted return
        """
        est_fit = self.fitModel(modelType) # get fit objects of model
        evt_chunks = self.createEvtChunk() # retrieve event chunks
        return pd.concat([self.predictReturn(name, est_fit, evt_chunks) for name in est_fit.keys()], ignore_index=True) # return concatenated dataframe with returns
    
    def getAbnormalReturns(self, modelType):
        """
        Calculates the abnormal returns, which is the actual return minus the predicted return from the model

        Returns:
        --------
        Pandas dataframe
           Entire dataFrame with added column for the abnormal return

        """
        abret = self.getPredictedReturns(modelType) # get predicted return from model
        if self.checkStandardModelType(modelType): # condition check for standard model
            abret['abret'] = abret['ret_dlst_adj'] - abret['predicted_return'] # Abnormal return = actual return - predicted return
        else:
            no_spaces = modelType.translate({ord(c): None for c in string.whitespace}) # else using custom model, parse custom model string to determine response var (return variable)
            response_var = no_spaces.partition("~")[0] # response var on left side of tilde
            abret['abret'] = abret[response_var] - abret['predicted_return'] # Abnormal return = actual return - predicted return
        return abret
    
    def getCARS(self,modelType):
        """
        Calculates the cumarlative abnormal return (CAR) for each permno, the CAR is the summation of all the abnormal returns.
        Also, calculates the standardized abnormal return, the scar_bmp, and the poscar (whether the CAR is positive (1) or negative (0))
        
        Returns:
        --------
        Pandas dataframe 
            Resulting dataframe has the following columns: permno, EVT_DATE, car, model, sigma,	CCAR, sar, e_est_pos, scar, scar_bmp, and poscar
        """
        # Step 1: Group and sum
        abret = self.getAbnormalReturns(modelType) # retrieve abnormal returns
        cars = abret.groupby([self.unique_id, "EVT_DATE"])['abret'].sum().reset_index(name='car') # group by unqiue identifier and event date in order to find summation of abnormal return with resulting sum being the cumalative abnormal (CAR)
        cars['model'] = modelType # include model type in dataframe

        # Step 2: Merge DataFrames
        sigmas, model_residual = self.getModelErrors(modelType) # get error data from estimation period (sigmas)
        cars = cars.merge(sigmas, on=[self.unique_id]) # merge sigmas dataframe with car dataframe on unique identifier
        e_est_pos_mean = cars.groupby(self.unique_id)['e_est_pos'].mean() # find the mean/average of the sign of the error from the estimation period
        cars = cars.drop('e_est_pos',axis=1) # drop column, don't need individual e_sign, just need the mean 
        cars = cars.drop('sar',axis=1) # drop column, don't think it is needed ??????????????
        cars.drop_duplicates(inplace = True) # no duplicates
        cars = cars.merge(e_est_pos_mean, on=[self.unique_id]) # merge cleaned data with cars

        # Step 3: Calculate new columns
        N = self.end - self.start + 1 # length of event period
        cars['scar'] = np.sqrt(1/N) * cars['car'] / cars['sigma'] # scar calculation, scar = standardized cumalative abnormal return
        cars['scar_bmp'] = (cars['car'] / cars['sigma']) / cars['scar'].std() # not sure what this does ???????????
        cars['poscar'] = np.where(cars['car'] > 0, 1, 0) # new column to hold whether cumalative abnormal return is positve (1) or negative (0)
        return cars
    
    def t_stat(self, series):
        """
        Uses scipy.stats to calculate 1 sample t-test

        Returns:
        --------
        Float
            t statistic resulting from 1 sample t-test 
        """
        return ttest_1samp(series, 0).statistic

    def getFullSampleTestStatistic(self,modelType):
        """
        Calculates various test statistics using entire results (ignores groups)

        Returns:
        --------
        Pandas Dataframe
            Resulting dataframe has the following columns: 
            model, car_mean, scar_mean, poscar_mean, poscar_cnt, e_est_pos, car_t, scar_t, tsign, tpatell, and gen_z
        """
        cars = self.getCARS(modelType) # retrieve cumalative abnormal returns
        cols = [self.unique_id,'model','EVT_DATE','car', 'sigma'] + self.groups + ['e_est_pos','scar','scar_bmp','poscar']
        cars = cars[cols].copy() # index columns that are needed
        cars.drop_duplicates(inplace = True) # drop duplicates
        evt_count = len(cars.index) # number of securities with events in the event study
        # Calculating necessary statistics
        car_mean = cars['car'].mean() # mean of cumalative abnormal return for all securities
        # car median
        scar_mean = cars['scar'].mean() # mean of standardized cumalative abnormal return for all securities
        # scar median
        poscar_mean = cars['poscar'].mean() # mean of positive cumalative abnormal return count for all securities
        poscar_cnt = cars['poscar'].sum() # count of positive cumalative abnormal returns over all securities

        e_est_pos = cars['e_est_pos'].mean() # mean of positive errors/residuals during estimation period all securities
        car_t = self.t_stat(cars['car']) # t test statistic for cumalative abnormal return
        scar_t = self.t_stat(cars['scar'])  # t test statistic for standardized cumalative abnormal return
        # scar bmp t 
        # Creating DataFrame
        stats = pd.DataFrame({'model': modelType, 'car_mean': car_mean, 'scar_mean': scar_mean, 'poscar_mean': poscar_mean, 'poscar_cnt': poscar_cnt, 'evt_count': evt_count, 'e_est_pos': e_est_pos, 'car_t': car_t, 'scar_t': scar_t}, index=[0])

        # Calculate new stats
        evt_count = len(cars)
        stats['tsign'] = (stats['poscar_mean']-0.5)/np.sqrt(0.25/evt_count) # sign test statistic
        stats['tpatell'] = stats['scar_mean']*np.sqrt(evt_count) # patell test statistic
        # Generalized Rank Z test

        # Generalized Sign Test ~ Cowan
        stats['gen_z'] = (stats['poscar_cnt'] - evt_count*stats['e_est_pos'])/np.sqrt(evt_count*stats['e_est_pos']*(1-stats['e_est_pos'])) # Generalized Sign Test statistic

        return stats
    
    def getGroupLevelTestStatistics(self, modelType, GRP):
        """
        Calculates various group level test statistics using the calculated results for each group 

        Returns:
        --------
        Pandas Dataframe
            Resulting dataframe has the following columns: 
            model, car_mean, scar_mean, poscar_mean, poscar_cnt, e_est_pos, car_t, scar_t, tsign, tpatell, and gen_z
        """ 
        if GRP not in self.groups:
            print('Group ' + GRP + " does not exist")
        cars = self.getCARS(modelType) # retrieve cumalative abnormal returns
        cars = cars[[self.unique_id,'model','EVT_DATE','car', 'sigma',GRP, 'e_est_pos','scar','scar_bmp','poscar']].copy() # index columns that are needed
        cars.drop_duplicates(inplace = True) # drop duplicates
        
        # Grouping by GRP and calculating required statistics
        stats_grp = cars.groupby(GRP).agg(
            car_mean = ('car', 'mean'), # mean of cumalative abnormal return for securities in a group
            car_median = ('car', 'median'), # median of cumalative abnormal return for securities in a group
            scar_mean = ('scar', 'mean'), # mean of standardized cumalative abnormal return for securities in a group
            scar_median = ('scar', 'median'), # median of standardized cumalative abnormal return for securities in a group
            poscar_mean = ('poscar', 'mean'),  # mean of positive cumalative abnormal return count for securities in a group
            poscar_cnt = ('poscar', 'sum'), # count of positive cumalative abnormal returns for securities in a group
            evt_count = (self.unique_id, 'count'),  # number of securities with events in a group
            e_est_pos = ('e_est_pos', 'mean'), # mean of positive errors/residuals during estimation period for securities in a group
            car_t = ('car', lambda x: ttest_1samp(x, 0).statistic), # t test statistic for cumalative abnormal return for a group
            scar_t = ('scar', lambda x: ttest_1samp(x, 0).statistic), # t test statistic for standardized cumalative abnormal return for a group
            #scar_bmp_t = ('scar_bmp', lambda x: ttest_1samp(x, 0).statistic)
        ).reset_index()
        stats_grp['model'] = modelType

        # Calculating new columns based on previously computed statistics
        stats_grp['tsign'] = (stats_grp['poscar_mean'] - 0.5) / np.sqrt((0.25 / stats_grp['evt_count']))  # sign test statistic
        stats_grp['tpatell'] = stats_grp['scar_mean'] * np.sqrt(stats_grp['evt_count'])  # patell test statistic
        stats_grp['gen_z'] = (stats_grp['poscar_cnt'] - stats_grp['evt_count'] * stats_grp['e_est_pos']) / \
                                np.sqrt(stats_grp['evt_count'] * stats_grp['e_est_pos'] * (1 - stats_grp['e_est_pos']))  # Generalized Sign Test statistic

        #stats_grp['wilcox_z'] = self.wilcox_rank_sum(self.getCARS(modelType), "scar", "CCAR")
        
        return stats_grp
    
    def getGRANK(self, modelType, GRP):
        """
        Calculates the Genaralized Rank Z Test as proposed by Kolari and Pynnonen

        Returns:
        --------
        Pandas Dataframe
            n rows where n is determined by group members and each row has a corresponding z test statistic
        """
        if GRP not in self.groups:
            print('Group ' + GRP + " does not exist")
        # get model errors from estimation period
        sigmas, me = self.getModelErrors(modelType)
        sar = me[['model', self.unique_id, 'date', GRP, 'sar']].copy()
        sar.rename(columns={'sar': 'GSAR'}, inplace=True)

        # get standardized cumulative returns in event period
        scar = self.getCARS(modelType)
        scar = scar[['model', self.unique_id, GRP, 'scar_bmp']].copy()
        # scar.rename(columns={'scar_bmp': 'GSAR'}, inplace=True)
        scar.drop_duplicates(inplace = True)
        scar['date'] = pd.Timestamp('9999-12-31')
        scar.rename(columns={'scar_bmp': 'GSAR'}, inplace=True) 

        # calculate the ranks
        GRANK = pd.concat([sar, scar])
        #print(GRANK)
        # rank/sort GRANK by model, permno, and GSAR
        GRANK = GRANK.sort_values(['model', self.unique_id, 'GSAR'])
        # create new column that contains rank
        GRANK['GRANK_rank'] = GRANK.groupby(['model', self.unique_id]).cumcount() + 1       
        GRANK['T'] = GRANK.groupby(['model', self.unique_id]).transform('size')
        GRANK['sd_factor'] = (GRANK['T'] - 1) / (GRANK['T'] + 1)
        GRANK['N'] = GRANK.groupby(['model', self.unique_id]).transform('size')
        GRANK['U'] = (GRANK['GRANK_rank'] / ((GRANK['N']) + 1)) - 0.5


        # calculate std. deviation
        GRANK_sd = GRANK[['model', GRP, self.unique_id, 'sd_factor']].drop_duplicates()
        GRANK_sd['N'] = GRANK_sd.groupby(['model', GRP]).transform('size')
        GRANK_sd = GRANK_sd.groupby(['model', 'CCAR', 'N']).sd_factor.sum().reset_index()
        GRANK_sd['U_sd'] = ((1 / (12 * (GRANK_sd['N'] ** 2))) * GRANK_sd['sd_factor']) ** 0.5

        # calculate test statistics
        GRANK_agg = GRANK.groupby(['model', 'date', 'CCAR']).U.mean().reset_index()
        GRANK_agg = GRANK_agg.rename(columns={'U': 'U_bar'})
        
        # 2. Merge GRANK_agg with a subset of GRANK_sd on 'model' and 'CCAR' columns.
        GRANK_agg = pd.merge(GRANK_agg, GRANK_sd[['model', GRP, 'U_sd']], on=['model', GRP], how='left')
 
        # 3. Create a new column 'GRANK_Z' by dividing 'U_bar' by 'U_sd'.
        GRANK_agg['GRANK_Z'] = GRANK_agg['U_bar'] / GRANK_agg['U_sd']
        GRANK_grp = GRANK_agg[GRANK_agg.date == pd.Timestamp('9999-12-31')][['model', GRP, 'GRANK_Z']]
        return GRANK_grp
    
    def getWilcoxon(self, modelType, GRP):
        """
        Gets the Wilcoxon signed-rank test proposed by Wilcoxon in 1945 

        Returns:
        --------
        Pandas Dataframe
            Dataframe has one row with following values: model type, test statistic from Wilcoxon test, as well as, the resulting 2 tailed p-value
        """
        if GRP not in self.groups:
            print('Group ' + GRP + " does not exist")
        wrank = self.wilcox_rank_sum(modelType, "scar", GRP)
        wrank_df = pd.DataFrame({
            'model': [modelType],
            'wilcox': [round(wrank['statistic'], 2)],
            'wilcox_p': [round(wrank['pval_2tail'], 8)]
        })

        return wrank_df
    
    def wilcox_rank_sum(self, modelType, var, GRP):
        """
        Helper function for getWilcoxon()

        Returns:
        --------
        dictionary
            Dictionary contains the following calculations for each group: rank sum, average rank, N (size of group), Wilcoxon test statistic, as well as, 1- and 2-tail p values
        """
        df = self.getCARS(modelType)

        # Remove NaNs and infinite values
        df = df.loc[df[var].notna() & np.isfinite(df[var]), [var, GRP]]

        # Sort by variable
        df = df.sort_values(by=var)

        # Rank by sequence, handling ties
        df['rank'] = df[var].rank(method='average')

        # Calculate rank-sum by group
        df_grouped = df.groupby(GRP).agg({'rank': 'sum', GRP: 'size'})

        # Rename the columns
        df_grouped.columns = ['rank_sum', 'N']

        # Calculate average rank
        df_grouped['avg_rank'] = df_grouped['rank_sum'] / df_grouped['N']

        # Order by N
        df_grouped = df_grouped.sort_values(by='N')

        # Z calculation
        N1 = df_grouped['N'].iloc[0]
        N2 = df_grouped['N'].iloc[1]
        rank_sum = df_grouped['rank_sum'].iloc[0]
        Z = (rank_sum - (N1 * (N1 + N2 + 1) / 2) + 0.5) / np.sqrt((N1 * N2 * (N1 + N2 + 1)) / 12)

        # p-values
        p1 = norm.cdf(Z)
        p2 = 2 * norm.sf(abs(Z))

        return {'rank_sums': df_grouped, 'statistic': Z, 'pval_1tail': p1, 'pval_2tail': p2}

     # def getSigmas(self, modelType):
    #     est_chunks = self.createEstChunk()
    #     est_fit = self.fitModel(modelType)
    #     return pd.concat([self.rmse(name, est_fit, est_chunks, modelType) for name in est_fit.keys()])


 # def rmse(self, x, est_params, est_chunk, modelType):
    #     if self.checkStandardModelType(modelType):
    #         model = est_params[x]
    #         d = est_chunk.get_group(x)
    #         fcast = model.predict(d)
    #         fcast = d[['date', 'permno', 'ret_dlst_adj']].assign(fcast=fcast)
    #         fcast['e'] = fcast['ret_dlst_adj'] - fcast['fcast']
    #         fcast['e_sign'] = np.where(fcast['e'] > 0, 1, 0)
    #         k = len(model.params)
    #         n = len(d)
    #         sigma = np.sqrt((fcast['e']**2).sum()/(n-k))
    #         e_est_pos = fcast['e_sign'].mean()
    #         x = int(x)
    #         return pd.DataFrame({'permno': [x], 'sigma': [sigma], 'e_est_pos': [e_est_pos]})
    #     else:
    #         no_spaces = modelType.translate({ord(c): None for c in string.whitespace})
    #         response_var = no_spaces.partition("~")[0]
    #         model = est_params[x]
    #         d = est_chunk.get_group(x)
    #         fcast = model.predict(d)
    #         fcast = d[['date', 'permno', response_var]].assign(fcast=fcast)
    #         fcast['e'] = fcast[response_var] - fcast['fcast']
    #         fcast['e_sign'] = np.where(fcast['e'] > 0, 1, 0)
    #         k = len(model.params)
    #         n = len(d)
    #         sigma = np.sqrt((fcast['e']**2).sum()/(n-k))
    #         e_est_pos = fcast['e_sign'].mean()
    #         x = int(x)
    #         return pd.DataFrame({'permno': [x], 'sigma': [sigma], 'e_est_pos': [e_est_pos]})
