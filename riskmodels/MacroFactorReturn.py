import logging
import numbers
import itertools
import pandas as pd
import datetime
import numpy as np
import numpy.ma as ma
from riskmodels import CurveLoader


class MacroFactorReturn:
    # Factor name: (curve name, node, alpha for mlog return, duration)
    axcurve_mapping = {'USD BBB Corp Spread': ('USD-ALLSECTORS-GLOBAL-BBB2', '5Y', 0.01, None),
                        'EUR BBB Corp Spread': ('EUR-ALLSECTORS-GLOBAL-BBB2', '5Y', 0.01, None),
                        'GBP BBB Corp Spread': ('GBP-ALLSECTORS-GLOBAL-BBB2', '5Y', 0.01, None),
                        'JPY BBB Corp Spread': ('JPY-ALLSECTORS-GLOBAL-BBB2', '5Y', 0.01, None)
                        }
    # curve name, node, alpha (for mlog), duration (adjust unit)
    legacy_curve_mapping = {'US-Sov-Yield-LT': ('US.USD.GVT.ZC', '10Y', None, None),
                             'US-Sov-Yield-ST': ('US.USD.GVT.ZC', '6M', None, None),
                             'EU-Sov-Yield-LT': ('EP.EUR.GVT.ZC', '10Y', None, None),
                             'EU-Sov-Yield-ST': ('EP.EUR.GVT.ZC', '6M', None, None),
                             'JP-Sov-Yield-LT': ('JP.JPY.GVT.ZC', '10Y', None, None),
                             'JP-Sov-Yield-ST': ('JP.JPY.GVT.ZC', '6M', None, None),
                             'GB-Sov-Yield-LT': ('GB.GBP.GVT.ZC', '10Y', None, None),
                             'GB-Sov-Yield-ST': ('GB.GBP.GVT.ZC', '6M', None, None),
                             'US Inflation': ('US.USD.BEI', '5Y', None, None),
                             'JP Inflation': ('JP.JPY.BEI', '5Y', None, None),
                             'GB Inflation': ('GB.GBP.BEI', '5Y', None, None),
                             'EU Inflation': ('DE.EUR.BEI', '5Y', None, None),
                             'WTI Crude Oil Future: 1M': ('NYMEX:CL', '1M', None, None),
                             'WTI Crude Oil Future: 2M': ('NYMEX:CL', '2M', None, None),
                             'WTI Crude Oil Future: 3M': ('NYMEX:CL', '3M', None, None),
                             'Brent Crude Oil Future: 1M': ('ICE:CO', '1M', None, None),
                             'Brent Crude Oil Future: 2M': ('ICE:CO', '1M', None, None),
                             'Brent Crude Oil Future: 3M': ('ICE:CO', '1M', None, None),
                             }


    # use CMF curve for Oil factor
    commodity_curve_mapping = {'Oil': ('NYMEX:CL', '1M'),
                               'Carbon Emission Price': ('EEX:FEUA', '1Y')
                               }


    # curve name, short term node, long term node
    term_spread_mapping = {'US Term Spread: 10Y6M': ('US.USD.GVT.ZC', '6M', '10Y', None),
                           'EU Term Spread: 10Y6M': ('EP.EUR.GVT.ZC', '6M', '10Y', None),
                           'GB Term Spread: 10Y6M': ('GB.GBP.GVT.ZC', '6M', '10Y', None),
                           'JP Term Spread: 10Y6M': ('JP.JPY.GVT.ZC', '6M', '10Y', None),}

    def __init__(self,
                 modelDB,
                 marketDB,
                 macDB):

        # prod: 'prod-mac-mkt-db'

        self.modelDB = modelDB
        self.marketDB = marketDB
        self.macDB = macDB
        self.axcurve_loader = create_axcurve_loader(macDB)
        self.legacy_curve_loader = create_legacy_curve_loader(macDB)



        # Dict of methods for retrieving inputs
        self.methods = {
                'Gold': self.get_gold_return,
                'Commodity': self.get_commodity_return,
        }

        # add curve returns to the methods
        add_dict = {}
        for key in self.axcurve_mapping.keys():

            curve_name, node, alpha, duration = self.axcurve_mapping[key]
            
            add_dict[key] = create_compute_mlog_ret_history(curve_name, node, alpha, self.axcurve_loader, duration)

        for key in self.legacy_curve_mapping.keys():
            
            curve_name, node, alpha, duration = self.legacy_curve_mapping[key]
            
            add_dict[key] = create_compute_level_ret_history(curve_name, node, self.legacy_curve_loader, duration)


        for key in self.term_spread_mapping.keys():
            
            curve_name, short_node, long_node, alpha = self.term_spread_mapping[key]
            
            add_dict[key] = create_compute_term_spread_ret_history(curve_name, short_node, long_node, self.legacy_curve_loader)

        for key in self.commodity_curve_mapping.keys():

            curve_name, node = self.commodity_curve_mapping[key]

            add_dict[key] = create_simple_ret_history(curve_name, node, self.legacy_curve_loader)

        self.methods.update(add_dict)


    def get_index_return(self, axid, date_list):

        """Retrieve index returns"""
        sorted_date_list = sorted(date_list)
        start_date = sorted_date_list[0]
        end_date = sorted_date_list[-1]
        level_df = pd.DataFrame([], index=sorted_date_list)
        level = self.marketDB.getMacroTs([axid], start_date)
        
        level = pd.Series(level[axid]).sort_index()
        level_df['Price'] = level
        level_df.fillna(method='ffill', inplace=True)
        ret = level_df['Price'] / level_df['Price'].shift(1) - 1.
        if isinstance(ret.index, pd.DatetimeIndex):
            ret.index = ret.index.date
        idx = [d for d in ret.index if d >= start_date and d <= end_date]
        return ret.reindex(index=idx).dropna()

    def get_oil_return(self, date_list):
        """Retrieve oil price returns in USD"""
        sorted_date_list = sorted(date_list)
        start_date = sorted_date_list[0]
        end_date = sorted_date_list[-1]
        min_date = datetime.date(1983, 1, 10)
        start_date = max(start_date, min_date)
        series = 'M000100056'
        # M000100056  OPEC Oil Basket Price U$/Bbl
        # M000100022  Crude Oil-West Texas Intermediate Spot Cushing United States Dollar Per Barrel

        return self.get_index_return(series, date_list)

    def get_gold_return(self, date_list):
        """Retrieve gsci gold spot index returns"""
        sorted_date_list = sorted(date_list)
        start_date = sorted_date_list[0]
        end_date = sorted_date_list[-1]
        min_date = datetime.date(1978, 1, 6)
        start_date = max(start_date, min_date)
        series = 'M000100093'
        return self.get_index_return(series, date_list)

    def get_commodity_return(self, date_list):
        """Retrieve gsci non-energy spot index returns"""
        sorted_date_list = sorted(date_list)
        start_date = sorted_date_list[0]
        end_date = sorted_date_list[-1]
        min_date = datetime.date(1969, 12, 31)
        start_date = max(start_date, min_date)
        series = 'M000100117'
        return self.get_index_return(series, date_list)


def create_axcurve_loader(macDB):

    curve_loader = CurveLoader.CreditCurveLoader(macDB)

    def load_curve_history(curve_name, node, date_list):
        """extract AxCurve history for given start date and end date
        date is in datetime.date format
        node is a str which describe the node name

        return:
        a Serie with datetime index 
        """

        sorted_date_list = sorted(date_list)
        start_date = sorted_date_list[0]
        end_date = sorted_date_list[-1]
        full_history = curve_loader.getCurveNodesHistory(curve_name, [node], start=start_date, end=end_date)
        full_history.index = pd.to_datetime(full_history.index)

        full_history = full_history.reindex(sorted_date_list)

        output = full_history.iloc[:,0]
        output.sort_index(ascending=True)

        return output

    return load_curve_history


def load_legacy_curve_helper(curve_name, curve_loader, node, date_list):

    sorted_date_list = sorted(date_list)
    start_date = sorted_date_list[0]
    end_date = sorted_date_list[-1]

    nodes = curve_loader.getAllCurveNodes(curve_name)
    this_node = [i for i in nodes if i.tenor==node]

    current_node_history = curve_loader.getNodesHistory(this_node, start_date, end_date).rename(columns=lambda x: x.tenor)
    current_node_history.index = pd.to_datetime(current_node_history.index)

    current_node_history = current_node_history.reindex(sorted_date_list)

    return current_node_history

def create_legacy_curve_loader(macDB):
    """Take host name and 
    return a function which takes curve_name
    """

    curve_loader = CurveLoader.CurveLoader(macDB)

    stich_date_dict = {'DE.EUR.BEI': ('FR.EUR.BEI(HICP)', datetime.date(2007, 6, 1)),
                       'US.USD.BEI': ('US.USD.BEI.ZC', datetime.date(2007, 1, 3))}

    composite_bei_dict = {'DE.EUR.BEI': ('EP.EUR.GVT.ZC', 'DE.EUR.RGVT.ZC', datetime.date(2020, 1, 2)),
                          'JP.JPY.BEI': ('JP.JPY.GVT.ZC', 'JP.JPY.RGVT.ZC', datetime.date(2020, 1, 2)),}

    # fix for a few dates on USD BEI curve, data source, FRED 5Y BEI curve
    patch_usd_bei_for_few_dates = {datetime.date(2008, 4, 18): 0.0224,
                                   datetime.date(2008, 4, 21): 0.0225,
                                   datetime.date(2008, 4, 22): 0.0224,
                                   datetime.date(2008, 2, 7 ): 0.0199,
                                   datetime.date(2008, 2, 8 ): 0.0202,
                                   datetime.date(2008, 2, 11): 0.0205,
                                   }

    def load_curve_history(curve_name, node, date_list):
        """extract AxCurve history for given start date and end date
        date is in datetime.date format
        node is a str which describe the node name

        return:
        a Serie with datetime index 
        """

        sorted_date_list = sorted(date_list)
        start_date = sorted_date_list[0]
        end_date = sorted_date_list[-1]

        # for BEI curve, some need to be adjusted due to history of the curve is not good.
        if curve_name in list(composite_bei_dict.keys()):
            nominal, real, switch_date = composite_bei_dict[curve_name]

            after_switch_date_list = [i for i in sorted_date_list if i >=switch_date]
            before_switch_date_list = [i for i in sorted_date_list if i <=switch_date]
            if end_date <= switch_date:
                nominal_history = load_legacy_curve_helper(nominal, curve_loader, node, sorted_date_list)
                real_history = load_legacy_curve_helper(real, curve_loader, node, sorted_date_list)
                current_node_history = nominal_history - real_history
                current_node_history.dropna(inplace=True)

            elif start_date >= switch_date:
                current_node_history = load_legacy_curve_helper(curve_name, curve_loader, node, sorted_date_list)
            
            else:
                nominal_history = load_legacy_curve_helper(nominal, curve_loader, node, before_switch_date_list)
                real_history = load_legacy_curve_helper(real, curve_loader, node, before_switch_date_list)
                current_node_history_prev = nominal_history - real_history
                current_node_history_prev.dropna(inplace=True)

                current_node_history_post = load_legacy_curve_helper(curve_name, curve_loader, node, after_switch_date_list)
                current_node_history = pd.concat([current_node_history_prev.drop([switch_date]), current_node_history_post.loc[switch_date:]], axis=0)

        else:
            current_node_history = load_legacy_curve_helper(curve_name, curve_loader, node, sorted_date_list)

        # stich EU inflation or others if needed
        if curve_name in list(stich_date_dict.keys()):
            backfill_curve_name, stich_date = stich_date_dict[curve_name]
            if start_date < stich_date:
                before_stich_date_list = [i for i in sorted_date_list if i <=stich_date]
                back_fill_history = load_legacy_curve_helper(backfill_curve_name, curve_loader, node, before_stich_date_list)

                if end_date < stich_date: # if end date is before stiching, take it all
                    output_history = back_fill_history.loc[start_date:end_date]
                else:
                    # compute level diff
                    diff = current_node_history.loc[stich_date] - back_fill_history.loc[stich_date]
                    back_fill_history += diff

                    # stiching 
                    output_history = pd.concat([back_fill_history.drop([stich_date]), current_node_history.loc[stich_date:]], axis=0)
            else:
                output_history = current_node_history


        else:
            output_history = current_node_history

        # patch USD BEI for 2008:
        if curve_name == 'US.USD.BEI':
            patch_df = pd.DataFrame.from_dict(patch_usd_bei_for_few_dates, orient='index')
            patch_df.columns=[node]
            output_history.update(patch_df)

        output = output_history.iloc[:,0] # convert to series
        output.sort_index(ascending=True)

        return output

    return load_curve_history


def create_simple_ret_history(curve_name, node, load_curve_history_func):

    def get_returns(date_list):
        sorted_date_list = sorted(date_list)
        start_date = sorted_date_list[0]
        end_date = sorted_date_list[-1]
        curve_df_raw = load_curve_history_func(curve_name, node, sorted_date_list)
        curve_df = curve_df_raw.fillna(0.0)
        return_df = curve_df[start_date:]

        return return_df

    return get_returns


def create_compute_level_ret_history(curve_name, node, load_curve_history_func, duration=None):
    """Take curve_name, node and a function who loads the history
    return a function which takes a date list and produce the require returns, note that the loaded curve nodes may contain Nan,
    the function will use forwardfill to fill NaN.
    """

    def get_returns(date_list):
        sorted_date_list = sorted(date_list)
        start_date = sorted_date_list[0]
        end_date = sorted_date_list[-1]
        # extract dates to make sure all needed dates are covered

        curve_df_raw = load_curve_history_func(curve_name, node, sorted_date_list)
        curve_df = curve_df_raw.fillna(method='ffill')
        return_df = compute_level_return(curve_df)
        return_df = return_df[start_date:] # make sure the output is a Serie

        if duration is not None:
            return_df = - duration * return_df

        return return_df

    return get_returns


def create_compute_mlog_ret_history(curve_name, node, alpha, load_curve_history_func, duration=None):
    """Take curve_name, node and a function who loads the history
    return a function which takes a date list and produce the require returns
    """

    def get_returns(date_list):
        sorted_date_list = sorted(date_list)
        start_date = sorted_date_list[0]
        end_date = sorted_date_list[-1]
        # extract dates to make sure all needed dates are covered

        curve_df_raw = load_curve_history_func(curve_name, node, sorted_date_list)
        curve_df = curve_df_raw.fillna(method='ffill')
        return_df = compute_mlog_return(curve_df, alpha)

        if duration is not None:
             mlog_dts = curve_df.shift(1).apply(lambda x: duration*max(x, alpha)).dropna()
             return_df = - mlog_dts * return_df 

        return_df = return_df[start_date:] # make sure the output is a Serie

        return return_df

    return get_returns


def create_compute_term_spread_ret_history(curve_name, short_node, long_node, load_curve_history_func):
    """Take curve_name, node and a function who loads the history
    return a function which takes a date list and produce the require returns
    """

    duration_mapping = {'10Y': 10.,
                        '6M': 0.5,}

    long_node_duration = duration_mapping[long_node]
    short_node_duration = duration_mapping[short_node]

    def get_returns(date_list):
        sorted_date_list = sorted(date_list)
        start_date = sorted_date_list[0]
        end_date = sorted_date_list[-1]

        # extract dates to make sure all needed dates are covered
        curve_df_short = load_curve_history_func(curve_name, short_node, sorted_date_list)
        curve_df_long = load_curve_history_func(curve_name, long_node, sorted_date_list)
        term_spread_raw = curve_df_long - curve_df_short
        term_spread = term_spread_raw.reindex(sorted_date_list)
        term_spread = term_spread.fillna(method='ffill')
        return_df = compute_level_return(term_spread)

        return_df = return_df[start_date:]

        return return_df

    return get_returns


def create_compute_bei_ret_history(nominal_curve, real_curve, node, load_curve_history_func):
    """Take curve_name, node and a function who loads the history
    return a function which takes a date list and produce the require returns
    """

    def get_returns(date_list):
        sorted_date_list = sorted(date_list)
        start_date = sorted_date_list[0]
        end_date = sorted_date_list[-1]
        # extract dates to make sure all needed dates are covered

        curve_df_nominal = load_curve_history_func(nominal_curve, node, sorted_date_list)
        curve_df_real = load_curve_history_func(real_curve, node, sorted_date_list)

        bei_raw = curve_df_nominal - curve_df_real
        bei = bei_raw.reindex(sorted_date_list)
        bei = bei.fillna(method='ffill')

        return_df = compute_level_return(bei)

        return_df = return_df[start_date:]

        return return_df

    return get_returns


def mlog(x, alpha):
        """mlog function to smooth return
        """
        if x > alpha:
            return np.log(x/alpha)+1
        else:
            return x/alpha

def compute_level_return(node_df):

    result = (node_df - node_df.shift(1)).dropna()
    return result
    
def compute_mlog_return(node_df, alpha):
    mlog_curve = node_df.apply(lambda x: np.vectorize(mlog)(x, alpha=alpha))
    result = (mlog_curve - mlog_curve.shift(1)).dropna()*alpha # change unit back
    return result


