
import logging
import datetime
from dateutil.relativedelta import relativedelta
from riskmodels import FactorLibrary


TEXT_COLS=[
'HOLT_UNIQUE_ID','SEDOL','CUSIP','TICKER','COMPANY_NM','COUNTRY_CD','PERIOD_ID','FY0','CURRENCY_CD',
]

TEXT_COL_DICT=dict([(i,'TEXT') for i in TEXT_COLS])

CSHOLT_COL_NAMES=[
['HOLT_UNIQUE_ID','HOLT Permanent Identifier','Descriptive'],
#['CUSIP','CUSIP','Descriptive'],
#['ISIN','ISIN','Descriptive'],
#['NAME','N/A','N/A'],
#['TICKER','Ticker','Descriptive'],
#['SEDOL','SEDOL','Descriptive'],
#['COUNTRY_CD','Country code','Descriptive'],
['ADJVALTG','Adjusted Value Target','Valuation'],
['ASSET_GROWTH_RATE','Growth in Growth Investment - LFY','Growth'],
['CDGI_LFY','Gross Investment Inflation Adjusted - LFY','Fundamental'],
['CFMELFY1','Free Cash Flow Multiple (equity) - FY1','Valuation'],
['CFMELFY2','Equity Free Cash Flow Multiple - FY2','Valuation'],
['CFME_LFY','Free Cash Flow Multiple (equity) - LFY','Valuation'],
['CF_NI','Net Income / Gross Cash Flow - FY0','Valuation'],
['CFROI_AVG','CFROI Average','Quality'],
['CFROICARET1Y','CFROI FY1','Quality'],
['CFROICARET2Y','CFROI FY2','Quality'],
['CFROICHG','CFROI Change FY1','Quality'],
['CFROI_DR_SPREAD','CFROI Discount Rate Spread','Quality'],
['CFROI_FY0','CFROI LFY','Quality'],
['CFROI_FY1_MOM','CFROI FY1 Momentum','Momentum'],
['CFROI_FY2_MOM','CFROI FY2 Momentum','Momentum'],
['CFROI_KEY_MOM','CFROI 13-Week Key Momentum ','Momentum'],
['CFROIMINUS1Y','CFROI LFY -1','Quality'],
['CFROIMINUS2Y','CFROI LFY -2','Quality'],
['CFROIMINUS3Y','CFROI LFY -3','Quality'],
['CFROIMINUS4Y','CFROI LFY -4','Quality'],
['CFROIMINUS5Y','CFROI LFY -5','Quality'],
['CFROI_MKTIMP','CFROI Market Implied FY(n)','Valuation'],
['CFROI_MO','CFROI Key Momentum (current month)','Momentum'],
['CFROIVAL','CFROI used in valuation','Quality'],
['CFROI_VOL_MEAN','CFROI Voliltility','Quality'],
['CFROIVP5','CFROI t+5','Valuation'],
['CFSR_CQ','Cash Flow Surprise','Momentum'],
['CHGVLCRT','Change in Value Creation','Quality'],
['COMPANY_NM','Company Name','Descriptive'],
['COVERAGE','Interest Coverage Ratio - LFY','Risk'],
['CURRENCY_CD','Currency Code','Descriptive'],
['DAY_LIQD','Daily Liquidity (mil)','Market'],
['DAY_VOL','Daily Volume Average (K shares)','Market'],
['DEP_ASSET_GROWTH_RATE','Change in Depreciating Assets','Growth'],
['DEV','Deviation, relative to calendar yr prices - annual %','Valuation'],
['DEVTKBG_FY1','Deviation, Tracking','Valuation'],
['DISCRATEVAL','Company-Specific Discount Rate used in Valuation','Risk'],
['DIVYIELD','Dividend Yield % - Current','Valuation'],
['DR_DIF','Discount Rate Differential','Risk'],
['ECEY','HOLT Economic P/E','Valuation'],
['ENTERPRISE_WINDDOWN','Enterprise Wind down Ratio','Valuation'],
['EPSGROWTH','Change in EPS','Growth'],
['EPSNESTFY1','EPS # of Analysts Coverage - FY1','Growth'],
['EPSNESTFY2','EPS # of Analysts Coverage - FY2','Growth'],
['EPSSURP_CQ','EPS Surprise - LFQ','Momentum'],
['FADERATE','Fade Rate','Valuation'],
['FY0','Last Fiscal Year (LFY)','Descriptive'],
['GCF_MARGIN','Gross Cash Flow Margin','Valuation'],
['GIC_SUBINDUSTRY_ID','MSCI GICS Sub industry ID','Descriptive'],
['GRCSHLF1','Gross Cash Flow, Inflation Adjusted - FY1','Fundamental'],
['GRCSHLFY','Gross Cash Flow, Inflation Adjusted - LFY','Fundamental'],
['GROW5EPS','Growth - Growth in EPS (5-year)','Growth'],
['GROW5INV','Growth - Growth in Investments (5-year)','Growth'],
['GROWSLFY','Normalized Growth Spread - LFY','Growth'],
['GROWVAL','Normalized Growth Rate Used in Valuation','Growth'],
['GRWSUCARET1','Growth - Normalized for FY1','Growth'],
['GRWSUCARET2','Growth - Normalized for FY2','Growth'],
['GRWSUMINUS1','Growth - Normalized for LFY -1','Growth'],
['GRWSUMINUS2','Growth - Normalized for LFY -2','Growth'],
['H1056','Asset Life LFY','Fundamental'],
['H1499FY1','Non Depreciating Assets - FY1','Fundamental'],
['H1499','Non Depreciating Assets - LFY','Fundamental'],
['H1500FY1','Gross Investment Inflation Adjusted - FY1','Fundamental'],
['H1500FY2','Gross Investment Inflation Adjusted - FY2','Fundamental'],
['H156FY1','Asset Life FY1','Fundamental'],
['H1807FY1','Economic Accumulated Depreciation','Fundamental'],
['H1809','Growth in Gross Investment, historic bounded','Growth'],
['H2413','Company-Specific Discount Rate LFY','Risk'],
['H2413M1','Company-Specific Discount Rate LFY-1','Risk'],
['H2413M2','Company-Specific Discount Rate LFY-2','Risk'],
['H2421','Debt and Equivalents (used in Valuation)','Fundamental'],
['H2421FY1','Market Value of Debt & Equivalents FY1','Fundamental'],
['H2440','Market Value of Investments - LFY','Fundamental'],
['H2443FY1','HOLT Warranted Enterprise Value FY1','Valuation'],
['H2446FY1','Market Value of Minority Interest FY1','Fundamental'],
['H2446','MV of Minority Interest','Fundamental'],
['H2488','Normalized Growth Rate Used in Valuation/Fade FY5','Growth'],
['H2492','Inflation Adjusted Net Assets Used in Valuation','Fundamental'],
['H3028FY1','Growth in Gross Investment - FY1','Growth'],
['H3028FY2','Growth in Gross Investment - FY2','Growth'],
['H3028M1','Growth in Growth Investment - LFY-1','Growth'],
['H3028M2','Growth in Growth Investment - LFY-2','Growth'],
['H3028M3','Growth in Growth Investment - LFY-3','Growth'],
['H3028M4','Growth in Growth Investment - LFY-4','Growth'],
['H3028M5','Growth in Growth Investment - LFY-5','Growth'],
['H3144FY1','Sustaining Ratio','Quality'],
['H3145FY1','Percent Growth','Growth'],
['LEVDSCRT','Leverage for Discount Rate Differential','Quality'],
['LEVG_MKT','Leverage at Market','Risk'],
['MKT_CAP','Market Capitalization (bil) in local','Market'],
['MKT_CAPUS','Market Capitalization (bil) in USD','Market'],
['MKTDISC','Market Implied Discount Rate','Valuation'],
['MNGFRVLWGTAVG','Managing for Value','Quality'],
['MVINV','Market Value of Investments - FY1','Fundamental'],
['NORMALIZED_GROWTH_FY0','Growth - Normalized for LFY','Growth'],
['NYEARS','Years of Data','Descriptive'],
['P2B','Percent Change To Warranted Value','Valuation'],
['PBE_CUR','Price / Book Equity - Current','Valuation'],
['PEBITAC','Price / EBITDA Ratio - Current Price / EBITDA (LFY)','Valuation'],
['PEBITA','Total Capital to EBITDA','Valuation'],
['PE_FY1','Price / Earnings Ratio - FY1','Valuation'],
['PE_FY2','Price / Earnings Ratio - FY2','Valuation'],
['PE','Price / Earnings Ratio - LFY','Valuation'],
['PERCFUT','Percent Future','Growth'],
['PERIOD_ID','Period ID','Descriptive'],
['PFCFPS','Price / Free Cash Flow per share','Valuation'],
['POD','Probabilty of Default','Risk'],
['PPERCENTCHGH00','Price, % Change to High','Valuation'],
['PPERCENTCHGL00','Price, % Change to Low','Valuation'],
['PRICE_ME','Current Price','Market'],
['PRICE_SALES','Price / Sales Ratio - Current','Valuation'],
['PTARGETB','Warranted Value, Actual','Valuation'],
['PV_EXIST','Present Value of Exist. Assets','Fundamental'],
['PV_FUTUR','Present Value of Future Invest','Fundamental'],
['RET1MOFO','Shareholder Return (Total) - One Month Forward','Market'],
['RETRAW1M','Price Return - One Month','Market'],
['RETTOT04','Shareholder Return (Total) - One Month','Market'],
['ROIMOMINUS1M','CFROI Key Momentum (current month -1)','Momentum'],
['ROIMOMINUS2M','CFROI Key Momentum (current month -2)','Momentum'],
['ROITRLF1','CFROI Transaction - FY-1','Quality'],
['ROITRLF2','CFROI Transaction - FY-2','Quality'],
['SALES_TO_CAPRD','Sales to Capitalized R&D','Valuation'],
['SALES_TO_GRS_INVEST','Sales to Gross Investment','Valuation'],
['SLSGROWTH','Growth in Sales - LFY','Growth'],
['SLSGRW5','Change in Sales','Growth'],
['SUS_GROWTH_FY5','Normalized Growth Rate Market Implied FY(n)','Growth'],
['SUS_GRWTH_RATE','Sustainable Growth Rate','Growth'],
['TROILFY1','CFROI Transaction FY1','Quality'],
['TROILFY2','CFROI Transaction FY2','Quality'],
['TROILFY','CFROI Transaction FY0','Quality'],
['TURNGLFY','Sales / Inflation Adjusted Gross Investment - LFY','Valuation'],
['VCR','Value/Cost Ratio (current)','Valuation'],
['VI_GRSINV','Inflation Adjusted Gross Investment used in Valuation','Fundamental'],
['VOL_A_HIS_3M_CALL','Implied Volatility 90-day Call','Risk'],
['VOL_A_HIS_3M','Realized Price Volatility','Risk'],
['WDRATIO','Wind down Ratio','Valuation'],
]

class CSHolt (FactorLibrary.FactorLibrary):
    def __init__(self, vendorDB, marketDB, modelDB, factorName, options):
        FactorLibrary.FactorLibrary.__init__(self,vendorDB, marketDB, modelDB, factorName, options)
        # store holidays in a list here
        self.holidays={}
        query="""
                select dt from META_HOLIDAY_CALENDAR_ACTIVE h where ISO_CTRY_CODE='US' and dt <= to_date('19961231','YYYYMMDD')
        """
        self.marketDB.dbCursor.execute(query)
        for r in self.marketDB.dbCursor.fetchall():
            self.holidays[str(r[0])[:10]]=True
         
    

    def getHeaders(self, dt):
        """
            create the headers
        """
        shortdate='%04d%02d%02d' % (dt.year, dt.month, dt.day)

        # first line is the factor name/categorization of the CS Holt factors
        # second line is the names as provided in the files
        # third line is the descriptions
        # fourth line is the dates
        # fifth line are the units

        #line1=['']+['CSHolt|CSHolt.%s.%s|Factor Library' % (c[2],c[0]) for c in CSHOLT_COL_NAMES]
        line1=['']+['HOLT|HOLT.%s|Factor Library|GROUP' % (c[2]) for c in CSHOLT_COL_NAMES]
        line2=['NAME'] + [c[0] for c in CSHOLT_COL_NAMES]
        line3=['DESC'] + [c[1] for c in CSHOLT_COL_NAMES]
        line4=['DATE'] + ['%s' % shortdate for c in CSHOLT_COL_NAMES]
        line5=['UNIT'] + ['%s' % TEXT_COL_DICT.get(c[0],'NUMBER') for c in CSHOLT_COL_NAMES]

        self.headers=[line1, line2, line3, line4, line5]
        return

    def getConstituents(self, dt):
        """
            get constitutents from CSHolt VendorDB for the given date
        """
        runDate=dt
        if self.options.monthly:
            # check if it is the last weekday of the month
                tempdt=dt+relativedelta(months=1)
                lastdt=datetime.date(tempdt.year, tempdt.month, 1) + datetime.timedelta(days=-1)
                while (lastdt.isoweekday() in (6,7)) or str(lastdt)[:10] in self.holidays:
                    lastdt=lastdt+datetime.timedelta(days=-1)
                if dt != lastdt:
                    logging.debug('%s not the last day of the month (%s), nothing to do', dt, lastdt)
                    return False
                query="""select min(dt) from csholt_axioma_data where dt >= :dt """
                self.vendorDB.dbCursor.execute(query,dt=dt)
                results=self.vendorDB.dbCursor.fetchall()
                if len(results) == 0 or not results[0][0]:
                    logging.warning('%s no results nothing to do', dt)
                    return False
                runDate=results[0][0] 
                if str(dt)[:10] != str(runDate)[:10]:
                    logging.info('Switching date to %s (%d) from %s (%d)', dt, dt.isoweekday(),str(runDate)[:10], runDate.isoweekday())
        # get list of active issues
        issueDict=dict([[i.getIDString()[1:],1] for i in self.modelDB.getActiveIssues(runDate)])
        query = """
                select axioma_id, %s 
                from CSHOLT_AXIOMA_DATA
                where dt=:dt
                order by axioma_id
        """ % ( ','.join([c[0] for c in CSHOLT_COL_NAMES]))
        self.vendorDB.dbCursor.execute(query,dt=runDate)
        results = self.vendorDB.dbCursor.fetchall()
        deadissues=[r for r in results if r[0] not in issueDict]
        self.data=[r for r in results if r[0] in issueDict]
        dtstr=str(dt)[:10]
        rundtstr=str(runDate)[:10]
        logging.info('On %s %d issues were dead %s', rundtstr, len(deadissues), ','.join(["'D%s'" % r[0] for r in deadissues]))
        logging.info('On %s %d issues were dead %s', rundtstr, len(deadissues), '|'.join(['%s' % r[0] for r in deadissues]))
        if len(results)==0:
            logging.warning('There are no results on %s for %s (%s)', dtstr, rundtstr,  self.factorName)
            return False
        else:
            logging.info('%d (out of %d) entries on %s (%s) for %s factor', len(self.data), len(results), dtstr, rundtstr, self.factorName)
            # fix up the header line now 
            return True

