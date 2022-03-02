#individual sections, if needed more then once
SUB_ISSUE_CUMULATIVE_RETURN_SECTION='SubIssueCumulativeReturn'
SUB_ISSUE_DATA_SECTION = 'SubIssueData'
SUB_ISSUE_RETURN_SECTION = 'SubIssueReturn'


#lists of sections
#order doesn't matter for industry sections, so, it is alphabetical
MODELDB_INDUSTRY_SECTIONS =[
                                'GICSCustomAU',
                                'GICSCustomAU2',
                                'GICSCustomCA',
                                'GICSCustomCN',
                                'GICSCustomEM2',
                                'GICSCustomEU',
                                'GICSCustomGB',
                                'GICSCustomJP',
                                'GICSCustomJP-2008',
                                'GICSCustomJP-2014',
                                'GICSCustomJP-2016',
                                'GICSCustomNA',
                                'GICSCustomNA4',
                                'GICSCustomNoMortgageREITs2018',
                                'GICSCustomTW',
                                'GICSCustomCA4',
                                'IndustryGICS-2008',
                                'IndustryGICS-2014',
                                'IndustryGICS-2016',
                                'IndustryGICS-2018',
                                'IndustryGroupGICS',
                                'IndustryGroupGICS-2008',
                                'IndustryGroupGICS-2016',
                                'IndustryGroupGICS-2018'
                            ]

MODELDB_ESTIMATES_SECTIONS   = ['EstimateCurrencyDataModelDB', 'AssetEstimateData']
MODELDB_FUNDAMENTAL_SECTIONS = ['FundamentalCurrencyData','FundamentalNumberData','SandP.Xpressfeed.FundamentalData']
MODELDB_SUBISSUE_SECTIONS    = [SUB_ISSUE_DATA_SECTION, SUB_ISSUE_RETURN_SECTION, SUB_ISSUE_CUMULATIVE_RETURN_SECTION,'SubIssueDivYield']


#master list
MODELDB_SECTIONS = MODELDB_INDUSTRY_SECTIONS+\
                            MODELDB_ESTIMATES_SECTIONS+\
                            MODELDB_FUNDAMENTAL_SECTIONS+\
                            MODELDB_SUBISSUE_SECTIONS
                            
