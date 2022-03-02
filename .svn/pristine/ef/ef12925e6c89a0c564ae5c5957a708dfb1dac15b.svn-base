import configparser
import datetime
import logging
import optparse
from difflib import SequenceMatcher
from riskmodels import Utilities
from riskmodels import Connections

def getClassificationRevisionLeaves(mkt, revision, level):
    """Returns a list of the leaf classifications
    for the given classification revision.
    """
    if level == 1:
        mkt.dbCursor.execute("""SELECT c.id, p.description
          FROM classification_ref c
          JOIN classification_dim_hier h ON c.id=h.child_classification_id
          JOIN classification_ref p ON p.id=h.parent_classification_id
          WHERE c.is_leaf='Y' AND c.revision_id = :rev_id""",
                             rev_id=revision.id)
    elif level == 2:
        mkt.dbCursor.execute("""SELECT c.id, p.description
          FROM classification_ref c
          JOIN classification_dim_hier h1 ON c.id=h1.child_classification_id
          JOIN classification_dim_hier h2
            ON h1.parent_classification_id=h2.child_classification_id
          JOIN classification_ref p ON p.id=h2.parent_classification_id
          WHERE c.is_leaf='Y' AND c.revision_id = :rev_id""",
                             rev_id=revision.id)
    elif level == 3:
        mkt.dbCursor.execute("""SELECT c.id, p.description
          FROM classification_ref c
          JOIN classification_dim_hier h1 ON c.id=h1.child_classification_id
          JOIN classification_dim_hier h2
            ON h1.parent_classification_id=h2.child_classification_id
          JOIN classification_dim_hier h3
            ON h2.parent_classification_id=h3.child_classification_id
          JOIN classification_ref p ON p.id=h3.parent_classification_id
          WHERE c.is_leaf='Y' AND c.revision_id = :rev_id""",
                             rev_id=revision.id)
    else:
        assert(not 'Unsupported level')
    return [(mkt.getClassificationByID(i[0]), i[1])
            for i in mkt.dbCursor.fetchall()]
guessMaps = {}
guessMaps[('GICSIndustries', datetime.date(2006,4,29))] = {
    '10102010': ('Oil, Gas & Consumable Fuels', 'N'),
    '10102020': ('Oil, Gas & Consumable Fuels', 'N'),
    '10102030': ('Oil, Gas & Consumable Fuels', 'N'),
    '20201020': ('IT Services', 'Y'),
    '20202010': ('Commercial Services & Supplies', 'N'),
    '20202020': ('Commercial Services & Supplies', 'N'),
    '20301010': ('Air Freight & Logistics', 'N'),
    '25202010': ('Leisure Equipment & Products', 'N'),
    '25203010': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25203020': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25203030': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25502020': ('Internet & Catalog Retail','N'), # CCHU: 2016-09-01 Addition
    '30101010': ('Food & Staples Retailing', 'N'),
    '30101020': ('Food & Staples Retailing', 'N'),
    '30101030': ('Food & Staples Retailing', 'N'),
    '40101010': ('Commercial Banks', 'N'),
    '40101015': ('Commercial Banks', 'N'),
    '40201010': ('Diversified Financial Services', 'N'),
    '40201020': ('Diversified Financial Services', 'N'),
    '40201030': ('Diversified Financial Services', 'N'),
    '40204010': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '40401010': ('Real Estate Investment Trusts (REITs)', 'Y'),
    '40401020': ('Real Estate Management & Development', 'Y'),
    '45102010': ('IT Services', 'N'),
    '45202030': ('Computers & Peripherals', 'N'),
    '45203010': ('Electronic Equipment & Instruments', 'N'),
    '45203015': ('Electronic Equipment & Instruments', 'N'),
    '45203020': ('Electronic Equipment & Instruments', 'N'),
    '45203030': ('Electronic Equipment & Instruments', 'N'),
    '45205010': ('Semiconductors & Semiconductor Equipment', 'Y'),
    '45205020': ('Semiconductors & Semiconductor Equipment', 'Y'),
    '55103010': ('Multi-Utilities', 'N'),
    '55105010': ('Independent Power Producers & Energy Traders', 'N'),
    '55105020': ('Independent Power Producers & Energy Traders', 'N'),
    '60101010': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101020': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101030': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101040': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101050': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101060': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101070': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101080': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    }

guessMaps[('GICSIndustries', datetime.date(2008,8,30))] = {
    '10102010': ('Oil, Gas & Consumable Fuels', 'N'),
    '10102020': ('Oil, Gas & Consumable Fuels', 'N'),
    '10102030': ('Oil, Gas & Consumable Fuels', 'N'),
    '20201020': ('IT Services', 'Y'),
    '20201040': ('Professional Services', 'Y'),
    '20301010': ('Air Freight & Logistics', 'N'),
    '25202010': ('Leisure Equipment & Products', 'N'),
    '25203010': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25203020': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25203030': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25502020': ('Internet & Catalog Retail','N'), # CCHU: 2016-09-01 Addition
    '30101010': ('Food & Staples Retailing', 'N'),
    '30101020': ('Food & Staples Retailing', 'N'),
    '30101030': ('Food & Staples Retailing', 'N'),
    '40101010': ('Commercial Banks', 'N'),
    '40101015': ('Commercial Banks', 'N'),
    '40201010': ('Diversified Financial Services', 'N'),
    '40201020': ('Diversified Financial Services', 'N'),
    '40201030': ('Diversified Financial Services', 'N'),
    '40204010': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '40401010': ('Real Estate Investment Trusts (REITs)', 'Y'),
    '40401020': ('Real Estate Management & Development', 'Y'),
    '45102010': ('IT Services', 'N'),
    '45202030': ('Computers & Peripherals', 'N'),
    '45203010': ('Electronic Equipment, Instruments & Components', 'N'),
    '45203020': ('Electronic Equipment, Instruments & Components', 'N'),
    '45203030': ('Electronic Equipment, Instruments & Components', 'N'),
    '45205010': ('Semiconductors & Semiconductor Equipment', 'Y'),
    '45205020': ('Semiconductors & Semiconductor Equipment', 'Y'),
    '50202010': ('Media', 'N'), # FYAU: 2018-09-29 Addition
    '50202020': ('Software', 'N'), # FYAU: 2018-09-29 Addition
    '50203010': ('Internet Software & Services', 'N'), # FYAU: 2018-09-29 Addition
    '55103010': ('Multi-Utilities', 'N'),
    '55105010': ('Independent Power Producers & Energy Traders', 'N'),
    '55105020': ('Independent Power Producers & Energy Traders', 'N'),
    '60101010': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101020': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101030': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101040': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101050': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101060': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101070': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101080': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    }

guessMaps[('GICSIndustries', datetime.date(2014,3,1))] = {
    '10102010': ('Oil, Gas & Consumable Fuels', 'N'),
    '10102020': ('Oil, Gas & Consumable Fuels', 'N'),
    '10102030': ('Oil, Gas & Consumable Fuels', 'N'),
    '20201020': ('IT Services', 'Y'),
    '20201040': ('Professional Services', 'Y'),
    '20301010': ('Air Freight & Logistics', 'N'),
    '25202010': ('Leisure Equipment & Products', 'N'),
    '25202020': ('Leisure Equipment & Products', 'N'),
    '25203010': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25203020': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25203030': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25502020': ('Internet & Catalog Retail','N'), # CCHU: 2016-09-01 Addition
    '30101010': ('Food & Staples Retailing', 'N'),
    '30101020': ('Food & Staples Retailing', 'N'),
    '30101030': ('Food & Staples Retailing', 'N'),
    '40101010': ('Commercial Banks', 'N'),
    '40101015': ('Commercial Banks', 'N'),
    '40201010': ('Diversified Financial Services', 'N'),
    '40201020': ('Diversified Financial Services', 'N'),
    '40201030': ('Diversified Financial Services', 'N'),
    '40204010': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '40401010': ('Real Estate Investment Trusts (REITs)', 'Y'),
    '40401020': ('Real Estate Management & Development', 'Y'),
    '45102010': ('IT Services', 'N'),
    '45202010': ('Computers & Peripherals', 'Y'),
    '45202020': ('Computers & Peripherals', 'Y'),
    '45202030': ('Computers & Peripherals', 'N'),
    '45203010': ('Electronic Equipment, Instruments & Components', 'N'),
    '45203020': ('Electronic Equipment, Instruments & Components', 'N'),
    '45203030': ('Electronic Equipment, Instruments & Components', 'N'),
    '45204010': ('Computers & Peripherals', 'Y'),
    '45205010': ('Semiconductors & Semiconductor Equipment', 'Y'),
    '45205020': ('Semiconductors & Semiconductor Equipment', 'Y'),
    '50202010': ('Media', 'N'), # FYAU: 2018-09-29 Addition
    '50202020': ('Software', 'N'), # FYAU: 2018-09-29 Addition
    '50203010': ('Internet Software & Services', 'N'), # FYAU: 2018-09-29 Addition
    '55103010': ('Multi-Utilities', 'N'),
    '55105010': ('Independent Power Producers & Energy Traders', 'N'),
    '55105020': ('Independent Power Producers & Energy Traders', 'N'),
    '60101010': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101020': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101030': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101040': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101050': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101060': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101070': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101080': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    }

guessMaps[('GICSIndustries', datetime.date(2016,9,1))] = {
    '10102010': ('Oil, Gas & Consumable Fuels', 'N'),
    '10102020': ('Oil, Gas & Consumable Fuels', 'N'),
    '10102030': ('Oil, Gas & Consumable Fuels', 'N'),
    '20201040': ('Professional Services', 'Y'),
    '20301010': ('Air Freight & Logistics', 'N'), 
    '25202010': ('Leisure Products', 'N'), # Leisure Products
    '25202020': ('Leisure Products', 'N'), # Photographic Products
    '25203010': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25203020': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25203030': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25502010': ('Internet & Direct Marketing Retail', 'N'), # Catalog Retail (Discontinued)
    '25502020': ('Internet & Direct Marketing Retail', 'N'), # Internet Retail (Renamed)
    '30101010': ('Food & Staples Retailing', 'N'),
    '30101020': ('Food & Staples Retailing', 'N'),
    '30101030': ('Food & Staples Retailing', 'N'),
    '35102010': ('Health Care Providers & Services', 'N'),
    '35102015': ('Health Care Providers & Services', 'N'),
    '35102020': ('Health Care Providers & Services', 'N'),
    '35102030': ('Health Care Providers & Services', 'N'),
    '40101010': ('Banks', 'N'), # Diversified Bank
    '40101015': ('Banks', 'N'), # Regional Banks
    '40201010': ('Diversified Financial Services', 'N'),
    '40201020': ('Diversified Financial Services', 'N'),
    '40201030': ('Diversified Financial Services', 'N'),
    '40401010': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Real Estate Investment Trusts
    '40401020': ('Real Estate Management & Development', 'N'), # Real Estate Management & Development
    '40402010': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Diversified REITs
    '40402020': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Industrial REITs
    '40402035': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Hotel & Resort REITs
    '40402040': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Office REITs
    '40402045': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Health Care REITs
    '40402050': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Residential REITs
    '40402060': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Retail REITs
    '40402070': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Specialized REITs
    '40402030': ('Mortgage Real Estate Investment Trusts (REITs)', 'N'), # Mortgage REITs <-- EXCLUDED FROM EQUITY REITS
    '45102010': ('IT Services', 'N'),
    '45202010': ('Technology Hardware, Storage & Peripherals', 'N'), # Computer Hardware
    '45202020': ('Technology Hardware, Storage & Peripherals', 'N'), # Computer Storage & Peripherals
    '45204010': ('Technology Hardware, Storage & Peripherals', 'N'), # Office Electronics
    '45203010': ('Electronic Equipment, Instruments & Components', 'N'), # Electronic Equipment & Instruments
    '45203020': ('Electronic Equipment, Instruments & Components', 'N'), # Electronic Manufacturing Services
    '45203030': ('Electronic Equipment, Instruments & Components', 'N'), # Technology Distributors
    '45205010': ('Semiconductors & Semiconductor Equipment', 'Y'),
    '45205020': ('Semiconductors & Semiconductor Equipment', 'Y'),
    '50202010': ('Media', 'N'), # FYAU: 2018-09-29 Addition
    '50202020': ('Software', 'N'), # FYAU: 2018-09-29 Addition
    '50203010': ('Internet Software & Services', 'N'), # FYAU: 2018-09-29 Addition
    '55103010': ('Multi-Utilities', 'N'),
    '55105010': ('Independent Power and Renewable Electricity Producers', 'N'), # Independent Power Producers & Energy Traders
    '55105020': ('Independent Power and Renewable Electricity Producers', 'N'),
    }

guessMaps[('GICSIndustries', datetime.date(2018,9,29))] = {
    '10102010': ('Oil, Gas & Consumable Fuels', 'N'),
    '10102020': ('Oil, Gas & Consumable Fuels', 'N'),
    '10102030': ('Oil, Gas & Consumable Fuels', 'N'),
    '20201040': ('Professional Services', 'Y'),
    '20301010': ('Air Freight & Logistics', 'N'),
    '25202010': ('Leisure Products', 'N'), # Leisure Products
    '25202020': ('Leisure Products', 'N'), # Photographic Products
    '25203010': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25203020': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25203030': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25502010': ('Internet & Direct Marketing Retail', 'N'), # Catalog Retail (Discontinued)
    '25502020': ('Internet & Direct Marketing Retail', 'N'), # Internet Retail (Renamed)
    '30101010': ('Food & Staples Retailing', 'N'),
    '30101020': ('Food & Staples Retailing', 'N'),
    '30101030': ('Food & Staples Retailing', 'N'),
    '35102010': ('Health Care Providers & Services', 'N'),
    '35102015': ('Health Care Providers & Services', 'N'),
    '35102020': ('Health Care Providers & Services', 'N'),
    '35102030': ('Health Care Providers & Services', 'N'),
    '40101010': ('Banks', 'N'), # Diversified Bank
    '40101015': ('Banks', 'N'), # Regional Banks
    '40201010': ('Diversified Financial Services', 'N'),
    '40201020': ('Diversified Financial Services', 'N'),
    '40201030': ('Diversified Financial Services', 'N'),
    '40401010': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Real Estate Investment Trusts
    '40401020': ('Real Estate Management & Development', 'N'), # Real Estate Management & Development
    '40402010': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Diversified REITs
    '40402020': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Industrial REITs
    '40402035': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Hotel & Resort REITs
    '40402040': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Office REITs
    '40402045': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Health Care REITs
    '40402050': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Residential REITs
    '40402060': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Retail REITs
    '40402070': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Specialized REITs
    '40402030': ('Mortgage Real Estate Investment Trusts (REITs)', 'N'), # Mortgage REITs <-- EXCLUDED FROM EQUITY REITS
    '45101010': ('Internet & Direct Marketing Retail', 'N'), # 2018 Addition
    '45102010': ('IT Services', 'N'),
    '45202010': ('Technology Hardware, Storage & Peripherals', 'N'), # Computer Hardware
    '45202020': ('Technology Hardware, Storage & Peripherals', 'N'), # Computer Storage & Peripherals
    '45204010': ('Technology Hardware, Storage & Peripherals', 'N'), # Office Electronics
    '45203010': ('Electronic Equipment, Instruments & Components', 'N'), # Electronic Equipment & Instruments
    '45203020': ('Electronic Equipment, Instruments & Components', 'N'), # Electronic Manufacturing Services
    '45203030': ('Electronic Equipment, Instruments & Components', 'N'), # Technology Distributors
    '45205010': ('Semiconductors & Semiconductor Equipment', 'Y'),
    '45205020': ('Semiconductors & Semiconductor Equipment', 'Y'),
    '55103010': ('Multi-Utilities', 'N'),
    '55105010': ('Independent Power and Renewable Electricity Producers', 'N'), # Independent Power Producers & Energy Traders
    '55105020': ('Independent Power and Renewable Electricity Producers', 'N'),
    }

guessMaps[('GICSIndustryGroups', datetime.date(2006,4,29))] = {
    '20201010': ('Commercial Services & Supplies', 'N'),
    '20201020': ('Software & Services', 'Y'),
    '20201050': ('Commercial Services & Supplies', 'N'),
    '20201060': ('Commercial Services & Supplies', 'N'),
    '20201070': ('Commercial Services & Supplies', 'N'),
    '20201080': ('Commercial Services & Supplies', 'N'),
    '20202010': ('Commercial Services & Supplies', 'N'),
    '20202020': ('Commercial Services & Supplies', 'N'),
    '25301010': ('Consumer Services', 'N'),
    '25301020': ('Consumer Services', 'N'),
    '25301030': ('Consumer Services', 'N'),
    '25301040': ('Consumer Services', 'N'),
    '30101010': ('Food & Staples Retailing', 'N'),
    '30101020': ('Food & Staples Retailing', 'N'),
    '30101030': ('Food & Staples Retailing', 'N'),
    '35201010': ('Pharmaceuticals, Biotechnology & Life Sciences', 'N'),
    '35202010': ('Pharmaceuticals, Biotechnology & Life Sciences', 'N'),
    '45205010': ('Semiconductors & Semiconductor Equipment', 'Y'),
    '45205020': ('Semiconductors & Semiconductor Equipment', 'Y'),
    '50201010': ('Media', 'Y'),
    '50201020': ('Media', 'Y'),
    '50201030': ('Media', 'Y'),
    '50201040': ('Media', 'Y'),
    '50202010': ('Media', 'Y'),
    '50202020': ('Software & Services', 'Y'),
    '50203010': ('Software & Services', 'Y')
    }

guessMaps[('GICSIndustryGroups', datetime.date(2008,8,30))] = {
    '20201010': ('Commercial & Professional Services', 'N'),
    '20201020': ('Software & Services', 'Y'),
    '20201030': ('Commercial & Professional Services', 'N'),
    '20201040': ('Commercial & Professional Services', 'N'),
    '20201050': ('Commercial & Professional Services', 'N'),
    '20201060': ('Commercial & Professional Services', 'N'),
    '20201070': ('Commercial & Professional Services', 'N'),
    '20201080': ('Commercial & Professional Services', 'N'),
    '25301010': ('Consumer Services', 'N'),
    '25301020': ('Consumer Services', 'N'),
    '25301030': ('Consumer Services', 'N'),
    '25301040': ('Consumer Services', 'N'),
    '30101010': ('Food & Staples Retailing', 'N'),
    '30101020': ('Food & Staples Retailing', 'N'),
    '30101030': ('Food & Staples Retailing', 'N'),
    '35201010': ('Pharmaceuticals, Biotechnology & Life Sciences', 'N'),
    '35202010': ('Pharmaceuticals, Biotechnology & Life Sciences', 'N'),
    '45205010': ('Semiconductors & Semiconductor Equipment', 'Y'),
    '45205020': ('Semiconductors & Semiconductor Equipment', 'Y'),
    '50201010': ('Media', 'Y'),
    '50201020': ('Media', 'Y'),
    '50201030': ('Media', 'Y'),
    '50201040': ('Media', 'Y'),
    '50202010': ('Media', 'Y'),
    '50202020': ('Software & Services', 'Y'),
    '50203010': ('Software & Services', 'Y'),
    }

guessMaps[('GICSIndustryGroups', datetime.date(2016,9,1))] = {
    '20201010': ('Commercial & Professional Services', 'N'),
    '20201020': ('Software & Services', 'Y'),
    '20201030': ('Commercial & Professional Services', 'N'),
    '20201040': ('Commercial & Professional Services', 'N'),
    '20201050': ('Commercial & Professional Services', 'N'),
    '20201060': ('Commercial & Professional Services', 'N'),
    '20201070': ('Commercial & Professional Services', 'N'),
    '20201080': ('Commercial & Professional Services', 'N'),
    '25301010': ('Consumer Services', 'N'),
    '25301020': ('Consumer Services', 'N'),
    '25301030': ('Consumer Services', 'N'),
    '25301040': ('Consumer Services', 'N'),
    '30101010': ('Food & Staples Retailing', 'N'),
    '30101020': ('Food & Staples Retailing', 'N'),
    '30101030': ('Food & Staples Retailing', 'N'),
    '35201010': ('Pharmaceuticals, Biotechnology & Life Sciences', 'N'),
    '35202010': ('Pharmaceuticals, Biotechnology & Life Sciences', 'N'),
    '45205010': ('Semiconductors & Semiconductor Equipment', 'Y'),
    '45205020': ('Semiconductors & Semiconductor Equipment', 'Y'),
    '50201010': ('Media', 'Y'),
    '50201020': ('Media', 'Y'),
    '50201030': ('Media', 'Y'),
    '50201040': ('Media', 'Y'),
    '50202010': ('Media', 'Y'),
    '50202020': ('Software & Services', 'Y'),
    '50203010': ('Software & Services', 'Y'),
    }

guessMaps[('GICSIndustryGroups', datetime.date(2018,9,29))] = {
    '20201010': ('Commercial & Professional Services', 'N'),
    '20201020': ('Software & Services', 'Y'),
    '20201030': ('Commercial & Professional Services', 'N'),
    '20201040': ('Commercial & Professional Services', 'N'),
    '20201050': ('Commercial & Professional Services', 'N'),
    '20201060': ('Commercial & Professional Services', 'N'),
    '20201070': ('Commercial & Professional Services', 'N'),
    '20201080': ('Commercial & Professional Services', 'N'),
    '25301010': ('Consumer Services', 'N'),
    '25301020': ('Consumer Services', 'N'),
    '25301030': ('Consumer Services', 'N'),
    '25301040': ('Consumer Services', 'N'),
    '25401010': ('Media & Entertainment', 'N'),
    '25401020': ('Media & Entertainment', 'N'),
    '25401025': ('Media & Entertainment', 'N'),
    '25401030': ('Media & Entertainment', 'N'),
    '25401040': ('Media & Entertainment', 'N'),
    '30101010': ('Food & Staples Retailing', 'N'),
    '30101020': ('Food & Staples Retailing', 'N'),
    '30101030': ('Food & Staples Retailing', 'N'),
    '35201010': ('Pharmaceuticals, Biotechnology & Life Sciences', 'N'),
    '35202010': ('Pharmaceuticals, Biotechnology & Life Sciences', 'N'),
    '45205010': ('Semiconductors & Semiconductor Equipment', 'Y'),
    '45205020': ('Semiconductors & Semiconductor Equipment', 'Y'),
    }

guessMaps[('GICSCustom-CA', datetime.date(2008,8,30))] = {
    # Chemicals
    '15101010': ('Materials ex Metals, Mining & Forestry', 'Y'),
    '15101020': ('Materials ex Metals, Mining & Forestry', 'Y'),
    '15101030': ('Materials ex Metals, Mining & Forestry', 'Y'),
    '15101040': ('Materials ex Metals, Mining & Forestry', 'Y'),
    '15101050': ('Materials ex Metals, Mining & Forestry', 'Y'),
    # Construction Materials
    '15102010': ('Materials ex Metals, Mining & Forestry', 'Y'),
    # Containers & Packaging
    '15103010': ('Materials ex Metals, Mining & Forestry', 'Y'),
    '15103020': ('Materials ex Metals, Mining & Forestry', 'Y'),
    # Metals & Mining
    '15104010': ('Metals & Mining ex Gold', 'Y'),
    '15104020': ('Metals & Mining ex Gold', 'Y'),
    '15104025': ('Metals & Mining ex Gold', 'Y'),
    '15104030': ('Gold', 'N'),
    '15104040': ('Metals & Mining ex Gold', 'Y'),
    '15104045': ('Metals & Mining ex Gold', 'Y'),
    '15104050': ('Metals & Mining ex Gold', 'Y'),
    # Paper & Forest Products
    '15105010': ('Paper & Forest Products', 'Y'),
    '15105020': ('Paper & Forest Products', 'Y'),
    # Commercial Services & Supplies
    '20201010': ('Commercial & Professional Services', 'N'),
    '20201020': ('Software & Services', 'Y'),
    '20201030': ('Commercial & Professional Services', 'N'),
    '20201040': ('Commercial & Professional Services', 'N'),
    '20201050': ('Commercial & Professional Services', 'N'),
    '20201060': ('Commercial & Professional Services', 'N'),
    # Automobiles & Components
    '25101010': ('Consumer Discretionary ex Media', 'Y'),
    '25101020': ('Consumer Discretionary ex Media', 'Y'),
    '25102010': ('Consumer Discretionary ex Media', 'Y'),
    '25102020': ('Consumer Discretionary ex Media', 'Y'),
    # Consumer Durables & Apparel
    '25201010': ('Consumer Discretionary ex Media', 'Y'),
    '25201020': ('Consumer Discretionary ex Media', 'Y'),
    '25201030': ('Consumer Discretionary ex Media', 'Y'),
    '25201040': ('Consumer Discretionary ex Media', 'Y'),
    '25201050': ('Consumer Discretionary ex Media', 'Y'),
    '25202010': ('Consumer Discretionary ex Media', 'Y'),
    '25202020': ('Consumer Discretionary ex Media', 'Y'),
    '25203010': ('Consumer Discretionary ex Media', 'Y'),
    '25203020': ('Consumer Discretionary ex Media', 'Y'),
    '25203030': ('Consumer Discretionary ex Media', 'Y'),
    # Consumer Services
    '25301010': ('Consumer Discretionary ex Media', 'Y'),
    '25301020': ('Consumer Discretionary ex Media', 'Y'),
    '25301030': ('Consumer Discretionary ex Media', 'Y'),
    '25301040': ('Consumer Discretionary ex Media', 'Y'),
    '25302010': ('Consumer Discretionary ex Media', 'Y'),
    '25302020': ('Consumer Discretionary ex Media', 'Y'),
    # Retailing
    '25501010': ('Consumer Discretionary ex Media', 'Y'),
    '25502010': ('Consumer Discretionary ex Media', 'Y'),
    '25502020': ('Consumer Discretionary ex Media', 'Y'),
    '25503010': ('Consumer Discretionary ex Media', 'Y'),
    '25503020': ('Consumer Discretionary ex Media', 'Y'),
    '25504010': ('Consumer Discretionary ex Media', 'Y'),
    '25504020': ('Consumer Discretionary ex Media', 'Y'),
    '25504030': ('Consumer Discretionary ex Media', 'Y'),
    '25504040': ('Consumer Discretionary ex Media', 'Y'),
    '25504050': ('Consumer Discretionary ex Media', 'Y'),
    '25504060': ('Consumer Discretionary ex Media', 'Y'),
    # Food & Staples Retailing
    '30101010': ('Food & Staples Retailing', 'N'),
    '30101020': ('Food & Staples Retailing', 'N'),
    '30101030': ('Food & Staples Retailing', 'N'),
    # Food, Beverage & Tobacco
    '30201010': ('Food & Staples Products', 'Y'),
    '30201020': ('Food & Staples Products', 'Y'),
    '30201030': ('Food & Staples Products', 'Y'),
    '30202010': ('Food & Staples Products', 'Y'),
    '30202020': ('Food & Staples Products', 'Y'),
    '30202030': ('Food & Staples Products', 'Y'),
    '30203010': ('Food & Staples Products', 'Y'),
    # Household & Personal Products
    '30301010': ('Food & Staples Products', 'Y'),
    '30302010': ('Food & Staples Products', 'Y'),
    # Health Care Equipment & Services
    '35101010': ('Health Care', 'N'),
    '35101020': ('Health Care', 'N'),
    '35102010': ('Health Care', 'N'),
    '35102015': ('Health Care', 'N'),
    '35102020': ('Health Care', 'N'),
    '35102030': ('Health Care', 'N'),
    '35103010': ('Health Care', 'N'),
    # Pharmaceuticals, Biotechnology & Life Sciences
    '35201010': ('Health Care', 'N'),
    '35202010': ('Health Care', 'N'),
    '35203010': ('Health Care', 'N'),
    # GICS 2016 - Mortgage REITs
    '40204010': ('Real Estate', 'N'), 
    # Technology Hardware & Equipment
    '45201010': ('Technology Hardware', 'Y'),
    '45201020': ('Technology Hardware', 'Y'),
    '45202010': ('Technology Hardware', 'Y'),
    '45202020': ('Technology Hardware', 'Y'),
    '45203010': ('Technology Hardware', 'Y'),
    '45203015': ('Technology Hardware', 'Y'),
    '45203020': ('Technology Hardware', 'Y'),
    '45203030': ('Technology Hardware', 'Y'),
    '45204010': ('Technology Hardware', 'Y'),
    # Semiconductors
    '45205010': ('Technology Hardware', 'Y'),
    '45205020': ('Technology Hardware', 'Y'),
    '45202030': ('Technology Hardware', 'Y'),
    '45301010': ('Technology Hardware', 'Y'),
    '45301020': ('Technology Hardware', 'Y'),
    # Media & Entertainment
    '50201010': ('Media', 'Y'),
    '50201020': ('Media', 'Y'),
    '50201030': ('Media', 'Y'),
    '50201040': ('Media', 'Y'),
    '50202010': ('Media', 'Y'),
    '50202020': ('Software & Services', 'Y'),
    '50203010': ('Software & Services', 'Y')
}

guessMaps[('GICSCustom-CA2', datetime.date(2018,9,29))] = {
    # ##### GICS revision 20/1050-01-01 
    # Chemicals
    '15101010': ('Materials ex Metals, Mining & Forestry', 'Y'),
    '15101020': ('Materials ex Metals, Mining & Forestry', 'Y'),
    '15101030': ('Materials ex Metals, Mining & Forestry', 'Y'),
    '15101040': ('Materials ex Metals, Mining & Forestry', 'Y'),
    '15101050': ('Materials ex Metals, Mining & Forestry', 'Y'),
    # Construction Materials
    '15102010': ('Materials ex Metals, Mining & Forestry', 'Y'),
    # Containers & Packaging
    '15103010': ('Materials ex Metals, Mining & Forestry', 'Y'),
    '15103020': ('Materials ex Metals, Mining & Forestry', 'Y'),
    # Metals & Mining
    '15104010': ('Metals & Mining ex Gold', 'Y'),
    '15104020': ('Metals & Mining ex Gold', 'Y'),
    '15104025': ('Metals & Mining ex Gold', 'Y'),
    '15104030': ('Gold', 'N'),
    '15104040': ('Metals & Mining ex Gold', 'Y'),
    '15104045': ('Metals & Mining ex Gold', 'Y'),
    '15104050': ('Metals & Mining ex Gold', 'Y'),
    # Paper & Forest Products
    '15105010': ('Paper & Forest Products', 'N'),
    '15105020': ('Paper & Forest Products', 'N'),
    # Commercial Services & Supplies
    '20201010': ('Commercial & Professional Services', 'N'),
    '20201020': ('Software & Services', 'N'),
    '20201030': ('Commercial & Professional Services', 'N'),
    '20201040': ('Commercial & Professional Services', 'N'),
    '20201050': ('Commercial & Professional Services', 'N'),
    '20201060': ('Commercial & Professional Services', 'N'),
    # Automobiles & Components
    '25101010': ('Consumer Discretionary', 'N'),
    '25101020': ('Consumer Discretionary', 'N'),
    '25102010': ('Consumer Discretionary', 'N'),
    '25102020': ('Consumer Discretionary', 'N'),
    # Consumer Durables & Apparel
    '25201010': ('Consumer Discretionary', 'N'),
    '25201020': ('Consumer Discretionary', 'N'),
    '25201030': ('Consumer Discretionary', 'N'),
    '25201040': ('Consumer Discretionary', 'N'),
    '25201050': ('Consumer Discretionary', 'N'),
    '25202010': ('Consumer Discretionary', 'N'),
    '25202020': ('Consumer Discretionary', 'N'),
    '25203010': ('Consumer Discretionary', 'N'),
    '25203020': ('Consumer Discretionary', 'N'),
    '25203030': ('Consumer Discretionary', 'N'),
    # Consumer Services
    '25301010': ('Consumer Discretionary', 'N'),
    '25301020': ('Consumer Discretionary', 'N'),
    '25301030': ('Consumer Discretionary', 'N'),
    '25301040': ('Consumer Discretionary', 'N'),
    '25302010': ('Consumer Discretionary', 'N'),
    '25302020': ('Consumer Discretionary', 'N'),
    # Retailing
    '25501010': ('Consumer Discretionary', 'N'),
    '25502010': ('Consumer Discretionary', 'N'),
    '25502020': ('Consumer Discretionary', 'N'),
    '25503010': ('Consumer Discretionary', 'N'),
    '25503020': ('Consumer Discretionary', 'N'),
    '25504010': ('Consumer Discretionary', 'N'),
    '25504020': ('Consumer Discretionary', 'N'),
    '25504030': ('Consumer Discretionary', 'N'),
    '25504040': ('Consumer Discretionary', 'N'),
    '25504050': ('Consumer Discretionary', 'N'),
    '25504060': ('Consumer Discretionary', 'N'),
    # Food & Staples Retailing
    '30101010': ('Food & Staples Retailing', 'N'),
    '30101020': ('Food & Staples Retailing', 'N'),
    '30101030': ('Food & Staples Retailing', 'N'),
    # Food, Beverage & Tobacco
    '30201010': ('Food & Staples Products', 'Y'),
    '30201020': ('Food & Staples Products', 'Y'),
    '30201030': ('Food & Staples Products', 'Y'),
    '30202010': ('Food & Staples Products', 'Y'),
    '30202020': ('Food & Staples Products', 'Y'),
    '30202030': ('Food & Staples Products', 'Y'),
    '30203010': ('Food & Staples Products', 'Y'),
    # Household & Personal Products
    '45102030': ('Food & Staples Products', 'Y'),
    '30302010': ('Food & Staples Products', 'Y'),
    # Health Care Equipment & Services
    '35101010': ('Health Care', 'N'),
    '35101020': ('Health Care', 'N'),
    '35102010': ('Health Care', 'N'),
    '35102015': ('Health Care', 'N'),
    '35102020': ('Health Care', 'N'),
    '35102030': ('Health Care', 'N'),
    '35103010': ('Health Care', 'N'),
    # Pharmaceuticals, Biotechnology & Life Sciences
    '35201010': ('Health Care', 'N'),
    '35202010': ('Health Care', 'N'),
    '35203010': ('Health Care', 'N'),
    # Technology Hardware & Equipment
    '45201010': ('Technology Hardware', 'Y'),
    '45201020': ('Technology Hardware', 'Y'),
    '45202010': ('Technology Hardware', 'Y'),
    '45202020': ('Technology Hardware', 'Y'),
    '45203010': ('Technology Hardware', 'Y'),
    '45203015': ('Technology Hardware', 'Y'),
    '45203020': ('Technology Hardware', 'Y'),
    '45203030': ('Technology Hardware', 'Y'),
    '45204010': ('Technology Hardware', 'Y'),
    # Semiconductors
    '45205010': ('Technology Hardware', 'Y'),
    '45205020': ('Technology Hardware', 'Y'),
    '45202030': ('Technology Hardware', 'Y'),
    '45301010': ('Technology Hardware', 'Y'),
    '45301020': ('Technology Hardware', 'Y'),
    # Media & Entertainment
    '25401010': ('Media & Entertainment', 'N'),
    '25401020': ('Media & Entertainment', 'N'),
    '25401030': ('Media & Entertainment', 'N'),
    '25401040': ('Media & Entertainment', 'N'),
    '25401025': ('Media & Entertainment', 'N'),
}

guessMaps[('GICSCustom-CA3', datetime.date(2018,9,29))] = {
    # Capital Goods
    '20107010': ('Industrials', 'N'),
    '20106010': ('Industrials', 'N'),
    '20106020': ('Industrials', 'N'),
    '20105010': ('Industrials', 'N'),
    '20104010': ('Industrials', 'N'),
    '20104020': ('Industrials', 'N'),
    '20103010': ('Industrials', 'N'),
    '20102010': ('Industrials', 'N'),
    '20101010': ('Industrials', 'N'),
    # Commercial Services & Supplies
    '20201010': ('Industrials', 'N'),
    '20201020': ('Industrials', 'N'),
    '20201030': ('Industrials', 'N'),
    '20201040': ('Industrials', 'N'),
    '20201050': ('Industrials', 'N'),
    '20201060': ('Industrials', 'N'),
    # Transportation
    '20305010': ('Industrials', 'N'),
    '20305020': ('Industrials', 'N'),
    '20305030': ('Industrials', 'N'),
    '20304010': ('Industrials', 'N'),
    '20304020': ('Industrials', 'N'),
    '20303010': ('Industrials', 'N'),
    '20302010': ('Industrials', 'N'),
    '20301010': ('Industrials', 'N'),
    # Inudstrials other
    '20202010': ('Industrials', 'N'),
    '20202020': ('Industrials', 'N'),
    '20201070': ('Industrials', 'N'),
    '20201080': ('Industrials', 'N'),
    '20106015': ('Industrials', 'N'),

    # Automobiles & Components
    '25102010': ('Consumer Discretionary', 'N'),
    '25102020': ('Consumer Discretionary', 'N'),
    '25101010': ('Consumer Discretionary', 'N'),
    '25101020': ('Consumer Discretionary', 'N'),
    # Consumer Durables & Apparel
    '25203010': ('Consumer Discretionary', 'N'),
    '25203020': ('Consumer Discretionary', 'N'),
    '25203030': ('Consumer Discretionary', 'N'),
    '25202010': ('Consumer Discretionary', 'N'),
    '25202020': ('Consumer Discretionary', 'N'),
    '25201010': ('Consumer Discretionary', 'N'),
    '25201020': ('Consumer Discretionary', 'N'),
    '25201030': ('Consumer Discretionary', 'N'),
    '25201040': ('Consumer Discretionary', 'N'),
    '25201050': ('Consumer Discretionary', 'N'),
     # Hotels Restaurants & Leisure
    '25301010': ('Consumer Discretionary', 'N'),
    '25301020': ('Consumer Discretionary', 'N'),
    '25301030': ('Consumer Discretionary', 'N'),
    '25301040': ('Consumer Discretionary', 'N'),
    # Retailing
    '25504010': ('Consumer Discretionary', 'N'),
    '25504020': ('Consumer Discretionary', 'N'),
    '25504030': ('Consumer Discretionary', 'N'),
    '25504040': ('Consumer Discretionary', 'N'),
    '25503010': ('Consumer Discretionary', 'N'),
    '25503020': ('Consumer Discretionary', 'N'),
    '25502010': ('Consumer Discretionary', 'N'),
    '25502020': ('Consumer Discretionary', 'N'),
    '25501010': ('Consumer Discretionary', 'N'),
    # Consumer discretionary - other
    '25302010': ('Consumer Discretionary', 'N'),
    '25302020': ('Consumer Discretionary', 'N'),
    '25504050': ('Consumer Discretionary', 'N'),
    '25504060': ('Consumer Discretionary', 'N'),

    # Food & Drug Retailing
    '30101010': ('Consumer Staples', 'N'),
    '30101020': ('Consumer Staples', 'N'),
    '30101030': ('Consumer Staples', 'N'),
    # Food Beverage & Tobacco
    '30203010': ('Consumer Staples', 'N'),
    '30202010': ('Consumer Staples', 'N'),
    '30202020': ('Consumer Staples', 'N'),
    '30202030': ('Consumer Staples', 'N'),
    '30201010': ('Consumer Staples', 'N'),
    '30201020': ('Consumer Staples', 'N'),
    '30201030': ('Consumer Staples', 'N'),
    # Household & Personal Product
    '30302010': ('Consumer Staples', 'N'),
    '30301010': ('Consumer Staples', 'N'),
    # other (consumer staples 
    '30101040': ('Consumer Staples', 'N'),


    # Health Care Equipment & Services
    '35102010': ('Health Care', 'N'),
    '35102020': ('Health Care', 'N'),
    '35102030': ('Health Care', 'N'),
    '35101010': ('Health Care', 'N'),
    '35101020': ('Health Care', 'N'),
    '35102015': ('Health Care', 'N'),
    # Pharmaceuticals & Biotechnology
    '35202010': ('Health Care', 'N'),
    '35201010': ('Health Care', 'N'),
    # health care (ohter)
    '35103010': ('Health Care', 'N'),
    '35203010': ('Health Care', 'N'),
    '35103010': ('Health Care', 'N'),
    '35203010': ('Health Care', 'N'),

    # Banks
    '40101010': ('Financials', 'N'),
    '40102010': ('Financials', 'N'),
    '40101015': ('Financials', 'N'),
    # Diversified Financials
    '40201010': ('Financials', 'N'),
    '40201020': ('Financials', 'N'),
    '40201030': ('Financials', 'N'),
    '40203010': ('Financials', 'N'),
    '40203020': ('Financials', 'N'),
    '40203030': ('Financials', 'N'),
    '40202010': ('Financials', 'N'),
    '40201040': ('Financials', 'N'),
    # Insurance
    '40301010': ('Financials', 'N'),
    '40301020': ('Financials', 'N'),
    '40301030': ('Financials', 'N'),
    '40301040': ('Financials', 'N'),
    '40301050': ('Financials', 'N'),
    '40204010': ('Financials', 'N'), # mortgage reits
    '40203040': ('Financials', 'N'),

    # IT
    '45103010': ('Information Technology', 'N'), # software and services
    '45103020': ('Information Technology', 'N'), # software and services
    '45102010': ('Information Technology', 'N'), # IT consulting
    '45205010': ('Information Technology', 'N'), # semiconductor equipment
    '45205020': ('Information Technology', 'N'), # semiconductor equipment
    '45204010': ('Information Technology', 'N'), # office electronics
    '45203010': ('Information Technology', 'N'), # electronic equipment
    '45202010': ('Information Technology', 'N'), # computer hardware
    '45202020': ('Information Technology', 'N'), # computer storage
    '45201010': ('Information Technology', 'N'), # networking equipment
    '45201020': ('Information Technology', 'N'), # telecom equipment
    '45103030': ('Information Technology', 'N'), # home entertainment software
    '45102020': ('Information Technology', 'N'), 
    '45203020': ('Information Technology', 'N'), 
    '45203030': ('Information Technology', 'N'), 
    '45301010': ('Information Technology', 'N'), 
    '45301020': ('Information Technology', 'N'), 
    '45203015': ('Information Technology', 'N'), 
    '45202030': ('Information Technology', 'N'), 
    '45102030': ('Information Technology', 'N'), 

    # Wireless telecom
    '50102010': ('Communication Services', 'N'),
    '50101010': ('Communication Services', 'N'),
    '50101020': ('Communication Services', 'N'),


    # Media & Entertainment
    '25401010': ('Communication Services', 'N'),
    '25401020': ('Communication Services', 'N'),
    '25401030': ('Communication Services', 'N'),
    '25401040': ('Communication Services', 'N'),
    '25401025': ('Communication Services', 'N'),

    # Medai & Entertainment
    '50203010': ('Communication Services', 'N'),
    '50202010': ('Communication Services', 'N'),
    '50202020': ('Communication Services', 'N'),
    '50201010': ('Communication Services', 'N'),
    '50201020': ('Communication Services', 'N'),
    '50201030': ('Communication Services', 'N'),
    '50201040': ('Communication Services', 'N'),

    # Internet Software & Services  (discontinued)
    '45101010': ('Communication Services', 'N'), # 2018 Internet & Direct Marketing Retail
}

guessMaps[('GICSCustom-CA4', datetime.date(2018,9,29))] = {
    # Chemicals
    '15101010': ('Materials ex Metals, Mining & Forestry', 'Y'),
    '15101020': ('Materials ex Metals, Mining & Forestry', 'Y'),
    '15101030': ('Materials ex Metals, Mining & Forestry', 'Y'),
    '15101040': ('Materials ex Metals, Mining & Forestry', 'Y'),
    '15101050': ('Materials ex Metals, Mining & Forestry', 'Y'),
    # Construction Materials
    '15102010': ('Materials ex Metals, Mining & Forestry', 'Y'),
    # Containers & Packaging
    '15103010': ('Materials ex Metals, Mining & Forestry', 'Y'),
    '15103020': ('Materials ex Metals, Mining & Forestry', 'Y'),
    # Metals & Mining
    '15104010': ('Metals & Mining ex Gold', 'Y'),
    '15104020': ('Metals & Mining ex Gold', 'Y'),
    '15104025': ('Metals & Mining ex Gold', 'Y'),
    '15104030': ('Gold', 'N'),
    '15104040': ('Metals & Mining ex Gold', 'Y'),
    '15104045': ('Metals & Mining ex Gold', 'Y'),
    '15104050': ('Metals & Mining ex Gold', 'Y'),
    # Paper & Forest Products
    '15105010': ('Paper & Forest Products', 'N'),
    '15105020': ('Paper & Forest Products', 'N'),
    # Commercial Services & Supplies
    '20201010': ('Commercial & Professional Services', 'N'),
    '20201020': ('Software & Services', 'N'),
    '20201030': ('Commercial & Professional Services', 'N'),
    '20201040': ('Commercial & Professional Services', 'N'),
    '20201050': ('Commercial & Professional Services', 'N'),
    '20201060': ('Commercial & Professional Services', 'N'),
    # Automobiles & Components
    '25101010': ('Consumer Discretionary', 'N'),
    '25101020': ('Consumer Discretionary', 'N'),
    '25102010': ('Consumer Discretionary', 'N'),
    '25102020': ('Consumer Discretionary', 'N'),
    # Consumer Durables & Apparel
    '25201010': ('Consumer Discretionary', 'N'),
    '25201020': ('Consumer Discretionary', 'N'),
    '25201030': ('Consumer Discretionary', 'N'),
    '25201040': ('Consumer Discretionary', 'N'),
    '25201050': ('Consumer Discretionary', 'N'),
    '25202010': ('Consumer Discretionary', 'N'),
    '25202020': ('Consumer Discretionary', 'N'),
    '25203010': ('Consumer Discretionary', 'N'),
    '25203020': ('Consumer Discretionary', 'N'),
    '25203030': ('Consumer Discretionary', 'N'),
    # Consumer Services
    '25301010': ('Consumer Discretionary', 'N'),
    '25301020': ('Consumer Discretionary', 'N'),
    '25301030': ('Consumer Discretionary', 'N'),
    '25301040': ('Consumer Discretionary', 'N'),
    '25302010': ('Consumer Discretionary', 'N'),
    '25302020': ('Consumer Discretionary', 'N'),
    # Retailing
    '25501010': ('Consumer Discretionary', 'N'),
    '25502010': ('Consumer Discretionary', 'N'),
    '25502020': ('Consumer Discretionary', 'N'),
    '25503010': ('Consumer Discretionary', 'N'),
    '25503020': ('Consumer Discretionary', 'N'),
    '25504010': ('Consumer Discretionary', 'N'),
    '25504020': ('Consumer Discretionary', 'N'),
    '25504030': ('Consumer Discretionary', 'N'),
    '25504040': ('Consumer Discretionary', 'N'),
    '25504050': ('Consumer Discretionary', 'N'),
    '25504060': ('Consumer Discretionary', 'N'),
    # Food & Staples Retailing
    '30101010': ('Food & Staples Retailing', 'N'),
    '30101020': ('Food & Staples Retailing', 'N'),
    '30101030': ('Food & Staples Retailing', 'N'),
    # Food, Beverage & Tobacco
    '30201010': ('Food & Staples Products', 'Y'),
    '30201020': ('Food & Staples Products', 'Y'),
    '30201030': ('Food & Staples Products', 'Y'),
    '30202010': ('Food & Staples Products', 'Y'),
    '30202020': ('Food & Staples Products', 'Y'),
    '30202030': ('Food & Staples Products', 'Y'),
    '30203010': ('Food & Staples Products', 'Y'),
    '30301010': ('Food & Staples Products', 'Y'), # Household Products-B, Household & Personal Products
    # Household & Personal Products
    '45102030': ('Software & Services', 'N'), # used to be 'Food & Staples Products' when model was populated
    '30302010': ('Food & Staples Products', 'Y'),
    # Health Care Equipment & Services
    '35101010': ('Health Care', 'N'),
    '35101020': ('Health Care', 'N'),
    '35102010': ('Health Care', 'N'),
    '35102015': ('Health Care', 'N'),
    '35102020': ('Health Care', 'N'),
    '35102030': ('Health Care', 'N'),
    '35103010': ('Health Care', 'N'),
    # Pharmaceuticals, Biotechnology & Life Sciences
    '35201010': ('Health Care', 'N'),
    '35202010': ('Health Care', 'N'),
    '35203010': ('Health Care', 'N'),
    # Technology Hardware & Equipment
    '45201010': ('Technology Hardware', 'Y'),
    '45201020': ('Technology Hardware', 'Y'),
    '45202010': ('Technology Hardware', 'Y'),
    '45202020': ('Technology Hardware', 'Y'),
    '45203010': ('Technology Hardware', 'Y'),
    '45203015': ('Technology Hardware', 'Y'),
    '45203020': ('Technology Hardware', 'Y'),
    '45203030': ('Technology Hardware', 'Y'),
    '45204010': ('Technology Hardware', 'Y'),
    # Semiconductors
    '45205010': ('Technology Hardware', 'Y'),
    '45205020': ('Technology Hardware', 'Y'),
    '45202030': ('Technology Hardware', 'Y'),
    '45301010': ('Technology Hardware', 'Y'),
    '45301020': ('Technology Hardware', 'Y'),
    # Media & Entertainment
    '25401010': ('Media & Entertainment', 'N'),
    '25401020': ('Media & Entertainment', 'N'),
    '25401030': ('Media & Entertainment', 'N'),
    '25401040': ('Media & Entertainment', 'N'),
    '25401025': ('Media & Entertainment', 'N'),
    # Oil, Gas & Consumable Fuels
    '10102010': ('Oil, Gas & Consumable Fuels', 'N'), # Integrated Oil & Gas-B, Energy
    '10102020': ('Oil, Gas & Consumable Fuels', 'N'), # Oil & Gas Exploration & Production-B, Energy
    '10102030': ('Oil, Gas & Consumable Fuels', 'N'), #Oil & Gas Refining & Marketing-B, Energy
    '10102040': ('Oil, Gas & Consumable Fuels', 'N'), # Oil & Gas Storage & Transportation-B, Energy
    '10102050': ('Oil, Gas & Consumable Fuels', 'N'), # Coal & Consumable Fuels-B, Energy
    # Energy Equityment & Services
    '10101010': ('Energy Equipment & Services', 'N'), # Oil & Gas Drilling-B, Energy
    '10101020': ('Energy Equipment & Services', 'N'), # Oil & Gas Equipment & Services-B, Energy
    # Equity Real Estate Investment Trusts (REITs)
    '40401010': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Real Estate Investment Trusts-B, Real Estate
    '40402010': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Diversified REITs-B, Real Estate
    '40402020': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Industrial REITs-B, Real Estate
    '40402030': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Mortgage REITs-B, Real Estate
    '40402040': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Office REITs-B, Real Estate
    '40402050': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Residential REITs-B, Real Estate
    '40402060': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Retail REITs-B, Real Estate
    '40402070': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Specialized REITs-B, Real Estate
    '40402035': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Hotel & Resort REITs-B, Real Estate
    '40402045': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Health Care REITs-B, Real Estate
    '60101010': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Diversified REITs-B, Real Estate
    '60101020': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Industrial REITs-B, Real Estate
    '60101030': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Hotel & Resort REITs-B, Real Estate
    '60101040': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Office REITs-B, Real Estate
    '60101050': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Health Care REITs-B, Real Estate
    '60101060': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Residential REITs-B, Real Estate
    '60101070': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Retail REITs-B, Real Estate
    '60101080': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Specialized REITs-B, Real Estate
    # Real Estate Management & Development
    '40401020': ('Real Estate Management & Development', 'N'), # Real Estate Management & Development-B, Real Estate
    '40403010': ('Real Estate Management & Development', 'N'), # Real Estate Management & Development-B, Real Estate
    '40403020': ('Real Estate Management & Development', 'N'), # Real Estate Operating Companies-B, Real Estate
    '40403030': ('Real Estate Management & Development', 'N'), # Real Estate Development-B, Real Estate
    '40403040': ('Real Estate Management & Development', 'N'), # Real Estate Services-B, Real Estate
    '60102010': ('Real Estate Management & Development', 'N'), # Diversified Real Estate Activities-B, Real Estate
    '60102020': ('Real Estate Management & Development', 'N'), # Real Estate Operating Companies-B, Real Estate
    '60102030': ('Real Estate Management & Development', 'N'), # Real Estate Development-B, Real Estate
    '60102040': ('Real Estate Management & Development', 'N'), # Real Estate Services-B, Real Estate
}

guessMaps[('GICSCustom-NA4', datetime.date(2018,9,29))] = {

    # Energy Equityment & Services
    '10101010': ('Energy Equipment & Services', 'N'), # Oil & Gas Drilling-B, Energy
    '10101020': ('Energy Equipment & Services', 'N'), # Oil & Gas Equipment & Services-B, Energy
    # Oil, Gas & Consumable Fuels
    '10102010': ('Oil, Gas & Consumable Fuels', 'N'), # Integrated Oil & Gas-B, Energy
    '10102020': ('Oil, Gas & Consumable Fuels', 'N'), # Oil & Gas Exploration & Production-B, Energy
    '10102030': ('Oil, Gas & Consumable Fuels', 'N'), #Oil & Gas Refining & Marketing-B, Energy
    '10102040': ('Oil, Gas & Consumable Fuels', 'N'), # Oil & Gas Storage & Transportation-B, Energy
    '10102050': ('Oil, Gas & Consumable Fuels', 'N'), # Coal & Consumable Fuels-B, Energy
    # Chemicals
    '15101010': ('Chemicals', 'N'),
    '15101020': ('Chemicals', 'N'),
    '15101030': ('Chemicals', 'N'),
    '15101040': ('Chemicals', 'N'),
    '15101050': ('Chemicals', 'N'),
    # Construction Materials
    '15102010': ('Construction Materials', 'N'),
    # Containers & Packaging
    '15103010': ('Containers & Packaging', 'N'),
    '15103020': ('Containers & Packaging', 'N'),
    # Metals & Mining
    '15104010': ('Metals & Mining ex Gold', 'Y'),
    '15104020': ('Metals & Mining ex Gold', 'Y'),
    '15104025': ('Metals & Mining ex Gold', 'Y'),
    '15104030': ('Gold', 'N'),
    '15104040': ('Metals & Mining ex Gold', 'Y'),
    '15104045': ('Metals & Mining ex Gold', 'Y'),
    '15104050': ('Metals & Mining ex Gold', 'Y'),
    # Paper & Forest Products
    '15105010': ('Paper & Forest Products', 'N'),
    '15105020': ('Paper & Forest Products', 'N'),

    # Capital Goods
    '20101010': ('Aerospace & Defense', 'N'),
    '20102010': ('Building Products', 'N'),
    '20103010': ('Construction & Engineering', 'N'),
    '20104010': ('Electrical Equipment', 'N'),
    '20104020': ('Electrical Equipment', 'N'),
    '20105010': ('Industrial Conglomerates', 'N'),

    '20106010': ('Machinery', 'N'),
    '20106015': ('Machinery', 'N'),
    '20106020': ('Machinery', 'N'),

    '20107010': ('Trading Companies & Distributors', 'N'),
    
    # Commercial Services & Supplies
    '20201010': ('Commercial Services & Supplies', 'N'),
    '20201050': ('Commercial Services & Supplies', 'N'),
    '20201060': ('Commercial Services & Supplies', 'N'),
    '20201070': ('Commercial Services & Supplies', 'N'),
    '20201080': ('Commercial Services & Supplies', 'N'),

    '20202010': ('Professional Services', 'N'),
    '20202020': ('Professional Services', 'N'),
    '20201040': ('Professional Services', 'N'),
    # Transportation
    '20301010': ('Air Freight & Logistics', 'N'),
    '20302010': ('Airlines', 'N'),
    '20303010': ('Marine', 'N'),
    '20304010': ('Road & Rail', 'N'),
    '20304020': ('Road & Rail', 'N'),

    '20305010': ('Transportation Infrastructure', 'N'),
    '20305020': ('Transportation Infrastructure', 'N'),
    '20305030': ('Transportation Infrastructure', 'N'),


    # Auto Components
    '25101010': ('Auto Components', 'N'),
    '25101020': ('Auto Components', 'N'),

    '25102010': ('Automobiles', 'N'),
    '25102020': ('Automobiles', 'N'),
    # Consumer Durables & Apparel
    '25201010': ('Household Durables', 'N'),
    '25201020': ('Household Durables', 'N'),
    '25201030': ('Household Durables', 'N'),
    '25201040': ('Household Durables', 'N'),
    '25201050': ('Household Durables', 'N'),
    # 
    '25202010': ('Leisure Products', 'N'),
    '25202020': ('Leisure Products', 'N'),
    

    # Textiles, Apparel & Luxury Goods
    '25203010': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25203020': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25203030': ('Textiles, Apparel & Luxury Goods', 'N'),
    # Consumer Services
    '25301010': ('Hotels, Restaurants & Leisure', 'N'),
    '25301020': ('Hotels, Restaurants & Leisure', 'N'),
    '25301030': ('Hotels, Restaurants & Leisure', 'N'),
    '25301040': ('Hotels, Restaurants & Leisure', 'N'),
    # Diversified Consumer Services
    '25302010': ('Diversified Consumer Services', 'N'),
    '25302020': ('Diversified Consumer Services', 'N'),
    # Distributors
    '25501010': ('Distributors', 'N'),

    #Internet & Direct Marketing Retail
    '25502010': ('Internet & Direct Marketing Retail', 'N'),
    '25502020': ('Internet & Direct Marketing Retail', 'N'),
    '45101010': ('Internet & Direct Marketing Retail', 'N'), # 2018 Addition
    # Multiline Retail
    '25503010': ('Multiline Retail', 'N'),
    '25503020': ('Multiline Retail', 'N'),
    # Specialty Retail
    '25504010': ('Specialty Retail', 'N'),
    '25504020': ('Specialty Retail', 'N'),
    '25504030': ('Specialty Retail', 'N'),
    '25504040': ('Specialty Retail', 'N'),
    '25504050': ('Specialty Retail', 'N'),
    '25504060': ('Specialty Retail', 'N'),
    # Food & Staples Retailing
    '30101010': ('Food & Staples Retailing', 'N'),
    '30101020': ('Food & Staples Retailing', 'N'),
    '30101030': ('Food & Staples Retailing', 'N'),
    '30101040': ('Food & Staples Retailing', 'N'),
    # Food, Beverage & Tobacco
    '30201010': ('Beverages', 'N'),
    '30201020': ('Beverages', 'N'),
    '30201030': ('Beverages', 'N'),

    '30202010': ('Food Products', 'N'),
    '30202020': ('Food Products', 'N'),
    '30202030': ('Food Products', 'N'),

    '30203010': ('Tobacco', 'N'),

    '30301010': ('Household Products', 'N'), # Household Products-B, Household & Personal Products
    '30302010': ('Personal Products', 'N'),
    
    
    
    # Health Care Equipment & Services
    '35101010': ('Health Care Equipment & Supplies', 'N'),
    '35101020': ('Health Care Equipment & Supplies', 'N'),

    '35102010': ('Health Care Providers & Services', 'N'),
    '35102015': ('Health Care Providers & Services', 'N'),
    '35102020': ('Health Care Providers & Services', 'N'),
    '35102030': ('Health Care Providers & Services', 'N'),

    '35103010': ('Health Care Technology', 'N'),
    # Pharmaceuticals, Biotechnology & Life Sciences
    '35201010': ('Biotechnology', 'N'),

    '35202010': ('Pharmaceuticals', 'N'),

    '35203010': ('Life Sciences Tools & Services', 'N'),
    # Technology Hardware & Equipment
    '40101010': ('Banks', 'N'),
    '40101015': ('Banks', 'N'),

    '40102010': ('Thrifts & Mortgage Finance', 'N'),

    '40201010': ('Diversified Financial Services', 'N'),
    '40201020': ('Diversified Financial Services', 'N'),
    '40201030': ('Diversified Financial Services', 'N'),
    '40201040': ('Diversified Financial Services', 'N'),

    '40202010': ('Consumer Finance', 'N'),

    '40203010': ('Capital Markets', 'N'),
    '40203020': ('Capital Markets', 'N'),
    '40203030': ('Capital Markets', 'N'),
    '40203040': ('Capital Markets', 'N'),

    '40204010': ('Mortgage Real Estate Investment Trusts (REITs)', 'N'),
    '40402030': ('Mortgage Real Estate Investment Trusts (REITs)', 'N'),



    '40301010': ('Insurance', 'N'),
    '40301020': ('Insurance', 'N'),
    '40301030': ('Insurance', 'N'),
    '40301040': ('Insurance', 'N'),
    '40301050': ('Insurance', 'N'),

    '45102010': ('IT Services', 'N'),
    '45102020': ('IT Services', 'N'),
    '45102030': ('IT Services', 'N'),


    '45103010': ('Software', 'N'),
    '45103020': ('Software', 'N'),
    '45103030': ('Software', 'N'),

    '45201010': ('Communications Equipment', 'N'),
    '45201020': ('Communications Equipment', 'N'),

    
    '45203010': ('Electronic Equipment, Instruments & Components', 'N'),
    '45203015': ('Electronic Equipment, Instruments & Components', 'N'),
    '45203020': ('Electronic Equipment, Instruments & Components', 'N'),
    '45203030': ('Electronic Equipment, Instruments & Components', 'N'),


    # # Semiconductors
    '45205010': ('Semiconductors & Semiconductor Equipment', 'N'),
    '45205020': ('Semiconductors & Semiconductor Equipment', 'N'),
    
    '45202010': ('Technology Hardware, Storage & Peripherals', 'N'),
    '45202020': ('Technology Hardware, Storage & Peripherals', 'N'),
    '45202030': ('Technology Hardware, Storage & Peripherals', 'N'),
    '45204010': ('Technology Hardware, Storage & Peripherals', 'N'),



    # '45301010': ('Technology Hardware', 'Y'),
    '50101010': ('Diversified Telecommunication Services', 'N'),
    '50101020': ('Diversified Telecommunication Services', 'N'),
    '50102010': ('Wireless Telecommunication Services', 'N'),
    # Media & Entertainment
    '50201010': ('Media', 'N'),
    '50201020': ('Media', 'N'),
    '50201030': ('Media', 'N'),
    '50201040': ('Media', 'N'),

    '50202010': ('Entertainment', 'N'),
    '50202020': ('Entertainment', 'N'),

    '50203010': ('Interactive Media & Services', 'N'),


    '55101010': ('Electric Utilities', 'N'),
    '55102010': ('Gas Utilities', 'N'),
    '55103010': ('Multi-Utilities', 'N'),
    '55104010': ('Water Utilities', 'N'),

    '55105010': ('Independent Power and Renewable Electricity Producers', 'N'),
    '55105020': ('Independent Power and Renewable Electricity Producers', 'N'),
    
    # # Equity Real Estate Investment Trusts (REITs)
    '40401010': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Real Estate Investment Trusts-B, Real Estate
    '40402010': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Diversified REITs-B, Real Estate
    '40402020': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Industrial REITs-B, Real Estate
    #'40402030': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Mortgage REITs-B, Real Estate
    '40402040': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Office REITs-B, Real Estate
    '40402050': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Residential REITs-B, Real Estate
    '40402060': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Retail REITs-B, Real Estate
    '40402070': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Specialized REITs-B, Real Estate
    '40402035': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Hotel & Resort REITs-B, Real Estate
    '40402045': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Health Care REITs-B, Real Estate
    '60101010': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Diversified REITs-B, Real Estate
    '60101020': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Industrial REITs-B, Real Estate
    '60101030': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Hotel & Resort REITs-B, Real Estate
    '60101040': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Office REITs-B, Real Estate
    '60101050': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Health Care REITs-B, Real Estate
    '60101060': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Residential REITs-B, Real Estate
    '60101070': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Retail REITs-B, Real Estate
    '60101080': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Specialized REITs-B, Real Estate

    # # Real Estate Management & Development
    '40401020': ('Real Estate Management & Development', 'N'), # Real Estate Management & Development-B, Real Estate
    '40403010': ('Real Estate Management & Development', 'N'), # Real Estate Management & Development-B, Real Estate
    '40403020': ('Real Estate Management & Development', 'N'), # Real Estate Operating Companies-B, Real Estate
    '40403030': ('Real Estate Management & Development', 'N'), # Real Estate Development-B, Real Estate
    '40403040': ('Real Estate Management & Development', 'N'), # Real Estate Services-B, Real Estate
    '60102010': ('Real Estate Management & Development', 'N'), # Diversified Real Estate Activities-B, Real Estate
    '60102020': ('Real Estate Management & Development', 'N'), # Real Estate Operating Companies-B, Real Estate
    '60102030': ('Real Estate Management & Development', 'N'), # Real Estate Development-B, Real Estate
    '60102040': ('Real Estate Management & Development', 'N'), # Real Estate Services-B, Real Estate
    }

guessMaps[('GICSCustom-JP', datetime.date(2016,9,1))] = {
    '10101010': ('Energy', 'N'),
    '10101020': ('Energy', 'N'),
    '10102010': ('Energy', 'N'),
    '10102020': ('Energy', 'N'),
    '10102030': ('Energy', 'N'),
    '10102040': ('Energy', 'N'),
    '10102050': ('Energy', 'N'),
    '20101010': ('Machinery', 'Y'),
    '20201020': ('IT Services', 'N'),
    '20201040': ('Professional Services', 'N'),
    '20202010': ('Professional Services', 'N'),
    '20301010': ('Air Freight & Airlines', 'Y'),
    '20302010': ('Air Freight & Airlines', 'Y'),
    '25202010': ('Leisure Products', 'N'),
    '25202020': ('Leisure Products', 'N'),
    '25203010': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25203020': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25203030': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25502010': ('Internet & Direct Marketing Retail', 'N'),
    '25502020': ('Internet & Direct Marketing Retail', 'N'),
    '30101010': ('Food & Staples Retailing', 'N'),
    '30101020': ('Food & Staples Retailing', 'N'),
    '30101030': ('Food & Staples Retailing', 'N'),
    '30201010': ('Food Beverage & Tobacco', 'Y'),
    '30201020': ('Food Beverage & Tobacco', 'Y'),
    '30201030': ('Food Beverage & Tobacco', 'Y'),
    '30202010': ('Food Beverage & Tobacco', 'Y'),
    '30202020': ('Food Beverage & Tobacco', 'Y'),
    '30202030': ('Food Beverage & Tobacco', 'Y'),
    '30203010': ('Food Beverage & Tobacco', 'Y'),
    '30301010': ('Household & Personal Products', 'N'),
    '30302010': ('Household & Personal Products', 'N'),
    '35101010': ('Health Care Equipment & Technology', 'Y'),
    '35101020': ('Health Care Equipment & Technology', 'Y'),
    '35103010': ('Health Care Equipment & Technology', 'Y'),
    '35201010': ('Biotechnology', 'N'),
    '35203010': ('Life Sciences Tools & Services','N'),
    '40101010': ('Banks', 'N'),
    '40101015': ('Banks', 'N'),
    '40102010': ('Banks', 'N'),
    '40201010': ('Diversified Financial Services & Mortgage REITs', 'Y'),
    '40201020': ('Diversified Financial Services & Mortgage REITs', 'Y'),
    '40201030': ('Diversified Financial Services & Mortgage REITs', 'Y'),
    '40201040': ('Diversified Financial Services & Mortgage REITs', 'Y'),
    '40204010': ('Diversified Financial Services & Mortgage REITs', 'Y'),
    '40401010': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '40401020': ('Real Estate Management & Development', 'N'),
    '40402010': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '40402020': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '40402030': ('Diversified Financial Services & Mortgage REITs', 'Y'),
    '40402035': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '40402040': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '40402045': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '40402050': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '40402060': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '40402070': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '45102010': ('IT Services', 'N'),
    '45202010': ('Technology Hardware, Storage & Peripherals', 'N'),
    '45202020': ('Technology Hardware, Storage & Peripherals', 'N'),
    '45202030': ('Technology Hardware, Storage & Peripherals', 'N'),
    '45203010': ('Electronic Equipment, Instruments & Components', 'N'),
    '45203020': ('Electronic Equipment, Instruments & Components', 'N'),
    '45203030': ('Electronic Equipment, Instruments & Components', 'N'),
    '45204010': ('Technology Hardware, Storage & Peripherals', 'N'),
    '45205010': ('Semiconductors & Semiconductor Equipment', 'Y'),
    '45205020': ('Semiconductors & Semiconductor Equipment', 'Y'),
    '50101010': ('Telecommunication Services', 'N'),
    '50101020': ('Telecommunication Services', 'N'),
    '50102010': ('Telecommunication Services', 'N'),
    '50202010': ('Media', 'Y'),
    '50202020': ('Software', 'Y'),
    '50203010': ('Internet Software & Services', 'Y'),
    '55101010': ('Utilities', 'N'),
    '55102010': ('Utilities', 'N'),
    '55103010': ('Utilities', 'N'),
    '55104010': ('Utilities', 'N'),
    '55105010': ('Utilities', 'N'),
    '55105020': ('Utilities', 'N'),
    }

guessMaps[('GICSCustom-JP', datetime.date(2014,3,1))] = {
    '10101010': ('Energy', 'N'),
    '10101020': ('Energy', 'N'),
    '10102010': ('Energy', 'N'),
    '10102020': ('Energy', 'N'),
    '10102030': ('Energy', 'N'),
    '10102040': ('Energy', 'N'),
    '10102050': ('Energy', 'N'),
    '20101010': ('Machinery', 'Y'),
    '20201020': ('IT Services', 'Y'),
    '20201040': ('Professional Services', 'Y'),
    '20202010': ('Professional Services', 'Y'),
    '20301010': ('Air Freight & Logistics', 'N'),
    '25202010': ('Leisure Equipment & Products', 'N'),
    '25203010': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25203020': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25203030': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25502020': ('Internet & Catalog Retail', 'Y'),
    '30101010': ('Food & Staples Retailing', 'N'),
    '30101020': ('Food & Staples Retailing', 'N'),
    '30101030': ('Food & Staples Retailing', 'N'),
    '30201010': ('Beverages & Tobacco', 'Y'),
    '30201020': ('Beverages & Tobacco', 'Y'),
    '30201030': ('Beverages & Tobacco', 'Y'),
    '30203010': ('Beverages & Tobacco', 'Y'),
    '30301010': ('Household & Personal Products', 'Y'),
    '30302010': ('Household & Personal Products', 'Y'),
    '35101010': ('Health Care Equipment & Technology', 'Y'),
    '35101020': ('Health Care Equipment & Technology', 'Y'),
    '35103010': ('Health Care Equipment & Technology', 'Y'),
    '35201010': ('Biotechnology & Life Sciences', 'Y'),
    '35203010': ('Biotechnology & Life Sciences', 'Y'),
    '40101010': ('Banks', 'N'),
    '40101015': ('Banks', 'N'),
    '40102010': ('Banks', 'N'),
    '40201010': ('Diversified Financial Services', 'N'),
    '40201020': ('Diversified Financial Services', 'N'),
    '40201030': ('Diversified Financial Services', 'N'),
    '40204010': ('Real Estate Investment Trusts (REITs)', 'Y'),
    '40401010': ('Real Estate Investment Trusts (REITs)', 'Y'),
    '40401020': ('Real Estate Management & Development', 'Y'),
    '45102010': ('IT Services', 'N'),
    '45202030': ('Computers & Peripherals', 'N'),
    '45203010': ('Electronic Equipment, Instruments & Components', 'N'),
    '45203020': ('Electronic Equipment, Instruments & Components', 'N'),
    '45203030': ('Electronic Equipment, Instruments & Components', 'N'),
    '45204010': ('Computers & Peripherals', 'N'),
    '45205010': ('Semiconductors & Semiconductor Equipment', 'Y'),
    '45205020': ('Semiconductors & Semiconductor Equipment', 'Y'),
    '50101010': ('Telecommunication Services', 'N'),
    '50101020': ('Telecommunication Services', 'N'),
    '50102010': ('Telecommunication Services', 'N'),
    '50202010': ('Media', 'Y'),
    '50202020': ('Software', 'Y'),
    '50203010': ('Internet Software & Services', 'Y'),
    '55101010': ('Electric Utilities', 'N'),
    '55103010': ('Electric Utilities', 'N'),
    '55104010': ('Electric Utilities', 'N'),
    '55105010': ('Electric Utilities', 'N'),
    '55105020': ('Electric Utilities', 'N'),
    '60101010': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101020': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101030': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101040': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101050': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101060': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101070': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101080': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    }

guessMaps[('GICSCustom-JP', datetime.date(2008,8,30))] = {
    '10101010': ('Energy', 'N'),
    '10101020': ('Energy', 'N'),
    '10102010': ('Energy', 'N'),
    '10102020': ('Energy', 'N'),
    '10102030': ('Energy', 'N'),
    '10102040': ('Energy', 'N'),
    '10102050': ('Energy', 'N'),
    '20101010': ('Machinery', 'Y'),
    '20201020': ('IT Services', 'Y'),
    '20201040': ('Professional Services', 'Y'),
    '20202010': ('Professional Services', 'Y'),
    '20301010': ('Air Freight & Logistics', 'N'),
    '25202010': ('Leisure Equipment & Products', 'N'),
    '25203010': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25203020': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25203030': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25502020': ('Internet & Catalog Retail', 'N'),
    '30101010': ('Food & Staples Retailing', 'N'),
    '30101020': ('Food & Staples Retailing', 'N'),
    '30101030': ('Food & Staples Retailing', 'N'),
    '30201010': ('Beverages & Tobacco', 'Y'),
    '30201020': ('Beverages & Tobacco', 'Y'),
    '30201030': ('Beverages & Tobacco', 'Y'),
    '30203010': ('Beverages & Tobacco', 'Y'),
    '30301010': ('Household & Personal Products', 'Y'),
    '30302010': ('Household & Personal Products', 'Y'),
    '35101010': ('Health Care Equipment & Technology', 'Y'),
    '35101020': ('Health Care Equipment & Technology', 'Y'),
    '35103010': ('Health Care Equipment & Technology', 'Y'),
    '35201010': ('Biotechnology & Life Sciences', 'Y'),
    '35203010': ('Biotechnology & Life Sciences', 'Y'),
    '40101010': ('Banks', 'N'),
    '40101015': ('Banks', 'N'),
    '40102010': ('Banks', 'N'),
    '40201010': ('Diversified Financial Services', 'N'),
    '40201020': ('Diversified Financial Services', 'N'),
    '40201030': ('Diversified Financial Services', 'N'),
    '40204010': ('Real Estate Investment Trusts (REITs)', 'Y'),
    '40401010': ('Real Estate Investment Trusts (REITs)', 'Y'),
    '40401020': ('Real Estate Management & Development', 'Y'),
    '45102010': ('IT Services', 'N'),
    '45202030': ('Computers & Peripherals', 'N'),
    '45203010': ('Electronic Equipment, Instruments & Components', 'N'),
    '45203020': ('Electronic Equipment, Instruments & Components', 'N'),
    '45203030': ('Electronic Equipment, Instruments & Components', 'N'),
    '45205010': ('Semiconductors & Semiconductor Equipment', 'Y'),
    '45205020': ('Semiconductors & Semiconductor Equipment', 'Y'),
    '50101010': ('Telecommunication Services', 'N'),
    '50101020': ('Telecommunication Services', 'N'),
    '50102010': ('Telecommunication Services', 'N'),
    '50202010': ('Media', 'Y'),
    '50202020': ('Software', 'Y'),
    '50203010': ('Internet Software & Services', 'Y'),
    '55101010': ('Electric Utilities', 'N'),
    '55103010': ('Electric Utilities', 'N'),
    '55104010': ('Electric Utilities', 'N'),
    '55105010': ('Electric Utilities', 'N'),
    '55105020': ('Electric Utilities', 'N'),
    '60101010': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101020': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101030': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101040': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101050': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101060': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101070': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101080': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    }

guessMaps[('GICSCustom-JP', datetime.date(2006,4,29))] = {
    '10101010': ('Energy', 'N'),
    '10101020': ('Energy', 'N'),
    '10102010': ('Energy', 'N'),
    '10102020': ('Energy', 'N'),
    '10102030': ('Energy', 'N'),
    '10102040': ('Energy', 'N'),
    '10102050': ('Energy', 'N'),
    '20101010': ('Machinery', 'Y'),
    '20201020': ('IT Services', 'Y'),
    '20202010': ('Commercial Services & Supplies', 'N'),
    '20301010': ('Air Freight & Logistics', 'N'),
    '20202020': ('Commercial Services & Supplies', 'N'),
    '25101010': ('Automobiles & Components', 'N'),
    '25101020': ('Automobiles & Components', 'N'),
    '25102010': ('Automobiles & Components', 'N'),
    '25102020': ('Automobiles & Components', 'N'),
    '25202010': ('Leisure Equipment & Products', 'N'),
    '25203010': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25203020': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25203030': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25502020': ('Internet & Catalog Retail', 'Y'),
    '30101010': ('Food & Staples Retailing', 'N'),
    '30101020': ('Food & Staples Retailing', 'N'),
    '30101030': ('Food & Staples Retailing', 'N'),
    '30201010': ('Beverages & Tobacco', 'Y'),
    '30201020': ('Beverages & Tobacco', 'Y'),
    '30201030': ('Beverages & Tobacco', 'Y'),
    '30203010': ('Beverages & Tobacco', 'Y'),
    '30301010': ('Household & Personal Products', 'Y'),
    '30302010': ('Household & Personal Products', 'Y'),
    '35101010': ('Health Care Equipment & Technology', 'Y'),
    '35101020': ('Health Care Equipment & Technology', 'Y'),
    '35103010': ('Health Care Equipment & Technology', 'Y'),
    '35201010': ('Biotechnology & Life Sciences', 'Y'),
    '35203010': ('Biotechnology & Life Sciences', 'Y'),
    '40101010': ('Banks', 'N'),
    '40101015': ('Banks', 'N'),
    '40102010': ('Banks', 'N'),
    '40201010': ('Diversified Financial Services', 'N'),
    '40201020': ('Diversified Financial Services', 'N'),
    '40201030': ('Diversified Financial Services', 'N'),
    '40204010': ('Real Estate Investment Trusts (REITs)', 'Y'),
    '40401010': ('Real Estate Investment Trusts (REITs)', 'Y'),
    '40401020': ('Real Estate Management & Development', 'Y'),
    '45102010': ('IT Services', 'N'),
    '45202030': ('Computers & Peripherals', 'N'),
    '45203010': ('Electronic Equipment & Instruments', 'N'),
    '45203015': ('Electronic Equipment & Instruments', 'N'),
    '45203020': ('Electronic Equipment & Instruments', 'N'),
    '45203030': ('Electronic Equipment & Instruments', 'N'),
    '45205010': ('Semiconductors & Semiconductor Equipment', 'Y'),
    '45205020': ('Semiconductors & Semiconductor Equipment', 'Y'),
    '50101010': ('Telecommunication Services', 'N'),
    '50101020': ('Telecommunication Services', 'N'),
    '50102010': ('Telecommunication Services', 'N'),
    '50202010': ('Media', 'Y'),
    '50202020': ('Software', 'Y'),
    '50203010': ('Internet Software & Services', 'Y'),
    '55101010': ('Utilities', 'N'),
    '55102010': ('Utilities', 'N'),
    '55103010': ('Utilities', 'N'),
    '55104010': ('Utilities', 'N'),
    '55105010': ('Utilities', 'N'),
    '55105020': ('Utilities', 'N'),
    '60101010': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101020': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101030': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101040': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101050': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101060': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101070': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101080': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    }

guessMaps[('GICSCustom-GB', datetime.date(2008,8,30))] = {
    '10101010': ('Energy', 'N'),                        # Oil & Gas Drilling
    '10101020': ('Energy', 'N'),                        # Oil & Gas Equipment & Services
    '10102010': ('Energy', 'N'),                        # Integrated Oil & Gas
    '10102020': ('Energy', 'N'),                        # Oil & Gas Exploration & Production
    '10102030': ('Energy', 'N'),                        # Oil & Gas Refining & Marketing
    '10102040': ('Energy', 'N'),                        # Oil & Gas Storage & Transportation
    '10102050': ('Energy', 'N'),                        # Coal & Consumable Fuels
    '15102010': ('Metals & Mining', 'Y'),               # Construction Materials
    '15103010': ('Forestry, Containers & Packaging', 'Y'),    # Metal & Glass Containers
    '15103020': ('Forestry, Containers & Packaging', 'Y'),    # Paper Packaging
    '15105010': ('Forestry, Containers & Packaging' ,'Y'),    # Forest Products
    '15105020': ('Forestry, Containers & Packaging', 'Y'),    # Paper Products
    '20102010': ('Construction & Engineering', 'Y'),    # Building Products
    '20104010': ('Electrical Equipment & Machinery', 'N'),  # Electrical Components & Equipment
    '20104020': ('Electrical Equipment & Machinery', 'N'),  # Heavy Electrical Equipment
    '20106010': ('Electrical Equipment & Machinery', 'N'),  # Construction & Farm Machinery & Heavy Trucks
    '20106015': ('Electrical Equipment & Machinery', 'N'),  # Agricultural & Farm Machinery
    '20106020': ('Electrical Equipment & Machinery', 'N'),  # Industrial Machinery
    '20201040': ('Professional Services', 'Y'),
    '20301010': ('Transportation ex Airlines', 'Y'),    # Air Freight & Logistics
    '20303010': ('Transportation ex Airlines', 'Y'),    # Marine
    '20304010': ('Transportation ex Airlines', 'Y'),    # Railroads
    '20304020': ('Transportation ex Airlines', 'Y'),    # Trucking
    '20305010': ('Transportation ex Airlines', 'Y'),    # Airport Services
    '20305020': ('Transportation ex Airlines', 'Y'),    # Highways & Railroads
    '20305030': ('Transportation ex Airlines', 'Y'),    # Marine Ports & Services
    '25101010': ('Automobiles & Components', 'N'),      # Auto Parts & Equipment
    '25101020': ('Automobiles & Components', 'N'),      # Tires & Rubber
    '25102010': ('Automobiles & Components', 'N'),      # Automobile Manufacturers
    '25102020': ('Automobiles & Components', 'N'),      # Motorcycle Manufacturers
    '25202010': ('Textiles, Apparel & Luxury Goods', 'Y'),  # Leisure Products
    '25202020': ('Textiles, Apparel & Luxury Goods', 'Y'),  # Photographic Products
    '25203010': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25203020': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25203030': ('Textiles, Apparel & Luxury Goods', 'N'),
    '30101010': ('Food & Staples Retailing', 'N'),
    '30101020': ('Food & Staples Retailing', 'N'),
    '30101030': ('Food & Staples Retailing', 'N'),
    '30301010': ('Household & Personal Products', 'N'), # Household Products
    '30302010': ('Household & Personal Products', 'N'), # Personal Products
    '35101010': ('Health Care Equipment & Services', 'N'),  # Health Care Equipment
    '35101020': ('Health Care Equipment & Services', 'N'),  # Health Care Supplies
    '35102010': ('Health Care Equipment & Services', 'N'),  # Health Care Distributors
    '35102015': ('Health Care Equipment & Services', 'N'),  # Health Care Services
    '35102020': ('Health Care Equipment & Services', 'N'),  # Health Care Facilities
    '35102030': ('Health Care Equipment & Services', 'N'),  # Managed Health Care
    '35103010': ('Health Care Equipment & Services', 'N'),  # Health Care Technology
    '35201010': ('Biotechnology & Life Sciences', 'Y'),     # Biotechnology
    '35203010': ('Biotechnology & Life Sciences', 'Y'),     # Life Sciences Tools & Services
    '25501010': ('General Retail', 'Y'),                # Distributors
    '25502010': ('General Retail', 'Y'),                # Catalog Retail
    '25502020': ('General Retail', 'Y'),                # Internet Retail
    '25503010': ('General Retail', 'Y'),                # Department Stores
    '25503020': ('General Retail', 'Y'),                # General Merchandise Stores
    '25301010': ('Consumer Services', 'N'),             # Casinos & Gaming
    '25301020': ('Consumer Services', 'N'),             # Hotels, Resorts & Cruise Lines
    '25301030': ('Consumer Services', 'N'),             # Leisure Facilities
    '25301040': ('Consumer Services', 'N'),             # Restaurants
    '25302010': ('Consumer Services', 'N'),             # Education Services
    '25302020': ('Consumer Services', 'N'),             # Specialized Consumer Services
    '50101010': ('Telecommunication Services', 'N'),    # Alternative Carriers
    '50101020': ('Telecommunication Services', 'N'),    # Integrated Telecommunication Services
    '50102010': ('Telecommunication Services', 'N'),    # Wireless Telecommunication Services
    '55101010': ('Utilities', 'N'),                     # Electric Utilities
    '55102010': ('Utilities', 'N'),                     # Gas Utilities
    '55103010': ('Utilities', 'N'),                     # Multi-Utilities
    '55104010': ('Utilities', 'N'),                     # Water Utilities
    '55105010': ('Utilities', 'N'),                     # Independent Power Producers & Energy Traders
    '55105020': ('Utilities', 'N'),                     # Renewable Electricity
    '40101010': ('Banks', 'N'),                         # Diversified Banks
    '40101015': ('Banks', 'N'),                         # Regional Banks
    '40102010': ('Banks', 'N'),                         # Thrifts & Mortgage Finance
    '40201010': ('Diversified Financial Services', 'N'),
    '40201020': ('Diversified Financial Services', 'N'),
    '40201030': ('Diversified Financial Services', 'N'),
    '40202010': ('Diversified Financial Services', 'Y'),    # Consumer Finance
    '40204010': ('Real Estate Investment Trusts (REITs)', 'Y'),
    '45101010': ('Internet & IT Services', 'Y'),        # Internet Software & Services
    '45102010': ('Internet & IT Services', 'Y'),        # IT Consulting & Other Services
    '45102020': ('Internet & IT Services', 'Y'),        # Data Processing & Outsourced Services
    '45102030': ('Internet & IT Services', 'Y'),        # Internet Services & Infrastructure
    '45201010': ('Technology Hardware', 'Y'),           # Communications Equipment
    '45201020': ('Technology Hardware', 'Y'),           # Communications Equipment
    '45202010': ('Technology Hardware', 'Y'),           # Computer Hardware
    '45202020': ('Technology Hardware', 'Y'),           # Computer Storage & Peripherals
    '45202030': ('Technology Hardware', 'Y'),           # Technology Hardware, Storage & Peripherals
    '45204010': ('Technology Hardware', 'Y'),           # Office Electronics
    '20201020': ('Internet & IT Services', 'Y'),
    '40401010': ('Real Estate Investment Trusts (REITs)', 'Y'),
    '40401020': ('Real Estate Management & Development', 'Y'),
    '45205010': ('Semiconductors & Semiconductor Equipment', 'Y'),
    '45205020': ('Semiconductors & Semiconductor Equipment', 'Y'),
    '45203010': ('Electronic Equipment, Instruments & Components', 'N'),
    '45203020': ('Electronic Equipment, Instruments & Components', 'N'),
    '45203030': ('Electronic Equipment, Instruments & Components', 'N'),
    '50202010': ('Media', 'Y'),
    '50202020': ('Software', 'Y'),
    '50203010': ('Internet & IT Services', 'Y'),
    '60101010': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101020': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101030': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101040': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101050': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101060': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101070': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    '60101080': ('Real Estate Investment Trusts (REITs)', 'Y'), # CCHU: 2016-09-01 Addition
    }

guessMaps[('GICSCustom-GB4', datetime.date(2018,9,29))] = {
    '10101010': ('Energy', 'N'),                        # Oil & Gas Drilling
    '10101020': ('Energy', 'N'),                        # Oil & Gas Equipment & Services
    '10102010': ('Energy', 'N'),                        # Integrated Oil & Gas
    '10102020': ('Energy', 'N'),                        # Oil & Gas Exploration & Production
    '10102030': ('Energy', 'N'),                        # Oil & Gas Refining & Marketing
    '10102040': ('Energy', 'N'),                        # Oil & Gas Storage & Transportation
    '10102050': ('Energy', 'N'),                        # Coal & Consumable Fuels
    '15101010': ('Chemicals', 'N'),                     # Commodity Chemicals
    '15101020': ('Chemicals', 'N'),                     # Diversified Chemicals
    '15101030': ('Chemicals', 'N'),                     # Fertilizers & Agricultural Chemicals
    '15101040': ('Chemicals', 'N'),                     # Industrial Gases
    '15101050': ('Chemicals', 'N'),                     # Specialty Chemicals
    '15102010': ('Construction Materials', 'N'),    # Construction Materials
    '15103010': ('Other Materials and Packaging', 'Y'),    # Metal & Glass Containers
    '15103020': ('Other Materials and Packaging', 'Y'),    # Paper Packaging
    '15104010': ('Metals & Mining', 'N'),           # Aluminum
    '15104020': ('Metals & Mining', 'N'),           # Diversified Metals & Mining
    '15104025': ('Metals & Mining', 'N'),           # Copper
    '15104030': ('Metals & Mining', 'N'),           # Gold
    '15104040': ('Metals & Mining', 'N'),           # Precious Metals & Minerals
    '15104045': ('Metals & Mining', 'N'),           # Silver
    '15104050': ('Metals & Mining', 'N'),           # Steel
    '15105010': ('Other Materials and Packaging' ,'Y'),     # Forest Products
    '15105020': ('Other Materials and Packaging', 'Y'),     # Paper Products
    '20101010': ('Aerospace & Defense', 'N'),               # Aerospace & Defense
    '20102010': ('Building Products', 'N'),                 # Building Products
    '20103010': ('Construction & Engineering', 'N'),        # Construction & Engineering
    '20104010': ('Electrical Equipment', 'N'),              # Electrical Components & Equipment
    '20104020': ('Electrical Equipment', 'N'),              # Heavy Electrical Equipment
    '20105010': ('Other Industrials', 'Y'),                 # Industrial Conglomerates
    '20106010': ('Other Industrials', 'Y'),                 # Construction Machinery & Heavy Trucks
    '20106015': ('Other Industrials', 'Y'),                 # Agricultural & Farm Machinery
    '20106020': ('Other Industrials', 'Y'),                 # Industrial Machinery
    '20107010': ('Trading Companies & Distributors', 'N'),  # Trading Companies & Distributors
    '20201010': ('Commercial Services & Supplies', 'N'),    # Commercial Printing
    '20201050': ('Commercial Services & Supplies', 'N'),    # Environmental & Facilities Services
    '20201060': ('Commercial Services & Supplies', 'N'),    # Office Services & Supplies
    '20201070': ('Commercial Services & Supplies', 'N'),    # Diversified Support Services
    '20201080': ('Commercial Services & Supplies', 'N'),    # Security & Alarm Services
    '20202010': ('Professional Services', 'N'),         # Human Resource & Employment Services
    '20202020': ('Professional Services', 'N'),         # Research & Consulting Services
    '20301010': ('Transportation', 'N'),    # Air Freight & Logistics
    '20302010': ('Transportation', 'N'),
    '20303010': ('Transportation', 'N'),    # Marine
    '20304010': ('Transportation', 'N'),    # Railroads
    '20304020': ('Transportation', 'N'),    # Trucking
    '20305010': ('Transportation', 'N'),    # Airport Services
    '20305020': ('Transportation', 'N'),    # Highways & Railroads
    '20305030': ('Transportation', 'N'),    # Marine Ports & Services
    '25101010': ('Automobiles & Components', 'N'),      # Auto Parts & Equipment
    '25101020': ('Automobiles & Components', 'N'),      # Tires & Rubber
    '25102010': ('Automobiles & Components', 'N'),      # Automobile Manufacturers
    '25102020': ('Automobiles & Components', 'N'),      # Motorcycle Manufacturers
    '25201010': ('Consumer Durables & Apparel', 'N'),
    '25201020': ('Consumer Durables & Apparel', 'N'),
    '25201030': ('Consumer Durables & Apparel', 'N'),
    '25201040': ('Consumer Durables & Apparel', 'N'),
    '25201050': ('Consumer Durables & Apparel', 'N'),
    '25202010': ('Consumer Durables & Apparel', 'N'),  # Leisure Products
    '25202020': ('Consumer Durables & Apparel', 'N'),  # Photographic Products
    '25203010': ('Consumer Durables & Apparel', 'N'),
    '25203020': ('Consumer Durables & Apparel', 'N'),
    '25203030': ('Consumer Durables & Apparel', 'N'),
    '25301010': ('Consumer Services', 'N'),             # Casinos & Gaming
    '25301020': ('Consumer Services', 'N'),             # Hotels, Resorts & Cruise Lines
    '25301030': ('Consumer Services', 'N'),             # Leisure Facilities
    '25301040': ('Consumer Services', 'N'),             # Restaurants
    '25302010': ('Consumer Services', 'N'),             # Education Services
    '25302020': ('Consumer Services', 'N'),             # Specialized Consumer Services
    '25501010': ('Non-Internet Retail', 'Y'),            # Distributors
    '25502010': ('Internet & Direct Marketing Retail', 'N'),                # Catalog Retail
    '25502020': ('Internet & Direct Marketing Retail', 'N'),                # Internet Retail
    '25503010': ('Non-Internet Retail', 'Y'),           # Department Stores
    '25503020': ('Non-Internet Retail', 'Y'),           # General Merchandise Stores
    '25504010': ('Non-Internet Retail', 'Y'),           # Apparel Retail
    '25504020': ('Non-Internet Retail', 'Y'),           # Computer & Electronics Retail
    '25504030': ('Non-Internet Retail', 'Y'),           # Home Improvement Retail
    '25504040': ('Non-Internet Retail', 'Y'),           # Specialty Stores
    '25504050': ('Non-Internet Retail', 'Y'),           # Automotive Retail
    '25504060': ('Non-Internet Retail', 'Y'),           # Homefurnishing Retail
    '45101010': ('Internet & Direct Marketing Retail', 'N'), # 2018 Addition
    '30101010': ('Food & Staples Retailing', 'N'),      # Drug Retail
    '30101020': ('Food & Staples Retailing', 'N'),      # Food Distributors
    '30101030': ('Food & Staples Retailing', 'N'),      # Food Retail
    '30101040': ('Food & Staples Retailing', 'N'),
    '30201010': ('Beverages & Tobacco', 'Y'),    # Brewers
    '30201020': ('Beverages & Tobacco', 'Y'),    # Distillers & Vintners
    '30201030': ('Beverages & Tobacco', 'Y'),    # Soft drinks
    '30202010': ('Food Products', 'N'),         # Agricultural Products
    '30202020': ('Food Products', 'N'),         # Meat Poultry & Fish
    '30202030': ('Food Products', 'N'),         # Packaged Foods & Meats
    '30203010': ('Beverages & Tobacco', 'Y'),    # Tobacco
    '30301010': ('Household & Personal Products', 'N'), # Household Products
    '30302010': ('Household & Personal Products', 'N'), # Personal Products
    '35101010': ('Health Care Equipment & Services', 'N'),  # Health Care Equipment
    '35101020': ('Health Care Equipment & Services', 'N'),  # Health Care Supplies
    '35102010': ('Health Care Equipment & Services', 'N'),  # Health Care Distributors
    '35102015': ('Health Care Equipment & Services', 'N'),  # Health Care Services
    '35102020': ('Health Care Equipment & Services', 'N'),  # Health Care Facilities
    '35102030': ('Health Care Equipment & Services', 'N'),  # Managed Health Care
    '35103010': ('Health Care Equipment & Services', 'N'),  # Health Care Technology
    '35201010': ('Biotechnology & Life Sciences', 'Y'),     # Biotechnology
    '35202010': ('Pharmaceuticals', 'N'),                   # Pharmaceuticals
    '35203010': ('Biotechnology & Life Sciences', 'Y'),     # Life Sciences Tools & Services
    '40101010': ('Banks', 'N'),                         # Diversified Banks
    '40101015': ('Banks', 'N'),                         # Regional Banks
    '40102010': ('Banks', 'N'),                         # Thrifts & Mortgage Finance
    '40201010': ('Diversified Financial Services', 'N'),
    '40201020': ('Diversified Financial Services', 'N'),
    '40201030': ('Diversified Financial Services', 'N'),
    '40201040': ('Diversified Financial Services', 'N'),
    '40202010': ('Consumer Finance', 'N'),
    '40203010': ('Capital Markets', 'N'),
    '40203020': ('Capital Markets', 'N'),
    '40203030': ('Capital Markets', 'N'),
    '40203040': ('Capital Markets', 'N'),
    '40204010': ('Diversified Financial Services', 'N'),
    '40402030': ('Diversified Financial Services', 'N'),
    '45102010': ('IT Services', 'N'),        # IT Consulting & Other Services
    '45102020': ('IT Services', 'N'),
    '45102030': ('IT Services', 'N'),
    '45103010': ('Software', 'N'),
    '45103020': ('Software', 'N'),
    '45201010': ('Technology Hardware', 'Y'),           # Communications Equipment
    '45201020': ('Technology Hardware', 'Y'),           # Communications Equipment
    '45202010': ('Technology Hardware', 'Y'),           # Computer Hardware
    '45202020': ('Technology Hardware', 'Y'),           # Computer Storage & Peripherals
    '45202030': ('Technology Hardware', 'Y'),           # Technology Hardware, Storage & Peripherals
    '45203010': ('Technology Hardware', 'Y'),
    '45203015': ('Technology Hardware', 'Y'),
    '45203020': ('Technology Hardware', 'Y'),
    '45203030': ('Technology Hardware', 'Y'),
    '45204010': ('Technology Hardware', 'Y'),           # Office Electronics
    '45205010': ('Technology Hardware', 'Y'),
    '45205020': ('Technology Hardware', 'Y'),
    '45301010': ('Technology Hardware', 'Y'),
    '45301020': ('Technology Hardware', 'Y'),
    '50101010': ('Telecommunication Services', 'N'),    # Alternative Carriers
    '50101020': ('Telecommunication Services', 'N'),    # Integrated Telecommunication Services
    '50102010': ('Telecommunication Services', 'N'),    # Wireless Telecommunication Services
    '25401010': ('Media & Entertainment','N'),
    '25401020': ('Media & Entertainment','N'),
    '25401025': ('Media & Entertainment','N'),
    '25401030': ('Media & Entertainment','N'),
    '25401040': ('Media & Entertainment','N'),
    '45103030': ('Media & Entertainment','N'),
    '50201010': ('Media & Entertainment','N'),
    '50201020': ('Media & Entertainment','N'),
    '50201030': ('Media & Entertainment','N'),
    '50201040': ('Media & Entertainment','N'),
    '50202010': ('Media & Entertainment','N'),
    '50202020': ('Media & Entertainment','N'),
    '50203010': ('Media & Entertainment','N'),
    '55101010': ('Utilities', 'Y'),                     # Electric Utilities
    '55102010': ('Utilities', 'Y'),                     # Gas Utilities
    '55103010': ('Utilities', 'Y'),                     # Multi-Utilities
    '55104010': ('Utilities', 'Y'),                     # Water Utilities
    '55105010': ('Independent Power and Renewable Electricity Producers', 'N'),   # Independent Power Producers & Energy Traders
    '55105020': ('Independent Power and Renewable Electricity Producers', 'N'),   # Renewable Electricity
    '40401010': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '40402010': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Diversified REITs
    '40402020': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Industrial REITs
    '40402035': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Hotel & Resort REITs
    '40402040': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Office REITs
    '40402045': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Health Care REITs
    '40402050': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Residential REITs
    '40402060': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Retail REITs
    '40402070': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Specialized REITs
    '40401020': ('Real Estate Management & Development', 'N'),
    '40403010': ('Real Estate Management & Development', 'N'),
    '40403020': ('Real Estate Management & Development', 'N'),
    '40403030': ('Real Estate Management & Development', 'N'),
    '40403040': ('Real Estate Management & Development', 'N'),
    '60102010': ('Real Estate Management & Development', 'N'),
    '60102020': ('Real Estate Management & Development', 'N'),
    '60102030': ('Real Estate Management & Development', 'N'),
    '60102040': ('Real Estate Management & Development', 'N'),
    '60101010': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '60101020': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '60101030': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '60101040': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '60101050': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '60101060': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '60101070': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '60101080': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    }

guessMaps[('GICSCustom-AU', datetime.date(2008,8,30))] = {
    # Chemicals
    '15101010': ('Materials ex Metals & Mining', 'Y'),
    '15101020': ('Materials ex Metals & Mining', 'Y'),
    '15101030': ('Materials ex Metals & Mining', 'Y'),
    '15101040': ('Materials ex Metals & Mining', 'Y'),
    '15101050': ('Materials ex Metals & Mining', 'Y'),
    # Construction Materials
    '15102010': ('Materials ex Metals & Mining', 'Y'),
    # Containers & Packaging
    '15103010': ('Materials ex Metals & Mining', 'Y'),
    '15103020': ('Materials ex Metals & Mining', 'Y'),
    # Metals & Mining
    '15104010': ('Metals & Mining ex Gold & Steel', 'Y'),
    '15104020': ('Metals & Mining ex Gold & Steel', 'Y'),
    '15104025': ('Metals & Mining ex Gold & Steel', 'Y'),
    '15104030': ('Gold', 'N'),
    '15104040': ('Metals & Mining ex Gold & Steel', 'Y'),
    '15104045': ('Metals & Mining ex Gold & Steel', 'Y'),
    '15104050': ('Steel', 'N'),
    # Paper & Forest Products
    '15105010': ('Materials ex Metals & Mining', 'Y'),
    '15105020': ('Materials ex Metals & Mining', 'Y'),
    # Commercial Services & Supplies
    '20201010': ('Commercial & Professional Services', 'N'),
    '20201020': ('Information Technology', 'Y'),
    '20201030': ('Commercial & Professional Services', 'N'),
    '20201040': ('Commercial & Professional Services', 'N'),
    '20201050': ('Commercial & Professional Services', 'N'),
    '20201060': ('Commercial & Professional Services', 'N'),
    # Automobiles & Components
    '25101010': ('Consumer Discretionary ex Media', 'Y'),
    '25101020': ('Consumer Discretionary ex Media', 'Y'),
    '25102010': ('Consumer Discretionary ex Media', 'Y'),
    '25102020': ('Consumer Discretionary ex Media', 'Y'),
    # Consumer Durables & Apparel
    '25201010': ('Consumer Discretionary ex Media', 'Y'),
    '25201020': ('Consumer Discretionary ex Media', 'Y'),
    '25201030': ('Consumer Discretionary ex Media', 'Y'),
    '25201040': ('Consumer Discretionary ex Media', 'Y'),
    '25201050': ('Consumer Discretionary ex Media', 'Y'),
    '25202010': ('Consumer Discretionary ex Media', 'Y'),
    '25202020': ('Consumer Discretionary ex Media', 'Y'),
    '25203010': ('Consumer Discretionary ex Media', 'Y'),
    '25203020': ('Consumer Discretionary ex Media', 'Y'),
    '25203030': ('Consumer Discretionary ex Media', 'Y'),
    # Consumer Services
    '25301010': ('Consumer Discretionary ex Media', 'Y'),
    '25301020': ('Consumer Discretionary ex Media', 'Y'),
    '25301030': ('Consumer Discretionary ex Media', 'Y'),
    '25301040': ('Consumer Discretionary ex Media', 'Y'),
    '25302010': ('Consumer Discretionary ex Media', 'Y'),
    '25302020': ('Consumer Discretionary ex Media', 'Y'),
    # Retailing
    '25501010': ('Consumer Discretionary ex Media', 'Y'),
    '25502010': ('Consumer Discretionary ex Media', 'Y'),
    '25502020': ('Consumer Discretionary ex Media', 'Y'),
    '25503010': ('Consumer Discretionary ex Media', 'Y'),
    '25503020': ('Consumer Discretionary ex Media', 'Y'),
    '25504010': ('Consumer Discretionary ex Media', 'Y'),
    '25504020': ('Consumer Discretionary ex Media', 'Y'),
    '25504030': ('Consumer Discretionary ex Media', 'Y'),
    '25504040': ('Consumer Discretionary ex Media', 'Y'),
    '25504050': ('Consumer Discretionary ex Media', 'Y'),
    '25504060': ('Consumer Discretionary ex Media', 'Y'),
    # Food & Staples Retailing
    '30101010': ('Food & Staples Retailing', 'N'),
    '30101020': ('Food & Staples Retailing', 'N'),
    '30101030': ('Food & Staples Retailing', 'N'),
    # Food, Beverage & Tobacco
    '30201010': ('Food & Staples Products', 'Y'),
    '30201020': ('Food & Staples Products', 'Y'),
    '30201030': ('Food & Staples Products', 'Y'),
    '30202010': ('Food & Staples Products', 'Y'),
    '30202020': ('Food & Staples Products', 'Y'),
    '30202030': ('Food & Staples Products', 'Y'),
    '30203010': ('Food & Staples Products', 'Y'),
    # Household & Personal Products
    '30301010': ('Food & Staples Products', 'Y'),
    '30302010': ('Food & Staples Products', 'Y'),
    # Health Care Equipment & Services
    '35101010': ('Health Care', 'N'),
    '35101020': ('Health Care', 'N'),
    '35102010': ('Health Care', 'N'),
    '35102015': ('Health Care', 'N'),
    '35102020': ('Health Care', 'N'),
    '35102030': ('Health Care', 'N'),
    '35103010': ('Health Care', 'N'),
    # Pharmaceuticals, Biotechnology & Life Sciences
    '35201010': ('Health Care', 'N'),
    '35202010': ('Health Care', 'N'),
    '35203010': ('Health Care', 'N'),
    # GICS2016 - Mortgage REITs
    '40204010': ('Real Estate Investment Trusts (REITs)', 'N'),
    # Financials / Real Estate
    '40402010': ('Real Estate Investment Trusts (REITs)', 'N'),
    '40402020': ('Real Estate Investment Trusts (REITs)', 'N'),
    '40402030': ('Real Estate Investment Trusts (REITs)', 'N'),
    '40402035': ('Real Estate Investment Trusts (REITs)', 'N'),
    '40402040': ('Real Estate Investment Trusts (REITs)', 'N'),
    '40402045': ('Real Estate Investment Trusts (REITs)', 'N'),
    '40402050': ('Real Estate Investment Trusts (REITs)', 'N'),
    '40402060': ('Real Estate Investment Trusts (REITs)', 'N'),
    '40402070': ('Real Estate Investment Trusts (REITs)', 'N'),
    '40401010': ('Real Estate Investment Trusts (REITs)', 'N'),
    '40403010': ('Diversified Financials', 'Y'),
    '40403020': ('Diversified Financials', 'Y'),
    '40403030': ('Diversified Financials', 'Y'),
    '40403040': ('Diversified Financials', 'Y'),
    '40401020': ('Diversified Financials', 'Y'),
    # Technology Hardware & Equipment
    '45201010': ('Information Technology', 'Y'),
    '45201020': ('Information Technology', 'Y'),
    '45202010': ('Information Technology', 'Y'),
    '45202020': ('Information Technology', 'Y'),
    '45202030': ('Information Technology', 'Y'),
    '45203010': ('Information Technology', 'Y'),
    '45203015': ('Information Technology', 'Y'),
    '45203020': ('Information Technology', 'Y'),
    '45203030': ('Information Technology', 'Y'),
    '45204010': ('Information Technology', 'Y'),
    # Semiconductors
    '45205010': ('Information Technology', 'Y'),
    '45205020': ('Information Technology', 'Y'),
    '45301010': ('Information Technology', 'Y'),
    '45301020': ('Information Technology', 'Y'),
    # Software & Services
    '45101010': ('Information Technology', 'Y'),
    '45102010': ('Information Technology', 'Y'),
    '45102020': ('Information Technology', 'Y'),
    '45102030': ('Information Technology', 'Y'),
    '45103010': ('Information Technology', 'Y'),
    '45103020': ('Information Technology', 'Y'),
    '45103030': ('Information Technology', 'Y'),
    # Media & Entertainment
    '50201010': ('Media', 'Y'),
    '50201020': ('Media', 'Y'),
    '50201030': ('Media', 'Y'),
    '50201040': ('Media', 'Y'),
    '50202010': ('Media', 'Y'),
    '50202020': ('Information Technology', 'Y'),
    '50203010': ('Information Technology', 'Y'),
    '60102010': ('Diversified Financials', 'Y'),
    '60102020': ('Diversified Financials', 'Y'),
    '60102030': ('Diversified Financials', 'Y'),
    '60102040': ('Diversified Financials', 'Y'),
    '60101010': ('Real Estate Investment Trusts (REITs)', 'Y'),
    '60101020': ('Real Estate Investment Trusts (REITs)', 'Y'),
    '60101030': ('Real Estate Investment Trusts (REITs)', 'Y'),
    '60101040': ('Real Estate Investment Trusts (REITs)', 'Y'),
    '60101050': ('Real Estate Investment Trusts (REITs)', 'Y'),
    '60101060': ('Real Estate Investment Trusts (REITs)', 'Y'),
    '60101070': ('Real Estate Investment Trusts (REITs)', 'Y'),
    '60101080': ('Real Estate Investment Trusts (REITs)', 'Y'),
    }

guessMaps[('GICSCustom-AU2', datetime.date(2016,9,1))] = {
    # Chemicals
    '15101010': ('Materials ex Metals & Mining', 'Y'),
    '15101020': ('Materials ex Metals & Mining', 'Y'),
    '15101030': ('Materials ex Metals & Mining', 'Y'),
    '15101040': ('Materials ex Metals & Mining', 'Y'),
    '15101050': ('Materials ex Metals & Mining', 'Y'),
    # Construction Materials
    '15102010': ('Materials ex Metals & Mining', 'Y'),
    # Containers & Packaging
    '15103010': ('Materials ex Metals & Mining', 'Y'),
    '15103020': ('Materials ex Metals & Mining', 'Y'),
    # Metals & Mining
    '15104010': ('Metals & Mining ex Gold & Steel', 'Y'),
    '15104020': ('Metals & Mining ex Gold & Steel', 'Y'),
    '15104025': ('Metals & Mining ex Gold & Steel', 'Y'),
    '15104030': ('Gold', 'N'),
    '15104040': ('Metals & Mining ex Gold & Steel', 'Y'),
    '15104045': ('Metals & Mining ex Gold & Steel', 'Y'),
    '15104050': ('Steel', 'N'),
    # Paper & Forest Products
    '15105010': ('Materials ex Metals & Mining', 'Y'),
    '15105020': ('Materials ex Metals & Mining', 'Y'),
    # Commercial Services & Supplies
    '20201010': ('Commercial & Professional Services', 'N'),
    '20201020': ('Information Technology', 'Y'),
    '20201030': ('Commercial & Professional Services', 'N'),
    '20201040': ('Commercial & Professional Services', 'N'),
    '20201050': ('Commercial & Professional Services', 'N'),
    '20201060': ('Commercial & Professional Services', 'N'),
    # Automobiles & Components
    '25101010': ('Consumer Discretionary ex Media', 'Y'),
    '25101020': ('Consumer Discretionary ex Media', 'Y'),
    '25102010': ('Consumer Discretionary ex Media', 'Y'),
    '25102020': ('Consumer Discretionary ex Media', 'Y'),
    # Consumer Durables & Apparel
    '25201010': ('Consumer Discretionary ex Media', 'Y'),
    '25201020': ('Consumer Discretionary ex Media', 'Y'),
    '25201030': ('Consumer Discretionary ex Media', 'Y'),
    '25201040': ('Consumer Discretionary ex Media', 'Y'),
    '25201050': ('Consumer Discretionary ex Media', 'Y'),
    '25202010': ('Consumer Discretionary ex Media', 'Y'),
    '25202020': ('Consumer Discretionary ex Media', 'Y'),
    '25203010': ('Consumer Discretionary ex Media', 'Y'),
    '25203020': ('Consumer Discretionary ex Media', 'Y'),
    '25203030': ('Consumer Discretionary ex Media', 'Y'),
    # Consumer Services
    '25301010': ('Consumer Discretionary ex Media', 'Y'),
    '25301020': ('Consumer Discretionary ex Media', 'Y'),
    '25301030': ('Consumer Discretionary ex Media', 'Y'),
    '25301040': ('Consumer Discretionary ex Media', 'Y'),
    '25302010': ('Consumer Discretionary ex Media', 'Y'),
    '25302020': ('Consumer Discretionary ex Media', 'Y'),
    # Retailing
    '25501010': ('Consumer Discretionary ex Media', 'Y'),
    '25502010': ('Consumer Discretionary ex Media', 'Y'),
    '25502020': ('Consumer Discretionary ex Media', 'Y'),
    '25503010': ('Consumer Discretionary ex Media', 'Y'),
    '25503020': ('Consumer Discretionary ex Media', 'Y'),
    '25504010': ('Consumer Discretionary ex Media', 'Y'),
    '25504020': ('Consumer Discretionary ex Media', 'Y'),
    '25504030': ('Consumer Discretionary ex Media', 'Y'),
    '25504040': ('Consumer Discretionary ex Media', 'Y'),
    '25504050': ('Consumer Discretionary ex Media', 'Y'),
    '25504060': ('Consumer Discretionary ex Media', 'Y'),
    # Food & Staples Retailing
    '30101010': ('Consumer Staples', 'N'),
    '30101020': ('Consumer Staples', 'N'),
    '30101030': ('Consumer Staples', 'N'),
    '30101040': ('Consumer Staples', 'N'),
    # Food, Beverage & Tobacco
    '30201010': ('Consumer Staples', 'N'),
    '30201020': ('Consumer Staples', 'N'),
    '30201030': ('Consumer Staples', 'N'),
    '30202010': ('Consumer Staples', 'N'),
    '30202020': ('Consumer Staples', 'N'),
    '30202030': ('Consumer Staples', 'N'),
    '30203010': ('Consumer Staples', 'N'),
    # Household & Personal Products
    '30301010': ('Consumer Staples', 'N'),
    '30302010': ('Consumer Staples', 'N'),
    # Health Care Equipment & Services
    '35101010': ('Health Care', 'N'),
    '35101020': ('Health Care', 'N'),
    '35102010': ('Health Care', 'N'),
    '35102015': ('Health Care', 'N'),
    '35102020': ('Health Care', 'N'),
    '35102030': ('Health Care', 'N'),
    '35103010': ('Health Care', 'N'),
    # Pharmaceuticals, Biotechnology & Life Sciences
    '35201010': ('Health Care', 'N'),
    '35202010': ('Health Care', 'N'),
    '35203010': ('Health Care', 'N'),
    # GICS2016 - Mortgage REITs
    '40204010': ('Diversified Financials', 'N'),
    # Financials / Real Estate
    '40402010': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '40402020': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '40402030': ('Diversified Financials', 'N'),   # before GICS2014 - Mortgage REITs
    '40402035': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '40402040': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '40402045': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '40402050': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '40402060': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '40402070': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '40401010': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '40403010': ('Real Estate Management & Development', 'N'),
    '40403020': ('Real Estate Management & Development', 'N'),
    '40403030': ('Real Estate Management & Development', 'N'),
    '40403040': ('Real Estate Management & Development', 'N'),
    '40401020': ('Real Estate Management & Development', 'N'),
    # Technology Hardware & Equipment
    '45201010': ('Information Technology', 'Y'),
    '45201020': ('Information Technology', 'Y'),
    '45202010': ('Information Technology', 'Y'),
    '45202020': ('Information Technology', 'Y'),
    '45202030': ('Information Technology', 'Y'),
    '45203010': ('Information Technology', 'Y'),
    '45203015': ('Information Technology', 'Y'),
    '45203020': ('Information Technology', 'Y'),
    '45203030': ('Information Technology', 'Y'),
    '45204010': ('Information Technology', 'Y'),
    # Semiconductors
    '45205010': ('Information Technology', 'Y'),
    '45205020': ('Information Technology', 'Y'),
    '45301010': ('Information Technology', 'Y'),
    '45301020': ('Information Technology', 'Y'),
    # Software & Services
    '45101010': ('Information Technology', 'Y'),
    '45102010': ('Information Technology', 'Y'),
    '45102020': ('Information Technology', 'Y'),
    '45102030': ('Information Technology', 'Y'),
    '45103010': ('Information Technology', 'Y'),
    '45103020': ('Information Technology', 'Y'),
    '45103030': ('Information Technology', 'Y'),
    # Media & Entertainment
    '50201010': ('Media', 'Y'),
    '50201020': ('Media', 'Y'),
    '50201030': ('Media', 'Y'),
    '50201040': ('Media', 'Y'),
    '50202010': ('Media', 'Y'),
    '50202020': ('Information Technology', 'Y'),
    '50203010': ('Information Technology', 'Y'),
    '60102010': ('Real Estate Management & Development', 'N'),
    '60102020': ('Real Estate Management & Development', 'N'),
    '60102030': ('Real Estate Management & Development', 'N'),
    '60102040': ('Real Estate Management & Development', 'N'),
    '60101010': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '60101020': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '60101030': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '60101040': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '60101050': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '60101060': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '60101070': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '60101080': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    }

guessMaps[('GICSCustom-EU', datetime.date(2016,9,1))] = {
    '10102010': ('Oil, Gas & Consumable Fuels', 'N'),
    '10102020': ('Oil, Gas & Consumable Fuels', 'N'),
    '10102030': ('Oil, Gas & Consumable Fuels', 'N'),
    '20201010': ('Commercial & Professional Services', 'N'),
    '20201020': ('Commercial & Professional Services', 'N'),
    '20201030': ('Commercial & Professional Services', 'N'),
    '20201040': ('Commercial & Professional Services', 'N'),
    '20201050': ('Commercial & Professional Services', 'N'),
    '20201060': ('Commercial & Professional Services', 'N'),
    '20201070': ('Commercial & Professional Services', 'N'),
    '20201080': ('Commercial & Professional Services', 'N'),
    '20202010': ('Commercial & Professional Services', 'N'),
    '20202020': ('Commercial & Professional Services', 'N'),
    '20301010': ('Air Freight & Logistics', 'N'),
    '25202010': ('Leisure Products', 'N'),
    '25202020': ('Leisure Products', 'N'),
    '25203010': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25203020': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25203030': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25301010': ('Consumer Services', 'N'),
    '25301020': ('Consumer Services', 'N'),
    '25301030': ('Consumer Services', 'N'),
    '25301040': ('Consumer Services', 'N'),
    '25302010': ('Consumer Services', 'N'),
    '25302020': ('Consumer Services', 'N'),
    '25502010': ('Internet & Direct Marketing Retail', 'N'),
    '25502020': ('Internet & Direct Marketing Retail', 'N'),
    '25503010': ('Multiline & Specialty Retail', 'Y'),
    '25503020': ('Multiline & Specialty Retail', 'Y'),
    '25504010': ('Multiline & Specialty Retail', 'Y'),
    '25504020': ('Multiline & Specialty Retail', 'Y'),
    '25504030': ('Multiline & Specialty Retail', 'Y'),
    '25504040': ('Multiline & Specialty Retail', 'Y'),
    '25504050': ('Multiline & Specialty Retail', 'Y'),
    '25504060': ('Multiline & Specialty Retail', 'Y'),
    '30101010': ('Food & Staples Retailing', 'N'),
    '30101020': ('Food & Staples Retailing', 'N'),
    '30101030': ('Food & Staples Retailing', 'N'),
    '30201030': ('Beverages & Tobacco', 'Y'),
    '30201020': ('Beverages & Tobacco', 'Y'),
    '30201020': ('Beverages & Tobacco', 'Y'),
    '30201010': ('Beverages & Tobacco', 'Y'),
    '30203010': ('Beverages & Tobacco', 'Y'),
    '30301010': ('Household & Personal Products', 'Y'),
    '30302010': ('Household & Personal Products', 'Y'),
    '40101010': ('Banks', 'N'),
    '40101015': ('Banks', 'N'),
    '40102010': ('Banks', 'N'),
    '40203010': ('Capital Markets', 'N'),
    '40203020': ('Capital Markets', 'N'),
    '40203030': ('Capital Markets', 'N'),
    '40203040': ('Capital Markets', 'N'),
    '40201010': ('Diversified Financial Services', 'Y'),
    '40201020': ('Diversified Financial Services', 'Y'),
    '40201030': ('Diversified Financial Services', 'Y'),
    '40202010': ('Diversified Financial Services', 'Y'),
    '40204010': ('Diversified Financial Services', 'Y'),
    '40402030': ('Diversified Financial Services', 'Y'),
    '40401010': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '40401020': ('Real Estate Management & Development', 'N'),
    '40402010': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '40402020': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '40402035': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '40402040': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '40402045': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '40402050': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '40402060': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '40402070': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '45102010': ('IT Services', 'N'),
    '45202010': ('Technology Hardware, Storage & Peripherals', 'N'),
    '45202020': ('Technology Hardware, Storage & Peripherals', 'N'),
    '45204010': ('Technology Hardware, Storage & Peripherals', 'N'),
    '45203010': ('Electronic Equipment, Instruments & Components', 'N'),
    '45203020': ('Electronic Equipment, Instruments & Components', 'N'),
    '45203030': ('Electronic Equipment, Instruments & Components', 'N'),
    '45205010': ('Semiconductors & Semiconductor Equipment', 'Y'),
    '45205020': ('Semiconductors & Semiconductor Equipment', 'Y'),
    '50202010': ('Media', 'Y'),
    '50202020': ('Software', 'Y'),
    '50203010': ('Internet Software & Services', 'Y'),
    '55102010': ('Gas, Water & Multi-Utilities', 'Y'),
    '55103010': ('Gas, Water & Multi-Utilities', 'Y'),
    '55104010': ('Gas, Water & Multi-Utilities', 'Y'),
    '55105010': ('Independent Power and Renewable Electricity Producers', 'N'),
    }

guessMaps[('GICSCustom-CN', datetime.date(2008,8,30))] = {
    # Energy Equipment & Services
    '10101010': ('Energy ex Coal', 'Y'),
    '10101020': ('Energy ex Coal', 'Y'),
    # Oil & Gas
    '10102010': ('Energy ex Coal', 'Y'),
    '10102020': ('Energy ex Coal', 'Y'),
    '10102030': ('Energy ex Coal', 'Y'),
    '10102040': ('Energy ex Coal', 'Y'),
    '10102050': ('Coal & Consumable Fuels', 'N'),
    # Chemicals
    '15101010': ('Chemicals', 'N'),
    '15101020': ('Chemicals', 'N'),
    '15101030': ('Chemicals', 'N'),
    '15101040': ('Chemicals', 'N'),
    '15101050': ('Chemicals', 'N'),

    # Construction Materials
    '15102010': ('Construction Materials', 'N'),
    # Containers & Packaging
    '15103010': ('Chemicals', 'Y'),
    '15103020': ('Chemicals', 'Y'),
    # Paper & Forest
    '15105010': ('Paper & Forest Products', 'N'),
    '15105020': ('Paper & Forest Products', 'N'),
    # Metals & Mining
    '15104010': ('Metals & Mining ex Steel', 'Y'),
    '15104020': ('Metals & Mining ex Steel', 'Y'),
    '15104025': ('Metals & Mining ex Steel', 'Y'),
    '15104030': ('Metals & Mining ex Steel', 'Y'),
    '15104040': ('Metals & Mining ex Steel', 'Y'),
    '15104045': ('Metals & Mining ex Steel','N'),
    '15104050': ('Steel', 'N'),

    # Construction & Engineering
    '20103010': ('Construction & Engineering', 'N'),
    # Building Products
    '20102010': ('Electrical Equipment', 'Y'),
    # Electrical Equipment
    '20104010': ('Electrical Equipment', 'Y'),
    '20104020': ('Electrical Equipment', 'Y'),
    # Aerospace
    '20101010': ('Machinery', 'Y'),
    # Machinery
    '20106010': ('Machinery', 'Y'),
    '20106015': ('Machinery', 'Y'),
    '20106020': ('Machinery', 'Y'),
    # Industrial Conglomberates, Trading Companies
    '20105010': ('Trading Companies, Distributors & Conglomerates', 'Y'),
    '20107010': ('Trading Companies, Distributors & Conglomerates', 'Y'),

    # Commercial Services & Supplies
    '20201010': ('Commercial & Professional Services', 'N'),
    '20201030': ('Commercial & Professional Services', 'N'),
    '20201040': ('Commercial & Professional Services', 'N'),
    '20201050': ('Commercial & Professional Services', 'N'),
    '20201060': ('Commercial & Professional Services', 'N'),
    '20201070': ('Commercial & Professional Services', 'N'),
    '20201080': ('Commercial & Professional Services', 'N'),

    # Air Freight, Airlines, Marine
    '20301010': ('Transportation Non-Infrastructure', 'Y'),
    '20302010': ('Transportation Non-Infrastructure', 'Y'),
    '20303010': ('Transportation Non-Infrastructure', 'Y'),
    # Road & Rail
    '20304010': ('Transportation Infrastructure', 'Y'),
    '20304020': ('Transportation Infrastructure', 'Y'),
    # Transportation Infrastructure
    '20305010': ('Transportation Infrastructure', 'Y'),
    '20305020': ('Transportation Infrastructure', 'Y'),
    '20305030': ('Transportation Infrastructure', 'Y'),

    # Auto Components
    '25101010': ('Auto Components', 'N'),
    '25101020': ('Auto Components', 'N'),
    # Automobiles
    '25102010': ('Automobiles', 'N'),
    '25102020': ('Automobiles', 'N'),

    # Household Durables
    '25201010': ('Household Durables', 'N'),
    '25201020': ('Household Durables', 'N'),
    '25201030': ('Household Durables', 'N'),
    '25201040': ('Household Durables', 'N'),
    '25201050': ('Household Durables', 'N'),

    # Leisure Equipmet& Products
    '25202010': ('Textiles, Apparel & Luxury Goods', 'Y'),
    '25202020': ('Textiles, Apparel & Luxury Goods', 'Y'),
    # Textiles, Apparel & Luxury Goods
    '25203010': ('Textiles, Apparel & Luxury Goods', 'Y'),
    '25203020': ('Textiles, Apparel & Luxury Goods', 'Y'),
    '25203030': ('Textiles, Apparel & Luxury Goods', 'Y'),

    # Hotels, Restaurants & Leisure
    '25301010': ('Consumer Services', 'N'),
    '25301020': ('Consumer Services', 'N'),
    '25301030': ('Consumer Services', 'N'),
    '25301040': ('Consumer Services', 'N'),
    # Diversified Consumer Services
    '25302010': ('Consumer Services', 'N'),
    '25302020': ('Consumer Services', 'N'),

    # Food & Staples Retailing
    '30101010': ('Beverages & Tobacco', 'Y'),
    '30101020': ('Beverages & Tobacco', 'Y'),
    '30101030': ('Beverages & Tobacco', 'Y'),
    '30101040': ('Beverages & Tobacco', 'Y'),

    # Beverages
    '30201010': ('Beverages & Tobacco', 'Y'),
    '30201020': ('Beverages & Tobacco', 'Y'),
    '30201030': ('Beverages & Tobacco', 'Y'),
    # Tobacco
    '30203010': ('Beverages & Tobacco', 'Y'),
    # Food Products
    '30202010': ('Food Products', 'N'),
    '30202020': ('Food Products', 'N'),
    '30202030': ('Food Products', 'N'),

    # Household & Personal Products
    '30301010': ('Beverages & Tobacco', 'Y'),
    '30302010': ('Beverages & Tobacco', 'Y'),

    # Health Care Equipment & Services
    '35101010': ('Health Care', 'N'),
    '35101020': ('Health Care', 'N'),
    '35102010': ('Health Care', 'N'),
    '35102015': ('Health Care', 'N'),
    '35102020': ('Health Care', 'N'),
    '35102030': ('Health Care', 'N'),
    '35103010': ('Health Care', 'N'),
    # Pharmaceuticals, Biotechnology & Life Sciences
    '35201010': ('Health Care', 'N'),
    '35202010': ('Health Care', 'N'),
    '35203010': ('Health Care', 'N'),

    '40101010': ('Financials', 'N'),
    '40101015': ('Financials', 'N'),
    '40102010': ('Financials', 'N'),
    '40201010': ('Financials', 'N'),
    '40201020': ('Financials', 'N'),
    '40201030': ('Financials', 'N'),
    '40201040': ('Financials', 'N'),
    '40202010': ('Financials', 'N'),
    '40203010': ('Financials', 'N'),
    '40203020': ('Financials', 'N'),
    '40203030': ('Financials', 'N'),
    '40203040': ('Financials', 'N'),
    '40204010': ('Real Estate', 'Y'),
    '40301010': ('Financials', 'N'),
    '40301020': ('Financials', 'N'),
    '40301030': ('Financials', 'N'),
    '40301040': ('Financials', 'N'),
    '40301050': ('Financials', 'N'),
    
    # Communications Equipment
    '45201010': ('Communications Equipment', 'N'),
    '45201020': ('Communications Equipment', 'N'),
    # Computers & Peripherals
    '45202010': ('Computers & Peripherals', 'N'),
    '45202020': ('Computers & Peripherals', 'N'),
    '45202030': ('Computers & Peripherals', 'N'),
    # Office Electronics
    '45204010': ('Semiconductors & Electronics', 'Y'),
    # Semiconductors
    '45301010': ('Semiconductors & Electronics', 'Y'),
    '45301020': ('Semiconductors & Electronics', 'Y'),
    '45205010': ('Semiconductors & Electronics', 'Y'),
    '45205020': ('Semiconductors & Electronics', 'Y'),
    # Electronic Equipment
    '45203010': ('Semiconductors & Electronics', 'Y'),
    '45203015': ('Semiconductors & Electronics', 'Y'),
    '45203020': ('Semiconductors & Electronics', 'Y'),
    '45203030': ('Semiconductors & Electronics', 'Y'),

    '20201020': ('Software & Services', 'Y'),
    '25301010': ('Consumer Services', 'N'),
    '25301020': ('Consumer Services', 'N'),
    '25301030': ('Consumer Services', 'N'),
    '25301040': ('Consumer Services', 'N'),
    '50201010': ('Media','Y'),               # CCHU 20180929
    '50201020': ('Media','Y'),               # CCHU 20180929
    '50201030': ('Media','Y'),               # CCHU 20180929
    '50201040': ('Media','Y'),               # CCHU 20180929
    '50202010': ('Media','Y'),               # CCHU 20180929
    '50202020': ('Software & Services','Y'), # CCHU 20180929
    '50203010': ('Software & Services','Y'), # CCHU 20180929
    }

guessMaps[('GICSCustom-CN2', datetime.date(2016,9,1))] = {
    # map from GICSsubIndustry to model industry factors:
    # Y: model Cust-Industry
    # N: model Industry using GICS-Industry: any level
    # model Industry using GICS-Industry mapping, (industry level) (21)
    '15101010':('Chemicals', 'N'),
    '15101020':('Chemicals', 'N'),
    '15101030':('Chemicals', 'N'),
    '15101040':('Chemicals', 'N'),
    '15101050':('Chemicals', 'N'),
    '15103010':('Chemicals', 'N'),
    '15103020':('Chemicals', 'N'),

    '15102010':('Construction Materials', 'N'),

    '15105010':('Paper & Forest Products', 'N'),
    '15105020':('Paper & Forest Products', 'N'),

    '20101010':('Aerospace & Defense', 'N'),

    '20103010':('Construction & Engineering', 'N'),

    '20106010':('Machinery', 'N'),
    '20106015':('Machinery', 'N'),
    '20106020':('Machinery', 'N'),

    '25101010':('Auto Components', 'N'),
    '25101020':('Auto Components', 'N'),

    '25102010':('Automobiles', 'N'),
    '25102020':('Automobiles', 'N'),

    '25201010':('Household Durables', 'N'),
    '25201020':('Household Durables', 'N'),
    '25201030':('Household Durables', 'N'),
    '25201040':('Household Durables', 'N'),
    '25201050':('Household Durables', 'N'),

    '25401010':('Media', 'N'),
    '25401020':('Media', 'N'),
    '25401025':('Media', 'N'),
    '25401030':('Media', 'N'),
    '25401040':('Media', 'N'),
    # map GICS2018-Sep to GICS2016Sep (need to discuss)
    '50201010':('Media', 'N'),
    '50201020':('Media', 'N'),
    '50201030':('Media', 'N'),
    '50201040':('Media', 'N'),
    # map GICS2018-Sep to GICS2016Sep (need to discuss) END

    '30101010':('Food & Staples Retailing', 'N'),
    '30101020':('Food & Staples Retailing', 'N'),
    '30101030':('Food & Staples Retailing', 'N'),
    '30101040':('Food & Staples Retailing', 'N'),

    '30202010':('Food Products', 'N'),
    '30202020':('Food Products', 'N'),
    '30202030':('Food Products', 'N'),

    '40101010':('Banks', 'N'),
    '40101015':('Banks', 'N'),
    '40102010':('Banks', 'N'),

    '45101010':('Internet Software & Services', 'N'),
    # map GICS2018-Sep to GICS2016Sep (need to discuss)
    '50202010':('Internet Software & Services', 'N'),
    '50202020':('Internet Software & Services', 'N'),
    '50203010':('Internet Software & Services', 'N'),
    # map GICS2018-Sep to GICS2016Sep (need to discuss) END
    '45102010':('IT Services', 'N'),
    '45102020':('IT Services', 'N'),
    '45102030':('IT Services', 'N'),

    '45103010':('Software', 'N'),
    '45103020':('Software', 'N'),
    '45103030':('Software', 'N'),

    '45201010':('Communications Equipment', 'N'),
    '45201020':('Communications Equipment', 'N'),

    '45202010':('Technology Hardware, Storage & Peripherals', 'N'),
    '45202020':('Technology Hardware, Storage & Peripherals', 'N'),
    '45202030':('Technology Hardware, Storage & Peripherals', 'N'),

    '45203010':('Electronic Equipment, Instruments & Components', 'N'),
    '45203015':('Electronic Equipment, Instruments & Components', 'N'),
    '45203020':('Electronic Equipment, Instruments & Components', 'N'),
    '45203030':('Electronic Equipment, Instruments & Components', 'N'),
    '45204010':('Electronic Equipment, Instruments & Components', 'N'),

    '45301010':('Semiconductors & Semiconductor Equipment', 'N'),
    '45301020':('Semiconductors & Semiconductor Equipment', 'N'),
    '45205010':('Semiconductors & Semiconductor Equipment', 'N'),
    '45205020':('Semiconductors & Semiconductor Equipment', 'N'),

    '55105010':('Independent Power and Renewable Electricity Producers', 'N'),
    '55105020':('Independent Power and Renewable Electricity Producers', 'N'),

    # model Industry using GICS-Industry mapping, (industry group level)(6)
    '20201010':('Commercial & Professional Services', 'N'),
    '20201020':('Commercial & Professional Services', 'N'),
    '20201030':('Commercial & Professional Services', 'N'),
    '20201040':('Commercial & Professional Services', 'N'),
    '20201050':('Commercial & Professional Services', 'N'),
    '20201060':('Commercial & Professional Services', 'N'),
    '20201070':('Commercial & Professional Services', 'N'),
    '20201080':('Commercial & Professional Services', 'N'),
    '20202010':('Commercial & Professional Services', 'N'),
    '20202020':('Commercial & Professional Services', 'N'),

    '25301010':('Consumer Services', 'N'),
    '25301020':('Consumer Services', 'N'),
    '25301030':('Consumer Services', 'N'),
    '25301040':('Consumer Services', 'N'),
    '25302010':('Consumer Services', 'N'),
    '25302020':('Consumer Services', 'N'),

    '25501010':('Retailing', 'N'),
    '25502010':('Retailing', 'N'),
    '25502020':('Retailing', 'N'),
    '25503010':('Retailing', 'N'),
    '25503020':('Retailing', 'N'),
    '25504010':('Retailing', 'N'),
    '25504020':('Retailing', 'N'),
    '25504030':('Retailing', 'N'),
    '25504040':('Retailing', 'N'),
    '25504050':('Retailing', 'N'),
    '25504060':('Retailing', 'N'),

    '35101010':('Health Care Equipment & Services', 'N'),
    '35101020':('Health Care Equipment & Services', 'N'),
    '35102010':('Health Care Equipment & Services', 'N'),
    '35102015':('Health Care Equipment & Services', 'N'),
    '35102020':('Health Care Equipment & Services', 'N'),
    '35102030':('Health Care Equipment & Services', 'N'),
    '35103010':('Health Care Equipment & Services', 'N'),

    '35201010':('Pharmaceuticals Biotechnology & Life Sciences', 'N'),
    '35202010':('Pharmaceuticals Biotechnology & Life Sciences', 'N'),
    '35203010':('Pharmaceuticals Biotechnology & Life Sciences', 'N'),

    '50101010':('Telecommunication Services', 'N'),
    '50101020':('Telecommunication Services', 'N'),
    '50102010':('Telecommunication Services', 'N'),

    # model Industry using GICS-Industry mapping, (subIndustry level) (2)
    '10102050': ('Coal & Consumable Fuels', 'N'),
    '15104050': ('Steel', 'N'),

    # model Cust-Industry (12)
    '10101010':('Energy ex Coal', 'Y'),
    '10101020':('Energy ex Coal', 'Y'),
    '10102010':('Energy ex Coal', 'Y'),
    '10102020':('Energy ex Coal', 'Y'),
    '10102030':('Energy ex Coal', 'Y'),
    '10102040':('Energy ex Coal', 'Y'),

    '15104010':('Metals & Mining ex Steel', 'Y'),
    '15104020':('Metals & Mining ex Steel', 'Y'),
    '15104025':('Metals & Mining ex Steel', 'Y'),
    '15104030':('Metals & Mining ex Steel', 'Y'),
    '15104040':('Metals & Mining ex Steel', 'Y'),
    '15104045':('Metals & Mining ex Steel', 'Y'),

    '20102010': ('Electrical Equipment', 'Y'), # Building Products
    '20104010': ('Electrical Equipment', 'Y'), # Electrical Equipment
    '20104020': ('Electrical Equipment', 'Y'), # Electrical Equipment

    '20105010': ('Trading Companies, Distributors & Conglomerates', 'Y'), # Industrial Conglomberates, Trading Companies
    '20107010': ('Trading Companies, Distributors & Conglomerates', 'Y'), # Industrial Conglomberates, Trading Companies

    '20301010': ('Transportation Non-Infrastructure', 'Y'),# Air Freight, Airlines, Marine
    '20302010': ('Transportation Non-Infrastructure', 'Y'),# Air Freight, Airlines, Marine
    '20303010': ('Transportation Non-Infrastructure', 'Y'),# Air Freight, Airlines, Marine

    '20304010': ('Transportation Infrastructure', 'Y'), # Road & Rail
    '20304020': ('Transportation Infrastructure', 'Y'), # Road & Rail
    '20305010': ('Transportation Infrastructure', 'Y'), # Transportation Infrastructure
    '20305020': ('Transportation Infrastructure', 'Y'), # Transportation Infrastructure
    '20305030': ('Transportation Infrastructure', 'Y'), # Transportation Infrastructure

    '25202010': ('Textiles, Apparel & Luxury Goods', 'Y'), # Leisure Equipmet& Products
    '25202020': ('Textiles, Apparel & Luxury Goods', 'Y'), # Leisure Equipmet& Products
    '25203010': ('Textiles, Apparel & Luxury Goods', 'Y'), # Textiles, Apparel & Luxury Goods
    '25203020': ('Textiles, Apparel & Luxury Goods', 'Y'), # Textiles, Apparel & Luxury Goods
    '25203030': ('Textiles, Apparel & Luxury Goods', 'Y'), # Textiles, Apparel & Luxury Goods

    '30201010': ('Beverages & Tobacco', 'Y'), # Beverages
    '30201020': ('Beverages & Tobacco', 'Y'),
    '30201030': ('Beverages & Tobacco', 'Y'),
    '30203010': ('Beverages & Tobacco', 'Y'),# Tobacco

    '30301010': ('Household & Personal Products', 'Y'), # Household Products
    '30302010': ('Household & Personal Products', 'Y'), # Personal Products

    '40201010':('Financials ex Banks', 'Y'),
    '40201020':('Financials ex Banks', 'Y'),
    '40201030':('Financials ex Banks', 'Y'),
    '40201040':('Financials ex Banks', 'Y'),
    '40202010':('Financials ex Banks', 'Y'),
    '40203010':('Financials ex Banks', 'Y'),
    '40203020':('Financials ex Banks', 'Y'),
    '40203030':('Financials ex Banks', 'Y'),
    '40203040':('Financials ex Banks', 'Y'),
    '40204010':('Financials ex Banks', 'Y'), # Mortgage REITs-B, no merge allow across sectors
    '40301010':('Financials ex Banks', 'Y'),
    '40301020':('Financials ex Banks', 'Y'),
    '40301030':('Financials ex Banks', 'Y'),
    '40301040':('Financials ex Banks', 'Y'),
    '40301050':('Financials ex Banks', 'Y'),

    '55101010':('Utilities ex Renewable', 'Y'),
    '55102010':('Utilities ex Renewable', 'Y'),
    '55103010':('Utilities ex Renewable', 'Y'),
    '55104010':('Utilities ex Renewable', 'Y'),

    '40402010':('Real Estate', 'Y'),
    '40402020':('Real Estate', 'Y'),
    '40402030':('Real Estate', 'Y'),
    '40402035':('Real Estate', 'Y'),
    '40402040':('Real Estate', 'Y'),
    '40402045':('Real Estate', 'Y'),
    '40402050':('Real Estate', 'Y'),
    '40402060':('Real Estate', 'Y'),
    '40402070':('Real Estate', 'Y'),
    '40403010':('Real Estate', 'Y'),
    '40403020':('Real Estate', 'Y'),
    '40403030':('Real Estate', 'Y'),
    '40403040':('Real Estate', 'Y'),
    '60101010':('Real Estate', 'Y'),
    '60101020':('Real Estate', 'Y'),
    '60101030':('Real Estate', 'Y'),
    '60101040':('Real Estate', 'Y'),
    '60101050':('Real Estate', 'Y'),
    '60101060':('Real Estate', 'Y'),
    '60101070':('Real Estate', 'Y'),
    '60101080':('Real Estate', 'Y'),
    '60102010':('Real Estate', 'Y'),
    '60102020':('Real Estate', 'Y'),
    '60102030':('Real Estate', 'Y'),
    '60102040':('Real Estate', 'Y'),
    }

guessMaps[('GICSCustom-CN2b', datetime.date(2018,9,29))] = {
    # map from GICSsubIndustry to model industry factors:
    # Y: model Cust-Industry
    # N: model Industry using GICS-Industry: any level
    # model Industry using GICS-Industry mapping, (industry level) (21)
    '15101010':('Chemicals', 'N'),
    '15101020':('Chemicals', 'N'),
    '15101030':('Chemicals', 'N'),
    '15101040':('Chemicals', 'N'),
    '15101050':('Chemicals', 'N'),
    '15103010':('Chemicals', 'N'),
    '15103020':('Chemicals', 'N'),

    '15102010':('Construction Materials', 'N'),

    '15105010':('Paper & Forest Products', 'N'),
    '15105020':('Paper & Forest Products', 'N'),

    '20101010':('Aerospace & Defense', 'N'),

    '20103010':('Construction & Engineering', 'N'),

    '20106010':('Machinery', 'N'),
    '20106015':('Machinery', 'N'),
    '20106020':('Machinery', 'N'),

    '25101010':('Auto Components', 'N'),
    '25101020':('Auto Components', 'N'),

    '25102010':('Automobiles', 'N'),
    '25102020':('Automobiles', 'N'),

    '25201010':('Household Durables', 'N'),
    '25201020':('Household Durables', 'N'),
    '25201030':('Household Durables', 'N'),
    '25201040':('Household Durables', 'N'),
    '25201050':('Household Durables', 'N'),

    # GICS2018-Sep new factor
    '25401010':('Media', 'Y'),
    '25401020':('Media', 'Y'),
    '25401025':('Media', 'Y'),
    '25401030':('Media', 'Y'),
    '25401040':('Media', 'Y'),
    '50201010':('Media', 'Y'),
    '50201020':('Media', 'Y'),
    '50201030':('Media', 'Y'),
    '50201040':('Media', 'Y'),
    '50203010':('Media', 'Y'),
    '50202010':('Entertainment', 'N'),
    '50202020':('Entertainment', 'N'),
    # GICS2018-Sep new factor END

    '30101010':('Food & Staples Retailing', 'N'),
    '30101020':('Food & Staples Retailing', 'N'),
    '30101030':('Food & Staples Retailing', 'N'),
    '30101040':('Food & Staples Retailing', 'N'),

    '30202010':('Food Products', 'N'),
    '30202020':('Food Products', 'N'),
    '30202030':('Food Products', 'N'),

    '40101010':('Banks', 'N'),
    '40101015':('Banks', 'N'),
    '40102010':('Banks', 'N'),

    # '45101010':('Internet Software & Services', 'N'),
    '45101010':('Retailing', 'N'), # mapped Internet Software & Services(left) to Internet & Direct Marketing Retail in GICS2018
    '45102010':('IT Services', 'N'),
    '45102020':('IT Services', 'N'),
    '45102030':('IT Services', 'N'),

    '45103010':('Software', 'N'),
    '45103020':('Software', 'N'),
    '45103030':('Software', 'N'),

    '45201010':('Communications Equipment', 'N'),
    '45201020':('Communications Equipment', 'N'),

    '45202010':('Technology Hardware, Storage & Peripherals', 'N'),
    '45202020':('Technology Hardware, Storage & Peripherals', 'N'),
    '45202030':('Technology Hardware, Storage & Peripherals', 'N'),

    '45203010':('Electronic Equipment, Instruments & Components', 'N'),
    '45203015':('Electronic Equipment, Instruments & Components', 'N'),
    '45203020':('Electronic Equipment, Instruments & Components', 'N'),
    '45203030':('Electronic Equipment, Instruments & Components', 'N'),
    '45204010':('Electronic Equipment, Instruments & Components', 'N'),

    '45301010':('Semiconductors & Semiconductor Equipment', 'N'),
    '45301020':('Semiconductors & Semiconductor Equipment', 'N'),
    '45205010':('Semiconductors & Semiconductor Equipment', 'N'),
    '45205020':('Semiconductors & Semiconductor Equipment', 'N'),

    '55105010':('Independent Power and Renewable Electricity Producers', 'N'),
    '55105020':('Independent Power and Renewable Electricity Producers', 'N'),

    # model Industry using GICS-Industry mapping, (industry group level)(6)
    '20201010':('Commercial & Professional Services', 'N'),
    '20201020':('Commercial & Professional Services', 'N'),
    '20201030':('Commercial & Professional Services', 'N'),
    '20201040':('Commercial & Professional Services', 'N'),
    '20201050':('Commercial & Professional Services', 'N'),
    '20201060':('Commercial & Professional Services', 'N'),
    '20201070':('Commercial & Professional Services', 'N'),
    '20201080':('Commercial & Professional Services', 'N'),
    '20202010':('Commercial & Professional Services', 'N'),
    '20202020':('Commercial & Professional Services', 'N'),

    '25301010':('Consumer Services', 'N'),
    '25301020':('Consumer Services', 'N'),
    '25301030':('Consumer Services', 'N'),
    '25301040':('Consumer Services', 'N'),
    '25302010':('Consumer Services', 'N'),
    '25302020':('Consumer Services', 'N'),

    '25501010':('Retailing', 'N'),
    '25502010':('Retailing', 'N'),
    '25502020':('Retailing', 'N'),
    '25503010':('Retailing', 'N'),
    '25503020':('Retailing', 'N'),
    '25504010':('Retailing', 'N'),
    '25504020':('Retailing', 'N'),
    '25504030':('Retailing', 'N'),
    '25504040':('Retailing', 'N'),
    '25504050':('Retailing', 'N'),
    '25504060':('Retailing', 'N'),

    '35101010':('Health Care Equipment & Services', 'N'),
    '35101020':('Health Care Equipment & Services', 'N'),
    '35102010':('Health Care Equipment & Services', 'N'),
    '35102015':('Health Care Equipment & Services', 'N'),
    '35102020':('Health Care Equipment & Services', 'N'),
    '35102030':('Health Care Equipment & Services', 'N'),
    '35103010':('Health Care Equipment & Services', 'N'),

    '35201010':('Pharmaceuticals Biotechnology & Life Sciences', 'N'),
    '35202010':('Pharmaceuticals Biotechnology & Life Sciences', 'N'),
    '35203010':('Pharmaceuticals Biotechnology & Life Sciences', 'N'),

    '50101010':('Telecommunication Services', 'N'),
    '50101020':('Telecommunication Services', 'N'),
    '50102010':('Telecommunication Services', 'N'),

    # model Industry using GICS-Industry mapping, (subIndustry level) (2)
    '10102050': ('Coal & Consumable Fuels', 'N'),
    '15104050': ('Steel', 'N'),

    # model Cust-Industry (12)
    '10101010':('Energy ex Coal', 'Y'),
    '10101020':('Energy ex Coal', 'Y'),
    '10102010':('Energy ex Coal', 'Y'),
    '10102020':('Energy ex Coal', 'Y'),
    '10102030':('Energy ex Coal', 'Y'),
    '10102040':('Energy ex Coal', 'Y'),

    '15104010':('Metals & Mining ex Steel', 'Y'),
    '15104020':('Metals & Mining ex Steel', 'Y'),
    '15104025':('Metals & Mining ex Steel', 'Y'),
    '15104030':('Metals & Mining ex Steel', 'Y'),
    '15104040':('Metals & Mining ex Steel', 'Y'),
    '15104045':('Metals & Mining ex Steel', 'Y'),

    '20102010': ('Electrical Equipment', 'Y'), # Building Products
    '20104010': ('Electrical Equipment', 'Y'), # Electrical Equipment
    '20104020': ('Electrical Equipment', 'Y'), # Electrical Equipment

    '20105010': ('Trading Companies, Distributors & Conglomerates', 'Y'), # Industrial Conglomberates, Trading Companies
    '20107010': ('Trading Companies, Distributors & Conglomerates', 'Y'), # Industrial Conglomberates, Trading Companies

    '20301010': ('Transportation Non-Infrastructure', 'Y'),# Air Freight, Airlines, Marine
    '20302010': ('Transportation Non-Infrastructure', 'Y'),# Air Freight, Airlines, Marine
    '20303010': ('Transportation Non-Infrastructure', 'Y'),# Air Freight, Airlines, Marine

    '20304010': ('Transportation Infrastructure', 'Y'), # Road & Rail
    '20304020': ('Transportation Infrastructure', 'Y'), # Road & Rail
    '20305010': ('Transportation Infrastructure', 'Y'), # Transportation Infrastructure
    '20305020': ('Transportation Infrastructure', 'Y'), # Transportation Infrastructure
    '20305030': ('Transportation Infrastructure', 'Y'), # Transportation Infrastructure

    '25202010': ('Textiles, Apparel & Luxury Goods', 'Y'), # Leisure Equipmet& Products
    '25202020': ('Textiles, Apparel & Luxury Goods', 'Y'), # Leisure Equipmet& Products
    '25203010': ('Textiles, Apparel & Luxury Goods', 'Y'), # Textiles, Apparel & Luxury Goods
    '25203020': ('Textiles, Apparel & Luxury Goods', 'Y'), # Textiles, Apparel & Luxury Goods
    '25203030': ('Textiles, Apparel & Luxury Goods', 'Y'), # Textiles, Apparel & Luxury Goods

    '30201010': ('Beverages & Tobacco', 'Y'), # Beverages
    '30201020': ('Beverages & Tobacco', 'Y'),
    '30201030': ('Beverages & Tobacco', 'Y'),
    '30203010': ('Beverages & Tobacco', 'Y'),# Tobacco

    '30301010': ('Household & Personal Products', 'Y'), # Household Products
    '30302010': ('Household & Personal Products', 'Y'), # Personal Products

    '40201010':('Financials ex Banks', 'Y'),
    '40201020':('Financials ex Banks', 'Y'),
    '40201030':('Financials ex Banks', 'Y'),
    '40201040':('Financials ex Banks', 'Y'),
    '40202010':('Financials ex Banks', 'Y'),
    '40203010':('Financials ex Banks', 'Y'),
    '40203020':('Financials ex Banks', 'Y'),
    '40203030':('Financials ex Banks', 'Y'),
    '40203040':('Financials ex Banks', 'Y'),
    '40204010':('Financials ex Banks', 'Y'), # Mortgage REITs-B, no merge allow across sectors
    '40301010':('Financials ex Banks', 'Y'),
    '40301020':('Financials ex Banks', 'Y'),
    '40301030':('Financials ex Banks', 'Y'),
    '40301040':('Financials ex Banks', 'Y'),
    '40301050':('Financials ex Banks', 'Y'),

    '55101010':('Utilities ex Renewable', 'Y'),
    '55102010':('Utilities ex Renewable', 'Y'),
    '55103010':('Utilities ex Renewable', 'Y'),
    '55104010':('Utilities ex Renewable', 'Y'),

    '40402010':('Real Estate', 'Y'),
    '40402020':('Real Estate', 'Y'),
    '40402030':('Real Estate', 'Y'),
    '40402035':('Real Estate', 'Y'),
    '40402040':('Real Estate', 'Y'),
    '40402045':('Real Estate', 'Y'),
    '40402050':('Real Estate', 'Y'),
    '40402060':('Real Estate', 'Y'),
    '40402070':('Real Estate', 'Y'),
    '40403010':('Real Estate', 'Y'),
    '40403020':('Real Estate', 'Y'),
    '40403030':('Real Estate', 'Y'),
    '40403040':('Real Estate', 'Y'),
    '60101010':('Real Estate', 'Y'),
    '60101020':('Real Estate', 'Y'),
    '60101030':('Real Estate', 'Y'),
    '60101040':('Real Estate', 'Y'),
    '60101050':('Real Estate', 'Y'),
    '60101060':('Real Estate', 'Y'),
    '60101070':('Real Estate', 'Y'),
    '60101080':('Real Estate', 'Y'),
    '60102010':('Real Estate', 'Y'),
    '60102020':('Real Estate', 'Y'),
    '60102030':('Real Estate', 'Y'),
    '60102040':('Real Estate', 'Y'),
    }

guessMaps[('GICSIndustries-Gold', datetime.date(2008,8,30))] = {
    '15104010': ('Metals & Mining ex Gold', 'Y'),
    '15104020': ('Metals & Mining ex Gold', 'Y'),
    '15104030': ('Gold', 'N'),
    '15104040': ('Metals & Mining ex Gold', 'Y'),
    '15104050': ('Metals & Mining ex Gold', 'Y'),
    '10102010': ('Oil, Gas & Consumable Fuels', 'N'),
    '10102020': ('Oil, Gas & Consumable Fuels', 'N'),
    '10102030': ('Oil, Gas & Consumable Fuels', 'N'),
    '20201020': ('IT Services', 'Y'),
    '20201040': ('Professional Services', 'Y'),
    '20301010': ('Air Freight & Logistics', 'N'),
    '25203010': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25203020': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25203030': ('Textiles, Apparel & Luxury Goods', 'N'),
    '30101010': ('Food & Staples Retailing', 'N'),
    '30101020': ('Food & Staples Retailing', 'N'),
    '30101030': ('Food & Staples Retailing', 'N'),
    '40101010': ('Commercial Banks', 'N'),
    '40201010': ('Diversified Financial Services', 'N'),
    '40201020': ('Diversified Financial Services', 'N'),
    '40201030': ('Diversified Financial Services', 'N'),
    '40401010': ('Real Estate Investment Trusts (REITs)', 'Y'),
    '40401020': ('Real Estate Management & Development', 'Y'),
    '45102010': ('IT Services', 'N'),
    '45203010': ('Electronic Equipment, Instruments & Components', 'N'),
    '45203020': ('Electronic Equipment, Instruments & Components', 'N'),
    '45203030': ('Electronic Equipment, Instruments & Components', 'N'),
    '45205010': ('Semiconductors & Semiconductor Equipment', 'Y'),
    '45205020': ('Semiconductors & Semiconductor Equipment', 'Y'),
    '55103010': ('Multi-Utilities', 'N'),
    }

guessMaps[('GICSCustom-NA', datetime.date(2008,8,30))] = {
    '10102010': ('Oil, Gas & Consumable Fuels', 'N'),
    '10102020': ('Oil, Gas & Consumable Fuels', 'N'),
    '10102030': ('Oil, Gas & Consumable Fuels', 'N'),
    '15101010': ('Materials ex Metals & Mining', 'Y'),
    '15101020': ('Materials ex Metals & Mining', 'Y'),
    '15101030': ('Materials ex Metals & Mining', 'Y'),
    '15101040': ('Materials ex Metals & Mining', 'Y'),
    '15101050': ('Materials ex Metals & Mining', 'Y'),
    '15102010': ('Materials ex Metals & Mining', 'Y'),
    '15103010': ('Materials ex Metals & Mining', 'Y'),
    '15103020': ('Materials ex Metals & Mining', 'Y'),
    '15104010': ('Metals & Mining ex Gold', 'Y'),
    '15104020': ('Metals & Mining ex Gold', 'Y'),
    '15104025': ('Metals & Mining ex Gold', 'Y'), 
    '15104030': ('Gold', 'N'),
    '15104040': ('Metals & Mining ex Gold', 'Y'),
    '15104045': ('Metals & Mining ex Gold', 'Y'),
    '15104050': ('Metals & Mining ex Gold', 'Y'),
    '15105010': ('Materials ex Metals & Mining', 'Y'),
    '15105020': ('Materials ex Metals & Mining', 'Y'),
    '20102010': ('Industrial & Machinery', 'Y'),
    '20103010': ('Industrial & Machinery', 'Y'),
    '20105010': ('Industrial & Machinery', 'Y'),
    '20106010': ('Industrial & Machinery', 'Y'),
    '20106015': ('Industrial & Machinery', 'Y'),
    '20106020': ('Industrial & Machinery', 'Y'),
    '20107010': ('Industrial & Machinery', 'Y'),
    '20201010': ('Commercial & Professional Services', 'N'),
    '20201030': ('Commercial & Professional Services', 'N'),
    '20201040': ('Commercial & Professional Services', 'N'),
    '20201050': ('Commercial & Professional Services', 'N'),
    '20201060': ('Commercial & Professional Services', 'N'),
    '20201070': ('Commercial & Professional Services', 'N'),
    '20201080': ('Commercial & Professional Services', 'N'),
    '20202010': ('Commercial & Professional Services', 'N'),
    '20202020': ('Commercial & Professional Services', 'N'),
    '20201020': ('IT Services', 'Y'),
    '20301010': ('Transportation', 'N'),
    '20302010': ('Transportation', 'N'),
    '20303010': ('Transportation', 'N'),
    '20304010': ('Transportation', 'N'),
    '20304020': ('Transportation', 'N'),
    '20305010': ('Transportation', 'N'),
    '20305020': ('Transportation', 'N'),
    '20305030': ('Transportation', 'N'),
    '25101010': ('Automobiles & Components', 'N'),
    '25101020': ('Automobiles & Components', 'N'),
    '25102010': ('Automobiles & Components', 'N'),
    '25102020': ('Automobiles & Components', 'N'),
    '25202010': ('Leisure & Apparel', 'Y'),
    '25202020': ('Leisure & Apparel', 'Y'),
    '25203010': ('Leisure & Apparel', 'Y'),
    '25203020': ('Leisure & Apparel', 'Y'),
    '25203030': ('Leisure & Apparel', 'Y'),
    '25301010': ('Consumer Services', 'N'),
    '25301020': ('Consumer Services', 'N'),
    '25301030': ('Consumer Services', 'N'),
    '25301040': ('Consumer Services', 'N'),
    '25302010': ('Consumer Services', 'N'),
    '25302020': ('Consumer Services', 'N'),
    '25501010': ('Retailing', 'N'),
    '25502010': ('Retailing', 'N'),
    '25502020': ('Retailing', 'N'),
    '25503010': ('Retailing', 'N'),
    '25503020': ('Retailing', 'N'),
    '25504010': ('Retailing', 'N'),
    '25504020': ('Retailing', 'N'),
    '25504030': ('Retailing', 'N'),
    '25504040': ('Retailing', 'N'),
    '25504050': ('Retailing', 'N'),
    '25504060': ('Retailing', 'N'),
    '30201010': ('Food, Beverage & Tobacco', 'N'),
    '30201020': ('Food, Beverage & Tobacco', 'N'),
    '30201030': ('Food, Beverage & Tobacco', 'N'),
    '30202010': ('Food, Beverage & Tobacco', 'N'),
    '30202020': ('Food, Beverage & Tobacco', 'N'),
    '30202030': ('Food, Beverage & Tobacco', 'N'),
    '30203010': ('Food, Beverage & Tobacco', 'N'),
    '30101010': ('Food & Staples Retailing', 'N'),
    '30101020': ('Food & Staples Retailing', 'N'),
    '30101030': ('Food & Staples Retailing', 'N'),
    '30301010': ('Household & Personal Products', 'N'),
    '30302010': ('Household & Personal Products', 'N'),
    '35101010': ('Health Care Equipment & Technology', 'N'),
    '35101020': ('Health Care Equipment & Technology', 'N'),
    '35103010': ('Health Care Equipment & Technology', 'N'),
    '35201010': ('Biotechnology & Life Sciences', 'N'),
    '35203010': ('Biotechnology & Life Sciences', 'N'),
    '40101010': ('Commercial Banks', 'N'),
    '40101015': ('Commercial Banks', 'N'),
    '40201010': ('Diversified Financials', 'N'),
    '40201020': ('Diversified Financials', 'N'),
    '40201030': ('Diversified Financials', 'N'),
    '40201040': ('Diversified Financials', 'N'),
    '40202010': ('Diversified Financials', 'N'),
    '40203010': ('Diversified Financials', 'N'),
    '40203020': ('Diversified Financials', 'N'),
    '40203030': ('Diversified Financials', 'N'),
    '40203040': ('Diversified Financials', 'N'),
    '40204010': ('Real Estate', 'N'),
    '40401010': ('Real Estate', 'N'),
    '40401020': ('Real Estate', 'N'),
    '40402010': ('Real Estate', 'N'),
    '40402020': ('Real Estate', 'N'),
    '40402030': ('Real Estate', 'N'),
    '40402035': ('Real Estate', 'N'),
    '40402040': ('Real Estate', 'N'),
    '40402045': ('Real Estate', 'N'),
    '40402050': ('Real Estate', 'N'),
    '40402060': ('Real Estate', 'N'),
    '40402070': ('Real Estate', 'N'),
    '40403010': ('Real Estate', 'N'),
    '40403020': ('Real Estate', 'N'),
    '40403030': ('Real Estate', 'N'),
    '40403040': ('Real Estate', 'N'),
    '45202030': ('Computers & Peripherals', 'N'),
    '45203010': ('Electronic Equipment, Instruments & Components', 'N'),
    '45203015': ('Electronic Equipment, Instruments & Components', 'N'),
    '45203020': ('Electronic Equipment, Instruments & Components', 'N'),
    '45203030': ('Electronic Equipment, Instruments & Components', 'N'),
    '45204010': ('Electronic Equipment, Instruments & Components', 'N'),
    '45102010': ('IT Services', 'N'),
    '45205010': ('Semiconductors & Semiconductor Equipment', 'Y'),
    '45205020': ('Semiconductors & Semiconductor Equipment', 'Y'),
    '50101010': ('Telecommunication Services', 'N'),
    '50101020': ('Telecommunication Services', 'N'),
    '50102010': ('Telecommunication Services', 'N'),
    '50202010': ('Media','Y'),    # CCHU 20180929
    '50202020': ('Software','Y'), # CCHU 20180929
    '50203010': ('Internet Software & Services','Y'), # CCHU 20180929
    '55101010': ('Utilities', 'N'),
    '55102010': ('Utilities', 'N'),
    '55103010': ('Utilities', 'N'),
    '55104010': ('Utilities', 'N'),
    '55105010': ('Utilities', 'N'),
    '55105020': ('Utilities', 'N'),
    '60102010': ('Real Estate', 'Y'),
    '60102020': ('Real Estate', 'Y'),
    '60102030': ('Real Estate', 'Y'),
    '60102040': ('Real Estate', 'Y'),
    '60101010': ('Real Estate', 'Y'),
    '60101020': ('Real Estate', 'Y'),
    '60101030': ('Real Estate', 'Y'),
    '60101040': ('Real Estate', 'Y'),
    '60101050': ('Real Estate', 'Y'),
    '60101060': ('Real Estate', 'Y'),
    '60101070': ('Real Estate', 'Y'),
    '60101080': ('Real Estate', 'Y'),
    }

guessMaps[('GICSCustom-NoMortgageREITs', datetime.date(2016,9,1))] = {
    '10102010': ('Oil, Gas & Consumable Fuels', 'N'),
    '10102020': ('Oil, Gas & Consumable Fuels', 'N'),
    '10102030': ('Oil, Gas & Consumable Fuels', 'N'),
    '20301010': ('Air Freight & Logistics', 'N'),
    '25202010': ('Leisure Products', 'N'), # Leisure Products
    '25202020': ('Leisure Products', 'N'), # Photographic Products
    '25203010': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25203020': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25203030': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25502010': ('Internet & Direct Marketing Retail', 'N'), # Catalog Retail (Discontinued)
    '25502020': ('Internet & Direct Marketing Retail', 'N'), # Internet Retail (Renamed)
    '30101010': ('Food & Staples Retailing', 'N'),
    '30101020': ('Food & Staples Retailing', 'N'),
    '30101030': ('Food & Staples Retailing', 'N'),
    '35102010': ('Health Care Providers & Services', 'N'),
    '35102015': ('Health Care Providers & Services', 'N'),
    '35102020': ('Health Care Providers & Services', 'N'),
    '35102030': ('Health Care Providers & Services', 'N'),
    '40101010': ('Banks', 'N'), # Diversified Bank
    '40101015': ('Banks', 'N'), # Regional Banks
    '40201010': ('Diversified Financial Services & Mortgage REITs', 'Y'),
    '40201020': ('Diversified Financial Services & Mortgage REITs', 'Y'),
    '40201030': ('Diversified Financial Services & Mortgage REITs', 'Y'),
    '40401010': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Real Estate Investment Trusts
    '40401020': ('Real Estate Management & Development', 'N'), # Real Estate Management & Development
    '40402010': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Diversified REITs
    '40402020': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Industrial REITs
    '40402035': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Hotel & Resort REITs
    '40402040': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Office REITs
    '40402045': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Health Care REITs
    '40402050': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Residential REITs
    '40402060': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Retail REITs
    '40402070': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Specialized REITs
    '40201040': ('Diversified Financial Services & Mortgage REITs', 'Y'),
    '40204010': ('Diversified Financial Services & Mortgage REITs', 'Y'),
    '40402030': ('Diversified Financial Services & Mortgage REITs', 'Y'), # Mortgage REITs <-- EXCLUDED FROM EQUITY REITS
    '45102010': ('IT Services', 'N'),
    '45202010': ('Technology Hardware, Storage & Peripherals', 'N'), # Computer Hardware
    '45202020': ('Technology Hardware, Storage & Peripherals', 'N'), # Computer Storage & Peripherals
    '45204010': ('Technology Hardware, Storage & Peripherals', 'N'), # Office Electronics
    '45203010': ('Electronic Equipment, Instruments & Components', 'N'), # Electronic Equipment & Instruments
    '45203020': ('Electronic Equipment, Instruments & Components', 'N'), # Electronic Manufacturing Services
    '45203030': ('Electronic Equipment, Instruments & Components', 'N'), # Technology Distributors
    '45205010': ('Semiconductors & Semiconductor Equipment', 'Y'),
    '45205020': ('Semiconductors & Semiconductor Equipment', 'Y'),
    '55103010': ('Multi-Utilities', 'N'),
    '55105010': ('Independent Power and Renewable Electricity Producers', 'N'), # Independent Power Producers & Energy Traders
    '55105020': ('Independent Power and Renewable Electricity Producers', 'N'),
    }

guessMaps[('GICSCustom-NoMortgageREITs2018', datetime.date(2018,9,29))] = {
    '10102010': ('Oil, Gas & Consumable Fuels', 'N'),
    '10102020': ('Oil, Gas & Consumable Fuels', 'N'),
    '10102030': ('Oil, Gas & Consumable Fuels', 'N'),
    '20201040': ('Professional Services', 'Y'),
    '20301010': ('Air Freight & Logistics', 'N'),
    '25202010': ('Leisure Products', 'N'), # Leisure Products
    '25202020': ('Leisure Products', 'N'), # Photographic Products
    '25203010': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25203020': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25203030': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25502010': ('Internet & Direct Marketing Retail', 'N'), # Catalog Retail (Discontinued)
    '25502020': ('Internet & Direct Marketing Retail', 'N'), # Internet Retail (Renamed)
    '30101010': ('Food & Staples Retailing', 'N'),
    '30101020': ('Food & Staples Retailing', 'N'),
    '30101030': ('Food & Staples Retailing', 'N'),
    '35102010': ('Health Care Providers & Services', 'N'),
    '35102015': ('Health Care Providers & Services', 'N'),
    '35102020': ('Health Care Providers & Services', 'N'),
    '35102030': ('Health Care Providers & Services', 'N'),
    '40101010': ('Banks', 'N'), # Diversified Bank
    '40101015': ('Banks', 'N'), # Regional Banks
    '40201010': ('Diversified Financial Services & Mortgage REITs', 'Y'),
    '40201020': ('Diversified Financial Services & Mortgage REITs', 'Y'),
    '40201030': ('Diversified Financial Services & Mortgage REITs', 'Y'),
    '40401010': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Real Estate Investment Trusts
    '40401020': ('Real Estate Management & Development', 'N'), # Real Estate Management & Development
    '40402010': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Diversified REITs
    '40402020': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Industrial REITs
    '40402035': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Hotel & Resort REITs
    '40402040': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Office REITs
    '40402045': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Health Care REITs
    '40402050': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Residential REITs
    '40402060': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Retail REITs
    '40402070': ('Equity Real Estate Investment Trusts (REITs)', 'N'), # Specialized REITs
    '40201040': ('Diversified Financial Services & Mortgage REITs', 'Y'),
    '40204010': ('Diversified Financial Services & Mortgage REITs', 'Y'),
    '40402030': ('Diversified Financial Services & Mortgage REITs', 'Y'), # Mortgage REITs <-- EXCLUDED FROM EQUITY REITS
    '45101010': ('Internet & Direct Marketing Retail', 'N'), # 2018 Addition
    '45102010': ('IT Services', 'N'),
    '45202010': ('Technology Hardware, Storage & Peripherals', 'N'), # Computer Hardware
    '45202020': ('Technology Hardware, Storage & Peripherals', 'N'), # Computer Storage & Peripherals
    '45204010': ('Technology Hardware, Storage & Peripherals', 'N'), # Office Electronics
    '45203010': ('Electronic Equipment, Instruments & Components', 'N'), # Electronic Equipment & Instruments
    '45203020': ('Electronic Equipment, Instruments & Components', 'N'), # Electronic Manufacturing Services
    '45203030': ('Electronic Equipment, Instruments & Components', 'N'), # Technology Distributors
    '45205010': ('Semiconductors & Semiconductor Equipment', 'Y'),
    '45205020': ('Semiconductors & Semiconductor Equipment', 'Y'),
    '55103010': ('Multi-Utilities', 'N'),
    '55105010': ('Independent Power and Renewable Electricity Producers', 'N'), # Independent Power Producers & Energy Traders
    '55105020': ('Independent Power and Renewable Electricity Producers', 'N'),
    }

guessMaps[('GICSCustom-EM', datetime.date(2016,9,1))] = {
    '10102010': ('Oil, Gas & Consumable Fuels', 'N'),
    '10102020': ('Oil, Gas & Consumable Fuels', 'N'),
    '10102030': ('Oil, Gas & Consumable Fuels', 'N'),
    '20201010': ('Commercial & Professional Services', 'Y'),
    '20201020': ('Commercial & Professional Services', 'Y'),
    '20201030': ('Commercial & Professional Services', 'Y'),
    '20201040': ('Commercial & Professional Services', 'Y'),
    '20201050': ('Commercial & Professional Services', 'Y'),
    '20201060': ('Commercial & Professional Services', 'Y'),
    '20201070': ('Commercial & Professional Services', 'Y'),
    '20201080': ('Commercial & Professional Services', 'Y'),
    '20202010': ('Commercial & Professional Services', 'Y'),
    '20202020': ('Commercial & Professional Services', 'Y'),
    '20301010': ('Air Freight & Logistics', 'N'),
    '25202010': ('Leisure Products', 'N'),
    '25202020': ('Leisure Products', 'N'),
    '25203010': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25203020': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25203030': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25502010': ('Internet & Direct Marketing Retail', 'N'),
    '25502020': ('Internet & Direct Marketing Retail', 'N'),
    '30101010': ('Food & Staples Retailing', 'N'),
    '30101020': ('Food & Staples Retailing', 'N'),
    '30101030': ('Food & Staples Retailing', 'N'),
    '35101010': ('Health Care Equipment, Supplies & Technology', 'Y'),
    '35101020': ('Health Care Equipment, Supplies & Technology', 'Y'),
    '35102010': ('Health Care Providers & Services', 'N'),
    '35102015': ('Health Care Providers & Services', 'N'),
    '35102020': ('Health Care Providers & Services', 'N'),
    '35102030': ('Health Care Providers & Services', 'N'),
    '35103010': ('Health Care Equipment, Supplies & Technology', 'Y'),
    '35201010': ('Biotechnology, Life Sciences Tools & Services', 'Y'),
    '35203010': ('Biotechnology, Life Sciences Tools & Services', 'Y'),
    '40101010': ('Banks, Thrifts & Mortgage Finance', 'Y'),
    '40101015': ('Banks, Thrifts & Mortgage Finance', 'Y'),
    '40102010': ('Banks, Thrifts & Mortgage Finance', 'Y'),
    '40201010': ('Diversified Financial Services & Mortgage REITs', 'Y'),
    '40201020': ('Diversified Financial Services & Mortgage REITs', 'Y'),
    '40201030': ('Diversified Financial Services & Mortgage REITs', 'Y'),
    '40201040': ('Diversified Financial Services & Mortgage REITs', 'Y'),
    '40204010': ('Diversified Financial Services & Mortgage REITs', 'Y'),
    '40401010': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '40401020': ('Real Estate Management & Development', 'N'),
    '40402010': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '40402020': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '40402030': ('Diversified Financial Services & Mortgage REITs', 'Y'),
    '40402035': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '40402040': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '40402045': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '40402050': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '40402060': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '40402070': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '45102010': ('IT Services', 'N'),
    '45202010': ('Technology Hardware, Storage & Peripherals', 'N'),
    '45202020': ('Technology Hardware, Storage & Peripherals', 'N'),
    '45203010': ('Electronic Equipment, Instruments & Components', 'N'),
    '45203020': ('Electronic Equipment, Instruments & Components', 'N'),
    '45203030': ('Electronic Equipment, Instruments & Components', 'N'),
    '45204010': ('Technology Hardware, Storage & Peripherals', 'N'),
    '45205010': ('Semiconductors & Semiconductor Equipment', 'Y'),
    '45205020': ('Semiconductors & Semiconductor Equipment', 'Y'),
    '55103010': ('Multi-Utilities', 'Y'),
    '55105010': ('Multi-Utilities', 'Y'),
    '55105020': ('Multi-Utilities', 'Y')}

guessMaps[('GICSCustom-EM2', datetime.date(2018,9,29))] = {
    '10102010': ('Oil, Gas & Consumable Fuels', 'N'),
    '10102020': ('Oil, Gas & Consumable Fuels', 'N'),
    '10102030': ('Oil, Gas & Consumable Fuels', 'N'),
    '20201010': ('Commercial & Professional Services', 'Y'),
    '20201020': ('Commercial & Professional Services', 'Y'),
    '20201030': ('Commercial & Professional Services', 'Y'),
    '20201040': ('Commercial & Professional Services', 'Y'),
    '20201050': ('Commercial & Professional Services', 'Y'),
    '20201060': ('Commercial & Professional Services', 'Y'),
    '20201070': ('Commercial & Professional Services', 'Y'),
    '20201080': ('Commercial & Professional Services', 'Y'),
    '20202010': ('Commercial & Professional Services', 'Y'),
    '20202020': ('Commercial & Professional Services', 'Y'),
    '20301010': ('Air Freight & Logistics', 'N'),
    '25202010': ('Leisure Products', 'N'),
    '25202020': ('Leisure Products', 'N'),
    '25203010': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25203020': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25203030': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25502010': ('Internet & Direct Marketing Retail', 'N'),
    '25502020': ('Internet & Direct Marketing Retail', 'N'),
    '30101010': ('Food & Staples Retailing', 'N'),
    '30101020': ('Food & Staples Retailing', 'N'),
    '30101030': ('Food & Staples Retailing', 'N'),
    '35101010': ('Health Care Equipment, Supplies & Technology', 'Y'),
    '35101020': ('Health Care Equipment, Supplies & Technology', 'Y'),
    '35102010': ('Health Care Providers & Services', 'N'),
    '35102015': ('Health Care Providers & Services', 'N'),
    '35102020': ('Health Care Providers & Services', 'N'),
    '35102030': ('Health Care Providers & Services', 'N'),
    '35103010': ('Health Care Equipment, Supplies & Technology', 'Y'),
    '35201010': ('Biotechnology, Life Sciences Tools & Services', 'Y'),
    '35203010': ('Biotechnology, Life Sciences Tools & Services', 'Y'),
    '40101010': ('Banks, Thrifts & Mortgage Finance', 'Y'),
    '40101015': ('Banks, Thrifts & Mortgage Finance', 'Y'),
    '40102010': ('Banks, Thrifts & Mortgage Finance', 'Y'),
    '40201010': ('Diversified Financial Services & Mortgage REITs', 'Y'),
    '40201020': ('Diversified Financial Services & Mortgage REITs', 'Y'),
    '40201030': ('Diversified Financial Services & Mortgage REITs', 'Y'),
    '40201040': ('Diversified Financial Services & Mortgage REITs', 'Y'),
    '40204010': ('Diversified Financial Services & Mortgage REITs', 'Y'),
    '40401010': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '40401020': ('Real Estate Management & Development', 'N'),
    '40402010': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '40402020': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '40402030': ('Diversified Financial Services & Mortgage REITs', 'Y'),
    '40402035': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '40402040': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '40402045': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '40402050': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '40402060': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '40402070': ('Equity Real Estate Investment Trusts (REITs)', 'N'),
    '45101010': ('Internet & Direct Marketing Retail', 'N'), # 2018 Addition
    '45102010': ('IT Services', 'N'),
    '45202010': ('Technology Hardware, Storage & Peripherals', 'N'),
    '45202020': ('Technology Hardware, Storage & Peripherals', 'N'),
    '45203010': ('Electronic Equipment, Instruments & Components', 'N'),
    '45203020': ('Electronic Equipment, Instruments & Components', 'N'),
    '45203030': ('Electronic Equipment, Instruments & Components', 'N'),
    '45204010': ('Technology Hardware, Storage & Peripherals', 'N'),
    '45205010': ('Semiconductors & Semiconductor Equipment', 'Y'),
    '45205020': ('Semiconductors & Semiconductor Equipment', 'Y'),
    '55103010': ('Multi-Utilities & Independent Power Producers & Traders', 'Y'),
    '55105010': ('Multi-Utilities & Independent Power Producers & Traders', 'Y'),
    '55105020': ('Multi-Utilities & Independent Power Producers & Traders', 'Y')}

guessMaps[('GICSCustom-SubInd', datetime.date(2008,8,30))] = {
    '10101010': ('Oil & Gas Drilling','N'),
    '10101020': ('Oil & Gas Equipment & Services','N'),
    '10102010': ('Integrated Oil & Gas','N'),
    '10102020': ('Oil & Gas Exploration & Production','N'),
    '10102030': ('Oil & Gas Refining & Marketing','N'),
    '10102040': ('Oil & Gas Storage & Transportation','N'),
    '10102050': ('Coal & Consumable Fuels','N'),
    '15101010': ('Commodity Chemicals','N'),
    '15101020': ('Diversified Chemicals','N'),
    '15101030': ('Fertilizers & Agricultural Chemicals','N'),
    '15101040': ('Industrial Gases','N'),
    '15101050': ('Specialty Chemicals','N'),
    '15102010': ('Construction Materials','N'),
    '15103010': ('Metal & Glass Containers','N'),
    '15103020': ('Paper Packaging','N'),
    '15104010': ('Aluminum','N'),
    '15104020': ('Diversified Metals & Mining','N'),
    '15104030': ('Gold','N'),
    '15104040': ('Precious Metals & Minerals','N'),
    '15104045': ('Precious Metals & Minerals','N'),
    '15104050': ('Steel','N'),
    '15105010': ('Forest Products','N'),
    '15105020': ('Paper Products','N'),
    '20101010': ('Aerospace & Defense','N'),
    '20102010': ('Building Products','N'),
    '20103010': ('Construction & Engineering','N'),
    '20104010': ('Electrical Components & Equipment','N'),
    '20104020': ('Heavy Electrical Equipment','N'),
    '20105010': ('Industrial Conglomerates','N'),
    '20106010': ('Construction & Farm Machinery & Heavy Trucks','N'),
    '20106015': ('Construction & Farm Machinery & Heavy Trucks','N'),
    '20106020': ('Industrial Machinery','N'),
    '20107010': ('Trading Companies & Distributors','N'),
    '20201010': ('Commercial Printing','N'),
    '20201020': ('Data Processing & Outsourced Services','N'),
    '20201030': ('Diversified Support Services','N'),
    '20201040': ('Human Resource & Employment Services','N'),
    '20201050': ('Environmental & Facilities Services','N'),
    '20201060': ('Office Services & Supplies','N'),
    '20201070': ('Diversified Support Services','N'),
    '20201080': ('Security & Alarm Services','N'),
    '20202010': ('Human Resource & Employment Services','N'),
    '20202020': ('Research & Consulting Services','N'),
    '20301010': ('Air Freight & Logistics','N'),
    '20302010': ('Airlines','N'),
    '20303010': ('Marine','N'),
    '20304010': ('Railroads','N'),
    '20304020': ('Trucking','N'),
    '20305010': ('Airport Services','N'),
    '20305020': ('Highways & Railtracks','N'),
    '20305030': ('Marine Ports & Services','N'),
    '25101010': ('Auto Parts & Equipment','N'),
    '25101020': ('Tires & Rubber','N'),
    '25102010': ('Automobile Manufacturers','N'),
    '25102020': ('Motorcycle Manufacturers','N'),
    '25201010': ('Consumer Electronics','N'),
    '25201020': ('Home Furnishings','N'),
    '25201030': ('Homebuilding','N'),
    '25201040': ('Household Appliances','N'),
    '25201050': ('Housewares & Specialties','N'),
    '25202010': ('Leisure Products','N'),
    '25202020': ('Photographic Products','N'),
    '25203010': ('Apparel Accessories & Luxury Goods','N'),
    '25203020': ('Footwear','N'),
    '25203030': ('Textiles','N'),
    '25301010': ('Casinos & Gaming','N'),
    '25301020': ('Hotels Resorts & Cruise Lines','N'),
    '25301030': ('Leisure Facilities','N'),
    '25301040': ('Restaurants','N'),
    '25302010': ('Education Services','N'),
    '25302020': ('Specialized Consumer Services','N'),
    '25401010': ('Advertising','N'),
    '25401020': ('Broadcasting','N'),
    '25401025': ('Cable & Satellite','N'),
    '25401030': ('Movies & Entertainment','N'),
    '25401040': ('Publishing','N'),
    '25501010': ('Distributors','N'),
    '25502010': ('Catalog Retail','N'),
    '25502020': ('Internet Retail','N'),
    '25503010': ('Department Stores','N'),
    '25503020': ('General Merchandise Stores','N'),
    '25504010': ('Apparel Retail','N'),
    '25504020': ('Computer & Electronics Retail','N'),
    '25504030': ('Home Improvement Retail','N'),
    '25504040': ('Specialty Stores','N'),
    '25504050': ('Automotive Retail','N'),
    '25504060': ('Homefurnishing Retail','N'),
    '30101010': ('Drug Retail','N'),
    '30101020': ('Food Distributors','N'),
    '30101030': ('Food Retail','N'),
    '30101040': ('Hypermarkets & Super Centers','N'),
    '30201010': ('Brewers','N'),
    '30201020': ('Distillers & Vintners','N'),
    '30201030': ('Soft Drinks','N'),
    '30202010': ('Agricultural Products','N'),
    '30202020': ('Packaged Foods & Meats','N'),
    '30202030': ('Packaged Foods & Meats','N'),
    '30203010': ('Tobacco','N'),
    '30301010': ('Household Products','N'),
    '30302010': ('Personal Products','N'),
    '35101010': ('Health Care Equipment','N'),
    '35101020': ('Health Care Supplies','N'),
    '35102010': ('Health Care Distributors','N'),
    '35102015': ('Health Care  Services','N'),
    '35102020': ('Health Care Facilities','N'),
    '35102030': ('Managed Health Care','N'),
    '35103010': ('Health Care Technology','N'),
    '35201010': ('Biotechnology','N'),
    '35202010': ('Pharmaceuticals','N'),
    '35203010': ('Life Sciences Tools & Services','N'),
    '40101010': ('Diversified Banks','N'),
    '40101015': ('Regional Banks','N'),
    '40102010': ('Thrifts & Mortgage Finance','N'),
    '40201010': ('Consumer Finance','N'),
    '40201020': ('Other Diversified Financial Services','N'),
    '40201030': ('Multi-Sector Holdings','N'),
    '40201040': ('Specialized Finance','N'),
    '40202010': ('Consumer Finance','N'),
    '40203010': ('Asset Management & Custody Banks','N'),
    '40203020': ('Investment Banking & Brokerage','N'),
    '40203030': ('Diversified Capital Markets','N'),
    '40301010': ('Insurance Brokers','N'),
    '40301020': ('Life & Health Insurance','N'),
    '40301030': ('Multi-line Insurance','N'),
    '40301040': ('Property & Casualty Insurance','N'),
    '40301050': ('Reinsurance','N'),
    '40401010': ('Specialized REITs','N'),
    '40401020': ('Diversified Real Estate Activities','N'),
    '40402010': ('Diversified REITs','N'),
    '40402020': ('Industrial REITs','N'),
    '40402030': ('Mortgage REITs','N'),
    '40402035': ('Mortgage REITs','N'),
    '40402040': ('Office REITs','N'),
    '40402045': ('Office REITs','N'),
    '40402050': ('Residential REITs','N'),
    '40402060': ('Retail REITs','N'),
    '40402070': ('Specialized REITs','N'),
    '40403010': ('Diversified Real Estate Activities','N'),
    '40403020': ('Real Estate Operating Companies','N'),
    '40403030': ('Real Estate Development','N'),
    '40403040': ('Real Estate Services','N'),
    '45101010': ('Internet Software & Services','N'),
    '45102010': ('IT Consulting & Other Services','N'),
    '45102020': ('Data Processing & Outsourced Services','N'),
    '45103010': ('Application Software','N'),
    '45103020': ('Systems Software','N'),
    '45103030': ('Home Entertainment Software','N'),
    '45201020': ('Communications Equipment','N'),
    '45201010': ('Communications Equipment','N'),
    '45202010': ('Computer Hardware','N'),
    '45202020': ('Computer Storage & Peripherals','N'),
    '45202030': ('Computer Storage & Peripherals','N'),
    '45203010': ('Electronic Equipment & Instruments','N'),
    '45203015': ('Electronic Components','N'),
    '45203020': ('Electronic Manufacturing Services','N'),
    '45203030': ('Technology Distributors','N'),
    '45204010': ('Office Electronics','N'),
    '45205010': ('Semiconductor Equipment','N'),
    '45205020': ('Semiconductors','N'),
    '45301010': ('Semiconductor Equipment','N'),
    '45301020': ('Semiconductors','N'),
    '50101010': ('Alternative Carriers','N'),
    '50101020': ('Integrated Telecommunication Services','N'),
    '50102010': ('Wireless Telecommunication Services','N'),
    '55101010': ('Electric Utilities','N'),
    '55102010': ('Gas Utilities','N'),
    '55103010': ('Multi-Utilities','N'),
    '55104010': ('Water Utilities','N'),
    '55105010': ('Independent Power Producers & Energy Traders','N'),
    '55105020': ('Independent Power Producers & Energy Traders','N'),
    }

guessMaps[('GICSCustom-NoOE', datetime.date(2008,8,30))] = {
    # Computers & Peripherals
    '45202030': ('Computers & Peripherals', 'N'),
    '45204010': ('Computers & Peripherals', 'N'),
    # Non custom mappings
    '10102010': ('Oil, Gas & Consumable Fuels', 'N'),
    '10102020': ('Oil, Gas & Consumable Fuels', 'N'),
    '10102030': ('Oil, Gas & Consumable Fuels', 'N'),
    '20201040': ('Professional Services', 'Y'),
    '20301010': ('Air Freight & Logistics', 'N'),
    '25202010': ('Leisure Equipment & Products', 'N'),
    '25203010': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25203020': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25203030': ('Textiles, Apparel & Luxury Goods', 'N'),
    '30101010': ('Food & Staples Retailing', 'N'),
    '30101020': ('Food & Staples Retailing', 'N'),
    '30101030': ('Food & Staples Retailing', 'N'),
    '40101010': ('Commercial Banks', 'N'),
    '40101015': ('Commercial Banks', 'N'),
    '40401010': ('Real Estate Investment Trusts (REITs)', 'N'),
    '40401020': ('Real Estate Management & Development', 'N'),
    '40201010': ('Diversified Financial Services', 'N'),
    '40201020': ('Diversified Financial Services', 'N'),
    '40201030': ('Diversified Financial Services', 'N'),
    '45102010': ('IT Services', 'N'),
    '45203010': ('Electronic Equipment, Instruments & Components', 'N'),
    '45203015': ('Electronic Equipment, Instruments & Components', 'N'),
    '45203020': ('Electronic Equipment, Instruments & Components', 'N'),
    '45203030': ('Electronic Equipment, Instruments & Components', 'N'),
    '45205010': ('Semiconductors & Semiconductor Equipment', 'N'),
    '45205020': ('Semiconductors & Semiconductor Equipment', 'N'),
    '55103010': ('Multi-Utilities', 'N'),
    '55105010': ('Independent Power Producers & Energy Traders', 'N'),
    '55105020': ('Independent Power Producers & Energy Traders', 'N'),
    }

guessMaps[('GICSCustom-US', datetime.date(2008,8,30))] = {
    # Chemicals & Constr. Mat.
    '15101010': ('Chemicals & Construction Materials', 'Y'),
    '15101020': ('Chemicals & Construction Materials', 'Y'),
    '15101030': ('Chemicals & Construction Materials', 'Y'),
    '15101040': ('Chemicals & Construction Materials', 'Y'),
    '15101050': ('Chemicals & Construction Materials', 'Y'),
    '15102010': ('Chemicals & Construction Materials', 'Y'),
    # Conglomerates & Machinery
    '20105010': ('Industrial Conglomerates & Machinery', 'Y'),
    '20106010': ('Industrial Conglomerates & Machinery', 'Y'),
    '20106015': ('Industrial Conglomerates & Machinery', 'Y'),
    '20106020': ('Industrial Conglomerates & Machinery', 'Y'),
    # Airlines, Air Freight & Transportation Infrastructure
    '20301010': ('Airline, Air Freight & Transportation Infrastructure', 'Y'),
    '20302010': ('Airline, Air Freight & Transportation Infrastructure', 'Y'),
    '20305010': ('Airline, Air Freight & Transportation Infrastructure', 'Y'),
    '20305020': ('Airline, Air Freight & Transportation Infrastructure', 'Y'),
    '20305030': ('Airline, Air Freight & Transportation Infrastructure', 'Y'),
    # Automobiles & Components
    '25101010': ('Automobiles & Components', 'N'),
    '25101020': ('Automobiles & Components', 'N'),
    '25102010': ('Automobiles & Components', 'N'),
    '25102020': ('Automobiles & Components', 'N'),
    # Distributors & Multiline Retail
    '25501010': ('Distributors & Multiline Retail', 'Y'),
    '25503010': ('Distributors & Multiline Retail', 'Y'),
    '25503020': ('Distributors & Multiline Retail', 'Y'),
    # Beverages & Tobacco
    '30201010': ('Beverages & Tobacco', 'Y'),
    '30201020': ('Beverages & Tobacco', 'Y'),
    '30201030': ('Beverages & Tobacco', 'Y'),
    '30203010': ('Beverages & Tobacco', 'Y'),
    # Household & Personal Products
    '30301010': ('Household & Personal Products', 'N'),
    '30302010': ('Household & Personal Products', 'N'),
    # Computers & Peripherals
    '45202030': ('Computers & Peripherals', 'N'),
    '45204010': ('Computers & Peripherals', 'N'),
    # Telecommunication Services
    '50101010': ('Telecommunication Services', 'N'),
    '50101020': ('Telecommunication Services', 'N'),
    '50102010': ('Telecommunication Services', 'N'),

    # Non custom mappings
    '10102010': ('Oil, Gas & Consumable Fuels', 'N'),
    '10102020': ('Oil, Gas & Consumable Fuels', 'N'),
    '10102030': ('Oil, Gas & Consumable Fuels', 'N'),
    '20201040': ('Professional Services', 'N'),
    '25202010': ('Leisure Equipment & Products', 'N'),
    '25203010': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25203020': ('Textiles, Apparel & Luxury Goods', 'N'),
    '25203030': ('Textiles, Apparel & Luxury Goods', 'N'),
    '30101010': ('Food & Staples Retailing', 'N'),
    '30101020': ('Food & Staples Retailing', 'N'),
    '30101030': ('Food & Staples Retailing', 'N'),
    '40101010': ('Commercial Banks', 'N'),
    '40101015': ('Commercial Banks', 'N'),
    '40401010': ('Real Estate Investment Trusts (REITs)', 'N'),
    '40401020': ('Real Estate Management & Development', 'N'),
    '40201010': ('Diversified Financial Services', 'N'),
    '40201020': ('Diversified Financial Services', 'N'),
    '40201030': ('Diversified Financial Services', 'N'),
    '45102010': ('IT Services', 'N'),
    '45203010': ('Electronic Equipment, Instruments & Components', 'N'),
    '45203015': ('Electronic Equipment, Instruments & Components', 'N'),
    '45203020': ('Electronic Equipment, Instruments & Components', 'N'),
    '45203030': ('Electronic Equipment, Instruments & Components', 'N'),
    '45205010': ('Semiconductors & Semiconductor Equipment', 'N'),
    '45205020': ('Semiconductors & Semiconductor Equipment', 'N'),
    '55103010': ('Multi-Utilities', 'N'),
    '55105010': ('Independent Power Producers & Energy Traders', 'N'),
    '55105020': ('Independent Power Producers & Energy Traders', 'N'),
    }

guessMaps[('GICSCustom-TW', datetime.date(2008,8,30))] = {
    # Chemicals
    '15101010': ('Chemicals', 'N'),
    '15101020': ('Chemicals', 'N'),
    '15101030': ('Chemicals', 'N'),
    '15101040': ('Chemicals', 'N'),
    '15101050': ('Chemicals', 'N'),
    # Construction Materials
    '15102010': ('Materials ex Chemicals', 'Y'),
    # Containers & Packaging
    '15103010': ('Materials ex Chemicals', 'Y'),
    '15103020': ('Materials ex Chemicals', 'Y'),
    # Metals & Mining
    '15104010': ('Materials ex Chemicals', 'Y'),
    '15104020': ('Materials ex Chemicals', 'Y'),
    '15104025': ('Materials ex Chemicals', 'Y'),
    '15104030': ('Materials ex Chemicals', 'Y'),
    '15104040': ('Materials ex Chemicals', 'Y'),
    '15104045': ('Materials ex Chemicals', 'Y'),
    '15104050': ('Materials ex Chemicals', 'Y'),
    # Paper & Forest Products
    '15105010': ('Materials ex Chemicals', 'Y'),
    '15105020': ('Materials ex Chemicals', 'Y'),
    # Aerospace
    '20101010': ('Capital Goods ex Electrical Equipment', 'Y'),
    # Machinery
    '20106010': ('Capital Goods ex Electrical Equipment', 'Y'),
    '20106015': ('Capital Goods ex Electrical Equipment', 'Y'),
    '20106020': ('Capital Goods ex Electrical Equipment', 'Y'),
    # Electrical Equipment
    '20104010': ('Electrical Equipment', 'N') ,
    '20104020': ('Electrical Equipment', 'N'),
    # Construction & Engineering
    '20103010': ('Capital Goods ex Electrical Equipment', 'Y'),
    # Building Products
    '20102010': ('Capital Goods ex Electrical Equipment', 'Y'),
    # Conglomerates
    '20105010': ('Capital Goods ex Electrical Equipment', 'Y'),
    #Trading Companies & Distributors
    '20107010': ('Capital Goods ex Electrical Equipment', 'Y'),
    #Professional Services
    '20201010': ('Commercial & Professional Services', 'N'),
    '20201020': ('Commercial & Professional Services', 'N'),
    '20201030': ('Commercial & Professional Services', 'N'),
    '20201040': ('Commercial & Professional Services', 'N'),
    '20201050': ('Commercial & Professional Services', 'N'),
    '20201060': ('Commercial & Professional Services', 'N'),
    '20201070': ('Commercial & Professional Services', 'N'),
    '20201080': ('Commercial & Professional Services', 'N'),
    '20202010': ('Commercial & Professional Services', 'N'),
    '20202020': ('Commercial & Professional Services', 'N'),
    #Transportation
    '20301010': ('Transportation', 'Y'),
    '20302010': ('Transportation', 'Y'),
    '20303010': ('Transportation', 'Y'),
    '20304010': ('Transportation', 'Y'),
    '20304020': ('Transportation', 'Y'),
    '20305010': ('Transportation', 'Y'),
    '20305020': ('Transportation', 'Y'),
    '20305030': ('Transportation', 'Y'),
    #Automobiles & Components
    '25101010': ('Consumer Discretionary ex Durables & Services', 'Y'),
    '25101020': ('Consumer Discretionary ex Durables & Services', 'Y'),
    '25102010': ('Consumer Discretionary ex Durables & Services', 'Y'),
    '25102020': ('Consumer Discretionary ex Durables & Services', 'Y'),
    # Media
    '25401010': ('Consumer Discretionary ex Durables & Services', 'Y'),
    '25401020': ('Consumer Discretionary ex Durables & Services', 'Y'),
    '25401025': ('Consumer Discretionary ex Durables & Services', 'Y'),
    '25401030': ('Consumer Discretionary ex Durables & Services', 'Y'),
    '25401040': ('Consumer Discretionary ex Durables & Services', 'Y'),
    #Retailing
    '25501010': ('Consumer Discretionary ex Durables & Services', 'Y'),
    '25502010': ('Consumer Discretionary ex Durables & Services', 'Y'),
    '25502020': ('Consumer Discretionary ex Durables & Services', 'Y'),
    '25503010': ('Consumer Discretionary ex Durables & Services', 'Y'),
    '25503020': ('Consumer Discretionary ex Durables & Services', 'Y'),
    '25504010': ('Consumer Discretionary ex Durables & Services', 'Y'),
    '25504020': ('Consumer Discretionary ex Durables & Services', 'Y'),
    '25504030': ('Consumer Discretionary ex Durables & Services', 'Y'),
    '25504040': ('Consumer Discretionary ex Durables & Services', 'Y'),
    '25504050': ('Consumer Discretionary ex Durables & Services', 'Y'),
    '25504060': ('Consumer Discretionary ex Durables & Services', 'Y'),
    # Household Durables & Consumer Services
    '25201010': ('Consumer Durables & Apparel', 'N'),
    '25201020': ('Consumer Durables & Apparel', 'N'),
    '25201030': ('Consumer Durables & Apparel', 'N'),
    '25201040': ('Consumer Durables & Apparel', 'N'),
    '25201050': ('Consumer Durables & Apparel', 'N'),
    # Leisure Equipmet& Products
    '25202010': ('Consumer Durables & Apparel', 'N'),
    '25202020': ('Consumer Durables & Apparel', 'N'),
    # Textiles, Apparel & Luxury Goods
    '25203010': ('Consumer Durables & Apparel', 'N'),
    '25203020': ('Consumer Durables & Apparel', 'N'),
    '25203030': ('Consumer Durables & Apparel', 'N'),
    #Consumer Services
    '25301010': ('Consumer Services', 'N'),
    '25301020': ('Consumer Services', 'N'),
    '25301030': ('Consumer Services', 'N'),
    '25301040': ('Consumer Services', 'N'),
    '25302010': ('Consumer Services', 'N'),
    '25302020': ('Consumer Services', 'N'),
    # Food & Staples Retailing
    '30101010': ('Consumer Staples', 'N'),
    '30101020': ('Consumer Staples', 'N'),
    '30101030': ('Consumer Staples', 'N'),
    '30101040': ('Consumer Staples', 'N'),
    # Food, Beverage & Tobacco
    '30201010': ('Consumer Staples', 'N'),
    '30201020': ('Consumer Staples', 'N'),
    '30201030': ('Consumer Staples', 'N'),
    '30202010': ('Consumer Staples', 'N'),
    '30202020': ('Consumer Staples', 'N'),
    '30202030': ('Consumer Staples', 'N'),
    '30203010': ('Consumer Staples', 'N'),
    # Household & Personal Products
    '30301010': ('Consumer Staples', 'N'),
    '30302010': ('Consumer Staples', 'N'),
    # Health Care Equipment & Services
    '35101010': ('Health Care', 'N'),
    '35101020': ('Health Care', 'N'),
    '35102010': ('Health Care', 'N'),
    '35102015': ('Health Care', 'N'),
    '35102020': ('Health Care', 'N'),
    '35102030': ('Health Care', 'N'),
    '35103010': ('Health Care', 'N'),
    # Pharmaceuticals, Biotechnology & Life Sciences
    '35201010': ('Health Care', 'N'),
    '35202010': ('Health Care', 'N'),
    '35203010': ('Health Care', 'N'),
    #Diversified Financials
    '40101010': ('Banks', 'N'),
    '40101015': ('Banks', 'N'),
    '40102010': ('Banks', 'N'),
    '40201010': ('Diversified Financials', 'N'),
    '40201020': ('Diversified Financials', 'N'),
    '40201030': ('Diversified Financials', 'N'),
    '40201040': ('Diversified Financials', 'N'),
    '40202010': ('Diversified Financials', 'N'),
    '40203010': ('Diversified Financials', 'N'),
    '40203020': ('Diversified Financials', 'N'),
    '40203030': ('Diversified Financials', 'N'),
    '40204010': ('Real Estate', 'N'),
    #Insurance
    '40301010': ('Insurance', 'N'),
    '40301020': ('Insurance', 'N'),
    '40301030': ('Insurance', 'N'),
    '40301040': ('Insurance', 'N'),
    '40301050': ('Insurance', 'N'),
    #Real Estate
    '40401010': ('Real Estate', 'N'),
    '40401020': ('Real Estate', 'N'),
    '40402010': ('Real Estate', 'N'),
    '40402020': ('Real Estate', 'N'),
    '40402030': ('Real Estate', 'N'),
    '40402040': ('Real Estate', 'N'),
    '40402050': ('Real Estate', 'N'),
    '40402060': ('Real Estate', 'N'),
    '40402070': ('Real Estate', 'N'),
    '40403010': ('Real Estate', 'N'),
    '40403020': ('Real Estate', 'N'),
    '40403030': ('Real Estate', 'N'),
    '40403040': ('Real Estate', 'N'),
    #Software & Services
    '45101010': ('Software & Services', 'N'),
    '45102010': ('Software & Services', 'N'),
    '45102020': ('Software & Services', 'N'),
    '45103010': ('Software & Services', 'N'),
    '45103020': ('Software & Services', 'N'),
    '45103030': ('Software & Services', 'N'),
    #Comunnications Equipment
    '45201020': ('Communications Equipment', 'N'),
    '45201010': ('Communications Equipment', 'N'),
    #Office Electronics
    '45204010': ('Electronics', 'Y'),
    #Computer hardware
    '45202010': ('Computer Hardware','Y'),
    #Storage & Peripherals
    '45202020': ('Computer Storage & Peripherals','Y'),
    '45202030': ('Computer Storage & Peripherals','Y'),
    #Equipment & Instruments
    '45203010': ('Electronics','Y'),
    #Manufacturing Services
    '45203020': ('Electronics','Y'),
    #Components
    '45203015': ('Electronics','Y'),
    #Technology Distributors
    '45203030': ('Electronics','Y'),
    #Semiconductors 
    '45205020': ('Semiconductors','Y'),
    '45301020': ('Semiconductors','Y'),
    #Semiconductor Equipment
    '45205010': ('Semiconductor Equipment','Y'),
    '45301010': ('Semiconductor Equipment','Y'),
    '50201010': ('Consumer Discretionary ex Durables & Services','Y'), # CCHU 20180929
    '50201020': ('Consumer Discretionary ex Durables & Services','Y'), # CCHU 20180929
    '50201030': ('Consumer Discretionary ex Durables & Services','Y'), # CCHU 20180929
    '50201040': ('Consumer Discretionary ex Durables & Services','Y'), # CCHU 20180929
    '50202010': ('Consumer Discretionary ex Durables & Services','Y'), # CCHU 20180929
    '50202020': ('Software & Services','Y'),                           # CCHU 20180929
    '50203010': ('Software & Services','Y'),                           # CCHU 20180929
    }

industryMaps = {}
industryMaps[datetime.date(2006,4,29)] = {
    'Hotels, Restaurants & Leisure': 'Hotels Restaurants & Leisure',
    'Food, Beverage & Tobacco': 'Food Beverage & Tobacco',
    }
industryMaps[datetime.date(2008,8,30)] = {
    'Hotels Restaurants & Leisure': 'Hotels, Restaurants & Leisure',
    'Food Beverage & Tobacco': 'Food, Beverage & Tobacco',
    }
industryMaps[datetime.date(2014,3,1)] = {
    'Hotels Restaurants & Leisure': 'Hotels, Restaurants & Leisure',
    'Food Beverage & Tobacco': 'Food, Beverage & Tobacco',
    }
industryMaps[datetime.date(2016,9,1)] = {
    'Hotels Restaurants & Leisure': 'Hotels, Restaurants & Leisure',
    'Food Beverage & Tobacco': 'Food, Beverage & Tobacco',
    }
industryMaps[datetime.date(2018,9,29)] = {
    'Hotels Restaurants & Leisure': 'Hotels, Restaurants & Leisure',
    'Food Beverage & Tobacco': 'Food, Beverage & Tobacco',
    }

def mapRevision(mktSubIndustries, myGuessMap, industryMap, mdlIndustriesMap,
                mdlIDIndustryMap, mdlLeafIDs, mdl):
    # Match sub-industries to industries by code
    # Match sub-industries in myGuessMap to the guessed value
    # and flag it as such
    for (si, parent) in mktSubIndustries:
        if si.code in myGuessMap:
            (mdlIndustry, guessFlag) = myGuessMap[si.code]
            mdlIndustryTry = mdlIndustriesMap.get(mdlIndustry)
            if mdlIndustryTry is None:
                allInds = sorted(i.description for i in mdlIndustriesMap.values())
                possibles = []
                for ind in allInds:
                    prob = SequenceMatcher(None, mdlIndustry, ind).ratio()
                    if prob > 0.5:
                        possibles.append(ind)
                logging.error('Can\'t map %s, %s: possible matches: %s',
                              si.code, mdlIndustry, possibles)
            mdlIndustry = mdlIndustryTry
        else:
            oldParent = industryMap.get(parent, parent)
            mdlIndustry = mdlIndustriesMap.get(oldParent)
            guessFlag = 'N'
        if mdlIndustry is None:
            logging.warning('No mapping for %s, %s, %s', si.code, si.name,
                         parent)
        else:
            logging.debug('Mapping %s, %s to %s', si.code, si.name,
                          mdlIndustry.name)
            mdl.dbCursor.execute("""SELECT model_ref_id, market_ref_id,
              flag_as_guessed FROM classification_market_map
              WHERE market_ref_id=:mref""", mref=si.id)
            r = mdl.dbCursor.fetchall()
            r = [i for i in r if i[0] in mdlLeafIDs]
            assert(len(r) <= 1)
            if len(r) == 1:
                if (mdlIndustry.id, si.id, guessFlag) == r[0]:
                    logging.debug('Matching record in database')
                else:
                    logging.warning('Changing map for %s from (%s,%s) to (%s,%s)',
                                 si.name, mdlIDIndustryMap[r[0][0]].name,
                                 r[0][2], mdlIndustry.name, guessFlag)
                    mdl.dbCursor.execute("""UPDATE classification_market_map
                          SET model_ref_id=:mdlID, flag_as_guessed=:flag
                          WHERE market_ref_id=:mktID
                          AND model_ref_id=:oldMdlID""",
                                         mdlID=mdlIndustry.id,
                                         oldMdlID=r[0][0],
                                         mktID=si.id, flag=guessFlag)
            else:
                logging.info('Inserting map for %s to (%s,%s)',
                             si.name, mdlIndustry.name, guessFlag)
                mdl.dbCursor.execute(
                    """INSERT INTO classification_market_map
                   (model_ref_id, market_ref_id, flag_as_guessed)
                   VALUES(:mdlID, :mktID, :flag)""",
                    mdlID=mdlIndustry.id,
                    mktID=si.id, flag=guessFlag)
    
def main():
    usage = "usage: %prog [options] config-file source-gics-date|all model-classification model-classification-date"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("-n", action="store_true",
                             default=False, dest="testOnly",
                             help="don't change the database")
    (options_, args_) = cmdlineParser.parse_args()
    if len(args_) != 4:
        cmdlineParser.error("Incorrect number of arguments")
    Utilities.processDefaultCommandLine(options_, cmdlineParser)
    
    configFile_ = open(args_[0])
    config_ = configparser.ConfigParser()
    config_.read_file(configFile_)
    configFile_.close()
    
    connections_ = Connections.createConnections(config_)
    mkt = connections_.marketDB
    mdl = connections_.modelDB

    if args_[1] == 'all':
        mktClas = mkt.getClassificationFamily('INDUSTRIES')
        mktGICS = [m for m in mkt.getClassificationFamilyMembers(mktClas)
                   if m.name == 'GICS'][0]
        mkt.dbCursor.execute("""SELECT from_dt FROM classification_revision
           WHERE member_id=:mem_id""", mem_id=mktGICS.id)
        gicsSourceDates = sorted([i[0].date() for i
                                  in mkt.dbCursor.fetchall()])
    else:
        gicsSourceDates = [Utilities.parseISODate(args_[1])]
    gicsMemberName = args_[2]
    mdlMemberDate = Utilities.parseISODate(args_[3])
    if gicsMemberName in ('GICSIndustryGroups', 'GICSCustom-AU', 'GICSCustom-AU2',
                          'GICSCustom-CA', 'GICSCustom-CA2', 'GICSCustom-CA3','GICSCustom-CA4', 'GICSCustom-CN', 'GICSCustom-TW','GICSCustom-CN2','GICSCustom-CN2b',
                          'GICSCustom-SubInd', 'GICSCustom-GB4'):
        level = 2
    elif gicsMemberName in ('GICSIndustries', 'GICSCustom-JP', 'GICSCustom-US', 'GICSCustom-NA4',
                            'GICSCustom-GB', 'GICSIndustries-Gold', 'GICSCustom-NA',
                            'GICSCustom-NoOE', 'GICSCustom-NoMortgageREITs', 'GICSCustom-EM',
                            'GICSCustom-EM2', 'GICSCustom-EU', 'GICSCustom-NoMortgageREITs2018'):
        level = 1
    else:
        msg = 'Invalid GICS-based classification name: %s' % gicsMemberName
        logging.error(msg)
        raise KeyError(msg)
    
    mdlClas = mdl.getMdlClassificationFamily('INDUSTRIES')
    mdlGICS = [m for m in mdl.getMdlClassificationFamilyMembers(mdlClas)
               if m.name == gicsMemberName][0]
    mdlRevision = mdl.getMdlClassificationMemberRevision(
        mdlGICS, mdlMemberDate)
    mdlIndustries = mdl.getMdlClassificationMemberLeaves(
        mdlGICS, mdlMemberDate)
    mdlIndustriesMap = dict([(i.description, i) for i in mdlIndustries])
    mdlLeafIDs = set([i.id for i in mdlIndustries])
    mdlIDIndustryMap = dict([(i.id, i) for i in mdlIndustries])
    mdlMemberDate = mdlRevision.from_dt
#    for k, v in guessMaps.items():
#         print k
    myGuessMap = guessMaps[(gicsMemberName, mdlMemberDate)]
#    for k, v in industryMaps.items():
#         print k, v
#    exit(1)
    
    
    industryMap = industryMaps[mdlMemberDate]
    
    mktClas = mkt.getClassificationFamily('INDUSTRIES')
    mktGICS = [m for m in mkt.getClassificationFamilyMembers(mktClas)
               if m.name == 'GICS'][0]
    for gicsSourceDate in gicsSourceDates:
        mktGICSRev = mkt.getClassificationMemberRevision(
            mktGICS, gicsSourceDate)
        logging.info('Processing Market GICS revision %d/%s',
                     mktGICSRev.id, mktGICSRev.from_dt)
        mktSubIndustries = getClassificationRevisionLeaves(
            mkt, mktGICSRev, level)
        mapRevision(mktSubIndustries, myGuessMap, industryMap,
                    mdlIndustriesMap, mdlIDIndustryMap, mdlLeafIDs, mdl)
    
    if options_.testOnly:
        logging.info('Reverting changes')
        mdl.revertChanges()
    else:
        logging.info('Committing changes')
        mdl.commitChanges()
    Connections.finalizeConnections(connections_)

# 
main()
