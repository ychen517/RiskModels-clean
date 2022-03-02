DROP TABLE sub_issue_fund_currency CASCADE CONSTRAINTS;

CREATE TABLE sub_issue_fund_currency
(
	sub_issue_id	CHAR(12) NOT NULL,
	item_code	INT NOT NULL,
	dt		DATE NOT NULL,
	value		NUMBER NOT NULL,
	currency_id	INT NOT NULL,
	eff_dt		DATE NOT NULL,
	eff_del_flag	CHAR(1) NOT NULL,
	rev_dt		DATE NOT NULL,
	rev_del_flag	CHAR(1) NOT NULL,
	CONSTRAINT pk_sub_issue_fund_currency
		PRIMARY KEY (sub_issue_id, item_code, dt, eff_dt, rev_dt)
		USING INDEX TABLESPACE gmdl_subissue_index
)
TABLESPACE gmdl_subissue;

ALTER TABLE sub_issue_fund_currency ADD (
  CONSTRAINT con_ad_fund_currency_rev CHECK (rev_del_flag IN ('Y', 'N')));

ALTER TABLE sub_issue_fund_currency ADD (
  CONSTRAINT con_ad_fund_currency_eff
  CHECK (eff_del_flag IN ('Y', 'N')));

ALTER TABLE sub_issue_fund_currency ADD (
  CONSTRAINT fk_ad_fund_currency_aid FOREIGN KEY (sub_issue_id)
    REFERENCES sub_issue (sub_id));

ALTER TABLE sub_issue_fund_currency ADD (
  CONSTRAINT fk_ad_fund_currency_ccy FOREIGN KEY (currency_id)
    REFERENCES marketdb_global.currency_ref (id));

CREATE INDEX idx_ad_fund_currency_rev ON sub_issue_fund_currency(rev_dt)
  TABLESPACE gmdl_subissue_index;
CREATE INDEX idx_sub_fund_curr ON sub_issue_fund_currency (item_code, eff_dt, sub_issue_id, rev_dt)
  TABLESPACE gmdl_subissue_index;

DROP VIEW sub_issue_fund_currency_active;
CREATE VIEW sub_issue_fund_currency_active AS (
  SELECT sub_issue_id, item_code, dt, value, currency_id, eff_dt, eff_del_flag
  FROM sub_issue_fund_currency t1
  WHERE rev_dt=(SELECT MAX(rev_dt) FROM sub_issue_fund_currency t2 
    WHERE t1.sub_issue_id=t2.sub_issue_id AND t1.item_code=t2.item_code
    AND t1.dt=t2.dt AND t1.eff_dt=t2.eff_dt) 
  AND rev_del_flag='N');
