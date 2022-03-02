DROP TABLE SUB_ISSUE_ESTI_CURRENCY CASCADE CONSTRAINTS;

CREATE TABLE SUB_ISSUE_ESTI_CURRENCY
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
	CONSTRAINT PK_SUB_ISSUE_ESTI_CURRENCY
		PRIMARY KEY (sub_issue_id, item_code, dt, eff_dt, rev_dt)
		USING INDEX TABLESPACE gmdl_subissue_index
)

ALTER TABLE SUB_ISSUE_ESTI_CURRENCY ADD (
  CONSTRAINT CON_AD_ESTI_CURRENCY_REV CHECK (rev_del_flag IN ('Y', 'N')))

ALTER TABLE SUB_ISSUE_ESTI_CURRENCY ADD (
  CONSTRAINT CON_AD_ESTI_CURRENCY_EFF
  CHECK (eff_del_flag IN ('Y', 'N')))

ALTER TABLE SUB_ISSUE_ESTI_CURRENCY ADD (
  CONSTRAINT FK_AD_ESTI_CURRENCY_AID FOREIGN KEY (sub_issue_id)
    REFERENCES sub_issue (sub_id))

ALTER TABLE SUB_ISSUE_ESTI_CURRENCY ADD (
  CONSTRAINT FK_AD_ESTI_CURRENCY_CCY FOREIGN KEY (currency_id)
    REFERENCES marketdb_global.currency_ref (id))

CREATE INDEX IDX_AD_ESTI_CURRENCY_REV ON SUB_ISSUE_ESTI_CURRENCY(rev_dt)
  TABLESPACE gmdl_subissue_index

CREATE INDEX IDX_SUB_ESTI_CURR ON SUB_ISSUE_ESTI_CURRENCY (item_code, eff_dt, sub_issue_id, rev_dt)
  TABLESPACE gmdl_subissue_index

DROP VIEW SUB_ISSUE_ESTI_CURRENCY_ACTIVE

CREATE VIEW SUB_ISSUE_ESTI_CURRENCY_ACTIVE AS (
  SELECT sub_issue_id, item_code, dt, value, currency_id, eff_dt, eff_del_flag
  FROM SUB_ISSUE_ESTI_CURRENCY t1
  WHERE rev_dt=(SELECT MAX(rev_dt) FROM SUB_ISSUE_ESTI_CURRENCY t2 
    WHERE t1.sub_issue_id=t2.sub_issue_id AND t1.item_code=t2.item_code
    AND t1.dt=t2.dt AND t1.eff_dt=t2.eff_dt) 
  AND rev_del_flag='N')
