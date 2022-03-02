DROP TABLE SUB_ISSUE_ESTI_NUMBER CASCADE CONSTRAINTS

CREATE TABLE SUB_ISSUE_ESTI_NUMBER
(
	sub_issue_id	CHAR(12) NOT NULL,
	item_code	INT NOT NULL,
	dt		DATE NOT NULL,
	value		NUMBER NOT NULL,
	eff_dt		DATE NOT NULL,
	eff_del_flag	CHAR(1) NOT NULL,
	rev_dt		DATE NOT NULL,
	rev_del_flag	CHAR(1) NOT NULL,
	CONSTRAINT PK_SUB_ISSUE_ESTI_NUMBER
		PRIMARY KEY (sub_issue_id, item_code, dt, eff_dt, rev_dt)
		USING INDEX TABLESPACE gmdl_subissue_index
)
TABLESPACE gmdl_subissue

ALTER TABLE SUB_ISSUE_ESTI_NUMBER ADD (
  CONSTRAINT CON_SI_ESTI_NUMBER_REV CHECK (rev_del_flag IN ('Y', 'N')))

ALTER TABLE SUB_ISSUE_ESTI_NUMBER ADD (
  CONSTRAINT CON_SI_ESTI_NUMBER_EFF
  CHECK (eff_del_flag IN ('Y', 'N')))

ALTER TABLE SUB_ISSUE_ESTI_NUMBER ADD (
  CONSTRAINT FK_SI_ESTI_NUMBER_AID FOREIGN KEY (sub_issue_id)
    REFERENCES sub_issue (sub_id))

CREATE INDEX IDX_SI_ESTI_NUMBER_REV ON SUB_ISSUE_ESTI_NUMBER(rev_dt)
  TABLESPACE gmdl_subissue_index

CREATE INDEX IDX_SUB_ESTI_NUMBER ON SUB_ISSUE_ESTI_NUMBER(item_code, eff_dt, sub_issue_id, rev_dt)
  TABLESPACE gmdl_subissue_index

DROP VIEW SUB_ISSUE_ESTI_NUMBER_ACTIVE

CREATE VIEW SUB_ISSUE_ESTI_NUMBER_ACTIVE AS (
  SELECT sub_issue_id, item_code, dt, value, eff_dt, eff_del_flag
  FROM SUB_ISSUE_ESTI_NUMBER t1
  WHERE rev_dt=(SELECT MAX(rev_dt) FROM SUB_ISSUE_ESTI_NUMBER t2 
    WHERE t1.sub_issue_id=t2.sub_issue_id AND t1.item_code=t2.item_code
    AND t1.dt=t2.dt AND t1.eff_dt=t2.eff_dt) 
  AND rev_del_flag='N')
