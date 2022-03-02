DROP TABLE sub_issue_data CASCADE CONSTRAINTS;
CREATE TABLE sub_issue_data
(
	sub_issue_id	CHAR(12) NOT NULL,
	dt		DATE NOT NULL,
	rev_dt		DATE NOT NULL,
	rev_del_flag	CHAR(1) NOT NULL,
	tdv		NUMBER,
	tso		NUMBER,
	ucp		NUMBER,
	price_marker	INT,
	currency_id	INT NOT NULL,
        rmg_id          INT NOT NULL,
	CONSTRAINT pk_sub_issue_data
		PRIMARY KEY (sub_issue_id, dt, rev_dt),
	CONSTRAINT fk_sub_issue_data_sid FOREIGN KEY (sub_issue_id)
		REFERENCES sub_issue(sub_id)
) ORGANIZATION INDEX
  PARTITION BY RANGE (dt) (
        PARTITION p_subissue_data_95 VALUES LESS THAN (to_date('1996-01-01', 'YYYY-MM-DD')) TABLESPACE gmdl_subissue_data,
        PARTITION p_subissue_data_96 VALUES LESS THAN (to_date('1997-01-01', 'YYYY-MM-DD')) TABLESPACE gmdl_subissue_data,
        PARTITION p_subissue_data_97 VALUES LESS THAN (to_date('1998-01-01', 'YYYY-MM-DD')) TABLESPACE gmdl_subissue_data,
        PARTITION p_subissue_data_98 VALUES LESS THAN (to_date('1999-01-01', 'YYYY-MM-DD')) TABLESPACE gmdl_subissue_data,
        PARTITION p_subissue_data_99 VALUES LESS THAN (to_date('2000-01-01', 'YYYY-MM-DD')) TABLESPACE gmdl_subissue_data,
        PARTITION p_subissue_data_00 VALUES LESS THAN (to_date('2001-01-01', 'YYYY-MM-DD')) TABLESPACE gmdl_subissue_data,
        PARTITION p_subissue_data_01 VALUES LESS THAN (to_date('2002-01-01', 'YYYY-MM-DD')) TABLESPACE gmdl_subissue_data,
        PARTITION p_subissue_data_02 VALUES LESS THAN (to_date('2003-01-01', 'YYYY-MM-DD')) TABLESPACE gmdl_subissue_data,
        PARTITION p_subissue_data_03 VALUES LESS THAN (to_date('2004-01-01', 'YYYY-MM-DD')) TABLESPACE gmdl_subissue_data,
        PARTITION p_subissue_data_04 VALUES LESS THAN (to_date('2005-01-01', 'YYYY-MM-DD')) TABLESPACE gmdl_subissue_data,
        PARTITION p_subissue_data_05 VALUES LESS THAN (to_date('2006-01-01', 'YYYY-MM-DD')) TABLESPACE gmdl_subissue_data,
        PARTITION p_subissue_data_06 VALUES LESS THAN (to_date('2007-01-01', 'YYYY-MM-DD')) TABLESPACE gmdl_subissue_data,
        PARTITION p_subissue_data_07 VALUES LESS THAN (to_date('2008-01-01', 'YYYY-MM-DD')) TABLESPACE gmdl_subissue_data,
        PARTITION p_subissue_data_08 VALUES LESS THAN (to_date('2009-01-01', 'YYYY-MM-DD')) TABLESPACE gmdl_subissue_data,
        PARTITION p_subissue_data_catchall VALUES LESS THAN (MAXVALUE) TABLESPACE gmdl_subissue_data
);

ALTER TABLE sub_issue_data ADD (
  CONSTRAINT sub_issue_data_rev CHECK (rev_del_flag IN ('Y', 'N')));

DROP VIEW sub_issue_data_active;
CREATE VIEW sub_issue_data_active AS (
  SELECT sub_issue_id, dt, tdv, tso, ucp, price_marker, currency_id, rmg_id
  FROM sub_issue_data t1
  WHERE rev_dt=(SELECT MAX(rev_dt) FROM sub_issue_data t2 
    WHERE t1.sub_issue_id=t2.sub_issue_id AND t1.dt=t2.dt)
  AND rev_del_flag='N');
