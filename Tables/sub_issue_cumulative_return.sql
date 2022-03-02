DROP TABLE sub_issue_cumulative_return CASCADE CONSTRAINTS;
CREATE TABLE sub_issue_cumulative_return
(
	sub_issue_id	CHAR(12) NOT NULL,
	dt		DATE NOT NULL,
	rev_dt		DATE NOT NULL,
	rev_del_flag	CHAR(1) NOT NULL,
	value		NUMBER,
	rmg_value	NUMBER,
        rmg_id          INT NOT NULL,
	CONSTRAINT pk_sub_issue_cum_return
		PRIMARY KEY (sub_issue_id, dt, rmg_id, rev_dt),
	CONSTRAINT fk_sub_issue_cum_return_sid FOREIGN KEY (sub_issue_id)
		REFERENCES sub_issue(sub_id)
) ORGANIZATION INDEX
  PARTITION BY RANGE (dt) (
        PARTITION p_subissue_cum_95 VALUES LESS THAN (to_date('1996-01-01', 'YYYY-MM-DD')) TABLESPACE gmdl_subissue_cumret,
        PARTITION p_subissue_cum_96 VALUES LESS THAN (to_date('1997-01-01', 'YYYY-MM-DD')) TABLESPACE gmdl_subissue_cumret,
        PARTITION p_subissue_cum_97 VALUES LESS THAN (to_date('1998-01-01', 'YYYY-MM-DD')) TABLESPACE gmdl_subissue_cumret,
        PARTITION p_subissue_cum_98 VALUES LESS THAN (to_date('1999-01-01', 'YYYY-MM-DD')) TABLESPACE gmdl_subissue_cumret,
        PARTITION p_subissue_cum_99 VALUES LESS THAN (to_date('2000-01-01', 'YYYY-MM-DD')) TABLESPACE gmdl_subissue_cumret,
        PARTITION p_subissue_cum_00 VALUES LESS THAN (to_date('2001-01-01', 'YYYY-MM-DD')) TABLESPACE gmdl_subissue_cumret,
        PARTITION p_subissue_cum_01 VALUES LESS THAN (to_date('2002-01-01', 'YYYY-MM-DD')) TABLESPACE gmdl_subissue_cumret,
        PARTITION p_subissue_cum_02 VALUES LESS THAN (to_date('2003-01-01', 'YYYY-MM-DD')) TABLESPACE gmdl_subissue_cumret,
        PARTITION p_subissue_cum_03 VALUES LESS THAN (to_date('2004-01-01', 'YYYY-MM-DD')) TABLESPACE gmdl_subissue_cumret,
        PARTITION p_subissue_cum_04 VALUES LESS THAN (to_date('2005-01-01', 'YYYY-MM-DD')) TABLESPACE gmdl_subissue_cumret,
        PARTITION p_subissue_cum_05 VALUES LESS THAN (to_date('2006-01-01', 'YYYY-MM-DD')) TABLESPACE gmdl_subissue_cumret,
        PARTITION p_subissue_cum_06 VALUES LESS THAN (to_date('2007-01-01', 'YYYY-MM-DD')) TABLESPACE gmdl_subissue_cumret,
        PARTITION p_subissue_cum_07 VALUES LESS THAN (to_date('2008-01-01', 'YYYY-MM-DD')) TABLESPACE gmdl_subissue_cumret,
        PARTITION p_subissue_cum_08 VALUES LESS THAN (to_date('2009-01-01', 'YYYY-MM-DD')) TABLESPACE gmdl_subissue_cumret,
        PARTITION p_subissue_cum_catchall VALUES LESS THAN (MAXVALUE) TABLESPACE gmdl_subissue_cumret
);

ALTER TABLE sub_issue_cumulative_return ADD (
  CONSTRAINT sub_issue_cum_return_rev CHECK (rev_del_flag IN ('Y', 'N')));

DROP VIEW sub_issue_cum_return_active;
CREATE VIEW sub_issue_cum_return_active AS (
  SELECT sub_issue_id, dt, value, rmg_value, rmg_id
  FROM sub_issue_cumulative_return t1
  WHERE rev_dt=(SELECT MAX(rev_dt) FROM sub_issue_cumulative_return t2 
    WHERE t1.sub_issue_id=t2.sub_issue_id AND t1.dt=t2.dt)
  AND rev_del_flag='N');
