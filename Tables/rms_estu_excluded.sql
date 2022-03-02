DROP TABLE rms_estu_excluded CASCADE CONSTRAINTS;
CREATE TABLE rms_estu_excluded
(
	rms_id		INT NOT NULL,
	issue_id	CHAR(10) NOT NULL,
	change_dt	DATE NOT NULL,
        change_del_flag CHAR(1) NOT NULL,
	src_id      	INT NOT NULL,
	ref         	NVARCHAR2(512) NULL,
	rev_dt      	DATE NOT NULL,
        rev_del_flag    CHAR(1) NOT NULL,
	CONSTRAINT pk_rms_estu_excluded PRIMARY KEY(rms_id,issue_id,change_dt,rev_dt)
) TABLESPACE gmdl_meta;

DROP VIEW rms_estu_excl_active;
CREATE VIEW rms_estu_excl_active AS (
  SELECT rms_id, issue_id, change_dt, change_del_flag,
    src_id, ref
  FROM rms_estu_excluded t1
  WHERE rev_dt=(SELECT MAX(rev_dt) FROM rms_estu_excluded t2 
    WHERE t1.issue_id=t2.issue_id AND t1.rms_id = t2.rms_id
    AND t1.change_dt=t2.change_dt)
  AND rev_del_flag='N');

DROP VIEW rms_estu_excl_active_int;
CREATE VIEW rms_estu_excl_active_int ( rms_id, issue_id, from_dt, thru_dt )
AS
SELECT t1.rms_id, t1.issue_id, t1.change_dt from_dt, 
    NVL((SELECT MIN(t2.change_dt) 
        FROM rms_estu_excl_active t2 
        WHERE t1.issue_id=t2.issue_id
        AND t1.rms_id=t2.rms_id
        AND t1.change_dt<t2.change_dt),
        to_date('2999-12-31', 'YYYY-MM-DD')) thru_dt
FROM rms_estu_excl_active t1
WHERE change_del_flag = 'N';
