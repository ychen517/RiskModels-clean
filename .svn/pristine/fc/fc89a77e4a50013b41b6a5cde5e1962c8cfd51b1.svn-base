/*
CREATE TABLE sub_issue
(
	issue_id	CHAR(10) NOT NULL,
	from_dt		DATE NOT NULL,
	thru_dt		DATE NOT NULL,
	sub_id		CHAR(12) NOT NULL,
	rmg_id		INT NOT NULL,
	PRIMARY KEY (issue_id, from_dt, thru_dt, rmg_id),
	UNIQUE (issue_id, thru_dt, rmg_id),
	UNIQUE (sub_id),
	CONSTRAINT uq_sub_issue_sid UNIQUE (sub_id, rmg_id)
) TABLESPACE gmdl_meta;
*/

create table sub_issue_log as
select issue_id, from_dt, thru_dt, sub_id, rmg_id, sysdate as action_dt, 'UPDATE' as action from sub_issue where 1=2;


CREATE OR REPLACE TRIGGER t_sub_issue_update
BEFORE UPDATE ON sub_issue
FOR EACH ROW
BEGIN
    INSERT INTO sub_issue_log (issue_id, from_dt, thru_dt, sub_id, rmg_id,action_dt, action)
       VALUES (:old.issue_id, :old.from_dt, :old.thru_dt, :old.sub_id, :old.rmg_id, SYSDATE, 'UPDATE');
END;

CREATE OR REPLACE TRIGGER t_sub_issue_delete
BEFORE DELETE ON sub_issue
FOR EACH ROW
BEGIN
    INSERT INTO sub_issue_log (issue_id, from_dt, thru_dt, sub_id, rmg_id, action_dt, action)
       VALUES (:old.issue_id, :old.from_dt, :old.thru_dt, :old.sub_id, :old.rmg_id,  SYSDATE, 'DELETE');
END;

