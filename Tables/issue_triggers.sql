/*
CREATE TABLE issue
(
	issue_id	CHAR(10) NOT NULL,
	from_dt		DATE NOT NULL,
	thru_dt		DATE NOT NULL,
	PRIMARY KEY(issue_id, from_dt, thru_dt),
	UNIQUE (issue_id, thru_dt)
) TABLESPACE gmdl_meta;
*/

create table issue_log as
select issue_id, from_dt, thru_dt, sysdate as action_dt, 'UPDATE' as action from issue where 1=2;


CREATE OR REPLACE TRIGGER t_issue_update
BEFORE UPDATE ON issue
FOR EACH ROW
BEGIN
    INSERT INTO issue_log (issue_id, from_dt, thru_dt, action_dt, action)
       VALUES (:old.issue_id, :old.from_dt, :old.thru_dt, SYSDATE, 'UPDATE');
END;

CREATE OR REPLACE TRIGGER t_issue_delete
BEFORE DELETE ON issue
FOR EACH ROW
BEGIN
    INSERT INTO issue_log (issue_id, from_dt, thru_dt, action_dt, action)
       VALUES (:old.issue_id, :old.from_dt, :old.thru_dt, SYSDATE, 'DELETE');
END;

