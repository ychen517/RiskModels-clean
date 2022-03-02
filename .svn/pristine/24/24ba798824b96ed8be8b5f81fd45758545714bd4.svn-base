create table rms_issue_log as 
select rms_id, issue_id, from_dt, thru_dt, sysdate as action_dt, 'UPDATE' as action from rms_issue where 1=2;

CREATE OR REPLACE TRIGGER t_rms_issue_update
BEFORE UPDATE ON rms_issue
FOR EACH ROW
BEGIN
    INSERT INTO rms_issue_log (rms_id, issue_id, from_dt, thru_dt, action_dt, action)
       VALUES (:old.rms_id, :old.issue_id, :old.from_dt, :old.thru_dt, SYSDATE, 'UPDATE');
END;

CREATE OR REPLACE TRIGGER t_rms_issue_delete
BEFORE DELETE ON rms_issue
FOR EACH ROW
BEGIN
    INSERT INTO rms_issue_log (rms_id, issue_id, from_dt, thru_dt, action_dt, action)
       VALUES (:old.rms_id, :old.issue_id, :old.from_dt, :old.thru_dt, SYSDATE, 'DELETE');
END;




