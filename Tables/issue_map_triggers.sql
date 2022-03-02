create table issue_map_log as 
select modeldb_id, marketdb_id, from_dt, thru_dt, sysdate as action_dt, 'UPDATE' as action from issue_map where 1=2;

CREATE OR REPLACE TRIGGER t_issue_map_update
BEFORE UPDATE ON issue_map
FOR EACH ROW
BEGIN
    INSERT INTO issue_map_log (modeldb_id, marketdb_id, from_dt, thru_dt, action_dt, action)
       VALUES (:old.modeldb_id, :old.marketdb_id, :old.from_dt, :old.thru_dt, SYSDATE, 'UPDATE');
END;

CREATE OR REPLACE TRIGGER t_issue_map_delete
BEFORE DELETE ON issue_map
FOR EACH ROW
BEGIN
    INSERT INTO issue_map_log (modeldb_id, marketdb_id, from_dt, thru_dt, action_dt, action)
       VALUES (:old.modeldb_id, :old.marketdb_id, :old.from_dt, :old.thru_dt, SYSDATE, 'DELETE');
END;



