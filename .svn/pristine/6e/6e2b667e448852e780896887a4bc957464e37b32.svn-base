create table csholt_product_exclude_log as
select issue_id, from_dt, thru_dt, REF, add_dt, src_id, sysdate as action_dt, 'UPDATE' as action from csholt_product_exclude  where 1=2;


CREATE OR REPLACE TRIGGER t_cs_product_exclude_update
BEFORE UPDATE ON csholt_product_exclude
FOR EACH ROW
BEGIN
    INSERT INTO csholt_product_exclude_log (issue_id, from_dt, thru_dt, REF, add_dt, src_id, action_dt, action)
       VALUES (:old.issue_id, :old.from_dt, :old.thru_dt, :old.REF, :old.add_dt, :old.src_id, SYSDATE, 'UPDATE');
END;

CREATE OR REPLACE TRIGGER t_cs_product_exclude_delete
BEFORE DELETE ON csholt_product_exclude
FOR EACH ROW
BEGIN
    INSERT INTO csholt_product_exclude_log (issue_id, from_dt, thru_dt, REF, add_dt, src_id, action_dt, action)
       VALUES (:old.issue_id, :old.from_dt, :old.thru_dt, :old.REF, :old.add_dt, :old.src_id, SYSDATE, 'DELETE');
END;

