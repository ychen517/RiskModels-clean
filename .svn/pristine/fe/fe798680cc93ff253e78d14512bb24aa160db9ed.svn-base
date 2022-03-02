/*
CREATE TABLE product_exclude
(
	issue_id	CHAR(10) NOT NULL,
	from_dt		DATE NOT NULL,
	thru_dt		DATE NOT NULL,
	REF          	NVARCHAR2(128) NULL,
	add_dt		date null,
	src_id		int null
	PRIMARY KEY(issue_id, from_dt, thru_dt),
	UNIQUE (issue_id, thru_dt)
) TABLESPACE gmdl_meta;
*/

create table product_exclude_log as
select issue_id, from_dt, thru_dt, REF, add_dt, src_id, sysdate as action_dt, 'UPDATE' as action from product_exclude  where 1=2;


CREATE OR REPLACE TRIGGER t_product_exclude_update
BEFORE UPDATE ON product_exclude
FOR EACH ROW
BEGIN
    INSERT INTO product_exclude_log (issue_id, from_dt, thru_dt, REF, add_dt, src_id, action_dt, action)
       VALUES (:old.issue_id, :old.from_dt, :old.thru_dt, :old.REF, :old.add_dt, :old.src_id, SYSDATE, 'UPDATE');
END;

CREATE OR REPLACE TRIGGER t_product_exclude_delete
BEFORE DELETE ON product_exclude
FOR EACH ROW
BEGIN
    INSERT INTO product_exclude_log (issue_id, from_dt, thru_dt, REF, add_dt, src_id, action_dt, action)
       VALUES (:old.issue_id, :old.from_dt, :old.thru_dt, :old.REF, :old.add_dt, :old.src_id, SYSDATE, 'DELETE');
END;

