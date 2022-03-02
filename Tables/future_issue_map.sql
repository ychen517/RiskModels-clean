-- Table defining the map between MarketDB --
-- and ModelDB issue IDs --
DROP TABLE future_issue_map CASCADE CONSTRAINTS;
CREATE TABLE future_issue_map
(
	modeldb_id	CHAR(10) NOT NULL,
	marketdb_id	CHAR(10) NOT NULL,
	from_dt		DATE NOT NULL,
	thru_dt		DATE NOT NULL,
	distribute  CHAR NOT NULL,
	PRIMARY KEY (modeldb_id, from_dt, thru_dt),
	UNIQUE (marketdb_id, thru_dt)
) TABLESPACE gmdl_meta;

CREATE INDEX idx_future_issue_map_mdl ON future_issue_map (modeldb_id)
TABLESPACE gmdl_meta;

CREATE TABLE
    future_issue_map_log AS
SELECT
    modeldb_id,
    marketdb_id,
    from_dt,
    thru_dt,
    distribute,
    SYSDATE  AS action_dt,
    'DELETE' AS action
FROM
    future_issue_map
WHERE
    1=2;

    
CREATE TRIGGER T_FUT_ISSUE_MAP_DELETE
BEFORE DELETE ON future_issue_map
FOR EACH ROW
BEGIN
    INSERT INTO future_issue_map_log (modeldb_id, marketdb_id, from_dt, thru_dt, distribute, action_dt, action)
       VALUES (:old.modeldb_id, :old.marketdb_id, :old.from_dt, :old.thru_dt, :old.distribute, SYSDATE, 'DELETE')
END
/

--/
CREATE TRIGGER T_FUT_ISSUE_MAP_UPDATE
BEFORE UPDATE ON future_issue_map
FOR EACH ROW
BEGIN
    INSERT INTO future_issue_map_log (modeldb_id, marketdb_id, from_dt, thru_dt, distribute, action_dt, action)
       VALUES (:old.modeldb_id, :old.marketdb_id, :old.from_dt, :old.thru_dt, :old.distribute, SYSDATE, 'UPDATE')
END
/
