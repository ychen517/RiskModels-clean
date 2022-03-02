-- Table defining the map between MarketDB --
-- and ModelDB issue IDs --
DROP TABLE issue_map CASCADE CONSTRAINTS;
CREATE TABLE issue_map
(
	modeldb_id	CHAR(10) NOT NULL,
	marketdb_id	CHAR(10) NOT NULL,
	from_dt		DATE NOT NULL,
	thru_dt		DATE NOT NULL,
	PRIMARY KEY (modeldb_id, from_dt, thru_dt),
	UNIQUE (marketdb_id, thru_dt)
) TABLESPACE gmdl_meta;

CREATE INDEX idx_issue_map_mdl ON issue_map (modeldb_id)
TABLESPACE gmdl_meta;
