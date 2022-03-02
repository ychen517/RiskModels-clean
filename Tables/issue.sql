-- Table defining Axioma IDs and their validity --
DROP TABLE issue CASCADE CONSTRAINTS;
CREATE TABLE issue
(
	issue_id	CHAR(10) NOT NULL,
	from_dt		DATE NOT NULL,
	thru_dt		DATE NOT NULL,
	PRIMARY KEY(issue_id, from_dt, thru_dt),
	UNIQUE (issue_id, thru_dt)
) TABLESPACE gmdl_meta;
