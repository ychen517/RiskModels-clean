DROP TABLE sub_issue CASCADE CONSTRAINTS;
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
