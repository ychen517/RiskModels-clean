DROP TABLE rms_issue CASCADE CONSTRAINTS;
CREATE TABLE rms_issue
(
	rms_id		INT NOT NULL,
	issue_id	CHAR(10) NOT NULL,
	from_dt		DATE NOT NULL,
	thru_dt		DATE NOT NULL,
	CONSTRAINT PK_RMS_ISSUE PRIMARY KEY (rms_id, issue_id, from_dt)
		USING INDEX TABLESPACE gmdl_meta_index
) TABLESPACE gmdl_meta;
