--- Historic beta of the risk model instances ---
DROP TABLE rmg_historic_beta CASCADE CONSTRAINTS;
CREATE TABLE rmg_historic_beta
(
	dt		DATE NOT NULL,
	sub_issue_id	CHAR(12) NOT NULL,
	value		NUMBER,
	new_value	NUMBER,
	fixed_value	NUMBER,
	CONSTRAINT pk_rmg_historic_beta
		PRIMARY KEY (dt, sub_issue_id)
) ORGANIZATION INDEX
  NOLOGGING
  TABLESPACE gmdl_subissue;

