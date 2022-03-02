DROP TABLE returns_timing_adjustment CASCADE CONSTRAINTS;
CREATE TABLE returns_timing_adjustment
(
	timing_id	INT NOT NULL,
	rmg_id		INT NOT NULL,
	dt		DATE NOT NULL,
	value		NUMBER NOT NULL,
	PRIMARY KEY (timing_id, dt, rmg_id)
) ORGANIZATION INDEX
  NOLOGGING
  TABLESPACE gmdl_meta;
