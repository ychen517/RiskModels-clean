DROP TABLE risk_model_instance CASCADE CONSTRAINTS;
CREATE TABLE risk_model_instance
(
	rms_id		INT NOT NULL,
	dt		DATE NOT NULL,
	has_exposures	INT NOT NULL,
	has_returns	INT NOT NULL,
	has_risks	INT NOT NULL,
	update_dt	DATE NOT NULL,
	is_final	INT NOT NULL,
	PRIMARY KEY (rms_id, dt)
) TABLESPACE gmdl_rms_main;
