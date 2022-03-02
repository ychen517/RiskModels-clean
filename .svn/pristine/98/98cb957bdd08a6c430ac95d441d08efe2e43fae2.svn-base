DROP TABLE rms_factor CASCADE CONSTRAINTS;
CREATE TABLE rms_factor
(
	rms_id		INT NOT NULL,
	factor_id	INT NOT NULL,
        from_dt         DATE NOT NULL,
        thru_dt         DATE NOT NULL,
	PRIMARY KEY (rms_id, factor_id)
) TABLESPACE gmdl_meta;

ALTER TABLE rms_factor ADD (
	CONSTRAINT FK_RMS_FACTOR FOREIGN KEY (factor_id)
		REFERENCES factor(factor_id));
