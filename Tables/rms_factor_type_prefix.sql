DROP TABLE rms_factor_type_prefix CASCADE CONSTRAINTS;
CREATE TABLE rms_factor_type_prefix
(
	rms_id		INT NOT NULL,
	factor_type_id	INT NOT NULL,
        prefix          VARCHAR2(10),
	PRIMARY KEY (rms_id, factor_type_id)
) TABLESPACE gmdl_meta;

ALTER TABLE rms_factor_type_prefix ADD (
	CONSTRAINT FK_RMS_FACTOR_TYPE_PREFIX FOREIGN KEY (factor_type_id)
		REFERENCES factor_type(factor_type_id));
