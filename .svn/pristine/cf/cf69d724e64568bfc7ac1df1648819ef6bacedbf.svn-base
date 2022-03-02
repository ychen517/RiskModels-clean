DROP TABLE rms_factor_descriptor CASCADE CONSTRAINTS;
CREATE TABLE rms_factor_descriptor
(
	rms_id		INT NOT NULL,
	factor_id	INT NOT NULL,
        descriptor_id   INT NOT NULL,
        scale           NUMBER,
	PRIMARY KEY (rms_id, factor_id, descriptor_id)
) TABLESPACE gmdl_meta;

ALTER TABLE rms_factor_descriptor ADD (
	CONSTRAINT FK_RMS_FACTOR_DESCR0 FOREIGN KEY (descriptor_id)
		REFERENCES descriptor(descriptor_id));
ALTER TABLE rms_factor_descriptor ADD (
	CONSTRAINT FK_RMS_FACTOR_DESCR1 FOREIGN KEY (factor_id)
		REFERENCES factor(factor_id));
