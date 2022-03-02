DROP TABLE risk_model_serie CASCADE CONSTRAINTS;
CREATE TABLE risk_model_serie
(
	serial_id	INT NOT NULL,
	rm_id		INT NOT NULL,
	revision	INT NOT NULL,
        from_dt         DATE DEFAULT '01-jan-1950' NOT NULL,
        thru_dt         DATE DEFAULT '31-dec-2999' NOT NULL,
        distribute      INT DEFAULT 0 NOT NULL,
        PRIMARY KEY (serial_id),
	UNIQUE (rm_id, revision)
) TABLESPACE gmdl_meta;

ALTER TABLE risk_model_serie
        ADD (CONSTRAINT con_rms_date_valid
                CHECK (from_dt < thru_dt))
;
