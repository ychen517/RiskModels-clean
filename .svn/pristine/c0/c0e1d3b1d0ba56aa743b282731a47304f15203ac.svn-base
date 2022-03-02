DROP TABLE risk_model_group CASCADE CONSTRAINTS;
CREATE TABLE risk_model_group
(
	rmg_id		INT NOT NULL,
	description	VARCHAR2(80),
	region_id	INT NOT NULL,
        mnemonic        VARCHAR2(2) NOT NULL,
        iso_code        VARCHAR2(3) NOT NULL,
        gmt_offset      NUMBER,
	PRIMARY KEY (rmg_id)
) TABLESPACE gmdl_meta;

ALTER TABLE risk_model_group ADD CONSTRAINT fk_region FOREIGN KEY (region_id)
  REFERENCES region(id);
