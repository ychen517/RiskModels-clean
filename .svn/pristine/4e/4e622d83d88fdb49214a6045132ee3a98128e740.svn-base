-- Table defining groups belonging to each model --
DROP TABLE rmg_model_map CASCADE CONSTRAINTS;
CREATE TABLE rmg_model_map
(
        rms_id          INT NOT NULL,
        rmg_id          INT NOT NULL,
        from_dt         DATE NOT NULL,
        thru_dt         DATE NOT NULL,
        fade_dt         DATE NOT NULL,
        full_dt         DATE NULL,
        PRIMARY KEY(rms_id, rmg_id)
) TABLESPACE gmdl_meta;

ALTER TABLE rmg_model_map ADD (
	CONSTRAINT FK_RMG_MODEL_MAP FOREIGN KEY (rmg_id)
		REFERENCES risk_model_group(rmg_id));
