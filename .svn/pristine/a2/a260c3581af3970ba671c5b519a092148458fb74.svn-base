-- Table defining each RMG's status as DM/EM/FM --
DROP TABLE rmg_dev_status CASCADE CONSTRAINTS;
CREATE TABLE rmg_dev_status
(
        rmg_id          INT NOT NULL,
        developed       INT NOT NULL,
	emerging	INT NOT NULL,
	frontier	INT NOT NULL,
        from_dt         DATE NOT NULL,
        thru_dt         DATE NOT NULL,
        PRIMARY KEY(rmg_id, from_dt));

ALTER TABLE rmg_dev_status ADD (
	CONSTRAINT FK_RMG_DEV_STATUS FOREIGN KEY (rmg_id)
		REFERENCES risk_model_group(rmg_id));
