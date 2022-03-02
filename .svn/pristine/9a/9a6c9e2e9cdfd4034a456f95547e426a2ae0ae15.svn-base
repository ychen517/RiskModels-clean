DROP TABLE ca_merger_survivor CASCADE CONSTRAINTS;
CREATE TABLE ca_merger_survivor
(
	dt		DATE NOT NULL,
	ca_sequence	INTEGER NOT NULL,
	modeldb_id	CHAR(10) NOT NULL,
	new_marketdb_id	CHAR(10) NOT NULL,
	old_marketdb_id	CHAR(10) NOT NULL,
	share_ratio	NUMBER,
	cash_payment	NUMBER,
	currency_id	INT,
	ref		VARCHAR2(128),
	rmg_id          INT DEFAULT 1 NOT NULL, --default is 1 for US model/existing records
	PRIMARY KEY (dt, modeldb_id)
) TABLESPACE gmdl_meta;
