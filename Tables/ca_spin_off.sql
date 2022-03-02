DROP TABLE ca_spin_off CASCADE CONSTRAINTS;
CREATE TABLE ca_spin_off
(
	dt		DATE NOT NULL,
	ca_sequence	INTEGER NOT NULL,
	parent_id	CHAR(10) NOT NULL,
	child_id	CHAR(10) NOT NULL,
	share_ratio	NUMBER,
	implied_div	NUMBER,
	currency_id	INT,
	ref		VARCHAR2(128),
	rmg_id          INT DEFAULT 1 NOT NULL, --default is 1 for US model/existing records
	PRIMARY KEY (dt, parent_id, child_id)
) TABLESPACE gmdl_meta;
