-- Table defining the currency belonging to each RMG at various times --
DROP TABLE rmg_currency CASCADE CONSTRAINTS;
CREATE TABLE rmg_currency
(
        rmg_id          INT NOT NULL,
        currency_code   VARCHAR2(3) NOT NULL,
        from_dt         DATE NOT NULL,
        thru_dt         DATE NOT NULL,
        PRIMARY KEY(rmg_id, currency_code)
) TABLESPACE gmdl_meta;

ALTER TABLE rmg_currency ADD (
	CONSTRAINT FK_RMG_CURRENCY FOREIGN KEY (rmg_id)
		REFERENCES risk_model_group(rmg_id));
