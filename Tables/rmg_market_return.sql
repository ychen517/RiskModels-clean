--- Tables for market return ---
DROP TABLE rmg_market_return CASCADE CONSTRAINTS;
CREATE TABLE rmg_market_return
(
	dt		DATE NOT NULL,
	rmg_id		INT NOT NULL,
	value           NUMBER,
	CONSTRAINT PK_RMG_MARKET_RETURN PRIMARY KEY (dt, rmg_id)
		USING INDEX TABLESPACE gmdl_meta_index
) TABLESPACE gmdl_meta;

ALTER TABLE rmg_market_return ADD (
	CONSTRAINT FK_RMG_MARKET_RETURN FOREIGN KEY (rmg_id)
		REFERENCES risk_model_group(rmg_id));
