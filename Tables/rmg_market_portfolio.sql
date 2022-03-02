--- Table for market portfolios ---
DROP TABLE rmg_market_portfolio CASCADE CONSTRAINTS;
CREATE TABLE rmg_market_portfolio
(
	dt		DATE NOT NULL,
	rmg_id		INT NOT NULL,
	sub_issue_id	CHAR(12) NOT NULL,
	value           NUMBER,
	CONSTRAINT PK_RMG_MARKET_PORT PRIMARY KEY (dt, rmg_id, sub_issue_id)
) ORGANIZATION INDEX
  TABLESPACE gmdl_subissue;

ALTER TABLE rmg_market_portfolio ADD (
	CONSTRAINT FK_RMG_MARKET_PORT FOREIGN KEY (rmg_id)
		REFERENCES risk_model_group(rmg_id));
