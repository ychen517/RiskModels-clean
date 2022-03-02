DROP TABLE currency_risk_free_rate CASCADE CONSTRAINTS;
CREATE TABLE currency_risk_free_rate
(
	currency_code	VARCHAR2(5) NOT NULL,
	dt		DATE NOT NULL,
	value		NUMBER NOT NULL,
	cumulative	NUMBER,
	CONSTRAINT pk_cur_rf_rate PRIMARY KEY (currency_code, dt)
		USING INDEX TABLESPACE gmdl_meta_index
) NOLOGGING
  TABLESPACE gmdl_meta;
