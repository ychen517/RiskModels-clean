CREATE TABLE risk_model_family_map
(
	family_id	int references RISK_MODEL_FAMILY(FAMILY_ID),
	model_id	int references RISK_MODEL(MODEL_ID),
	from_dt		date NOT NULL,
	thru_dt		date NOT NULL,
	PRIMARY KEY (family_id, model_id, from_dt)
) TABLESPACE gmdl_meta;
