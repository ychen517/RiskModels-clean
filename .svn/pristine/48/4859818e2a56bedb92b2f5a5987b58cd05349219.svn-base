-- Table mapping futures families to the models that support them --
DROP TABLE future_family_model_map CASCADE CONSTRAINTS;
CREATE TABLE future_family_model_map
(
        model_id             INT NOT NULL,
        future_family_id  INT NOT NULL,
        PRIMARY KEY(model_id, future_family_id)
) TABLESPACE gmdl_meta;

ALTER TABLE future_family_model_map ADD (
	CONSTRAINT FK_FUTURE_FAMILY_MODEL_MAP FOREIGN KEY (model_id)
		REFERENCES risk_model(model_id));
