-- Table mapping composite families to the models that support them --
DROP TABLE composite_family_model_map CASCADE CONSTRAINTS;
CREATE TABLE composite_family_model_map
(
        model_id             INT NOT NULL,
        composite_family_id  INT NOT NULL,
        PRIMARY KEY(model_id, composite_family_id)
) TABLESPACE gmdl_meta;

ALTER TABLE composite_family_model_map ADD (
	CONSTRAINT FK_COMPOSITE_FAMILY_MODEL_MAP FOREIGN KEY (model_id)
		REFERENCES risk_model(model_id));
