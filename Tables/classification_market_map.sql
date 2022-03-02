DROP TABLE classification_market_map CASCADE CONSTRAINTS;
CREATE TABLE classification_market_map
(
  model_ref_id          INT NOT NULL,
  market_ref_id		INT NOT NULL,
  flag_as_guessed	CHAR(1) NOT NULL,
  CONSTRAINT pk_class_map PRIMARY KEY (model_ref_id, market_ref_id)
) TABLESPACE gmdl_classification;

ALTER TABLE classification_market_map ADD (
  CONSTRAINT fk_class_map_mdl_id FOREIGN KEY (model_ref_id) 
    REFERENCES classification_ref (id));

ALTER TABLE classification_market_map ADD (
  CONSTRAINT con_cmm_flag
  CHECK (flag_as_guessed IN ('Y', 'N')));
