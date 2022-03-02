DROP TABLE classification_dim_hier CASCADE CONSTRAINTS;
CREATE TABLE classification_dim_hier
(
  parent_classification_id    INT NOT NULL,
  child_classification_id     INT NOT NULL,
  weight                      NUMBER NOT NULL,
  CONSTRAINT PK_CLASS_DIM_HIER PRIMARY KEY (parent_classification_id, child_classification_id)
) TABLESPACE gmdl_classification;
