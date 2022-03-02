DROP TABLE classification_revision CASCADE CONSTRAINTS;
CREATE TABLE classification_revision
(
  id               	INT NOT NULL,
  member_id		INT NOT NULL,
  from_dt		DATE NOT NULL,
  thru_dt		DATE NOT NULL,
  CONSTRAINT pk_class_revision PRIMARY KEY (from_dt, thru_dt, member_id)
) TABLESPACE gmdl_classification;

ALTER TABLE classification_revision ADD (
  CONSTRAINT con_class_revision_dt CHECK (from_dt < thru_dt));

ALTER TABLE classification_revision ADD (
  CONSTRAINT fk_class_member_id FOREIGN KEY (member_id) 
    REFERENCES classification_member (id));

ALTER TABLE classification_revision ADD CONSTRAINT
  class_revision_uniq_id UNIQUE (id);
