DROP TABLE classification_member CASCADE CONSTRAINTS;
CREATE TABLE classification_member
(
  id               	INT NOT NULL,
  name			VARCHAR2(32),
  description		VARCHAR2(512),
  family_id		INT NOT NULL,
  CONSTRAINT pk_class_member PRIMARY KEY (id)
) TABLESPACE gmdl_classification;

ALTER TABLE classification_member ADD (
  CONSTRAINT fk1_classification_member FOREIGN KEY (family_id) 
    REFERENCES classification_family (id));
