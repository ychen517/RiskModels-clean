DROP TABLE classification_ref CASCADE CONSTRAINTS;
CREATE TABLE classification_ref
(
  id               	INT NOT NULL,
  revision_id		INT NOT NULL,
  name			VARCHAR(64),
  description		VARCHAR2(128),
  is_root		CHAR(1) NOT NULL,
  is_leaf		CHAR(1) NOT NULL,
  CONSTRAINT pk_class_ref PRIMARY KEY (id)
) TABLESPACE gmdl_classification;

ALTER TABLE classification_ref ADD (
  CONSTRAINT check_leaf CHECK (is_leaf IN ('Y','N')));

ALTER TABLE classification_ref ADD (
  CONSTRAINT check_root CHECK (is_root IN ('Y','N')));

ALTER TABLE classification_ref ADD (
  CONSTRAINT fk_class_rev_id FOREIGN KEY (revision_id) 
    REFERENCES classification_revision (id));
