DROP TABLE classification_family CASCADE CONSTRAINTS;
CREATE TABLE classification_family
(
  id               	INT NOT NULL,
  name			VARCHAR2(32),
  description		VARCHAR2(512),
  CONSTRAINT pk_class_family PRIMARY KEY (id)
) TABLESPACE gmdl_classification;

INSERT INTO classification_family VALUES
(1, 'INDUSTRIES', 'INDUSTRY CLASSIFICATIONS');
INSERT INTO classification_family VALUES
(2, 'REGIONS', 'Country and region classifications');