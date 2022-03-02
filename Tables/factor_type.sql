DROP TABLE factor_type CASCADE CONSTRAINTS;
CREATE TABLE factor_type
(
	factor_type_id	INT NOT NULL,
	name		VARCHAR2(64),
	description	VARCHAR2(80),
	PRIMARY KEY (factor_type_id)
) TABLESPACE gmdl_meta;
