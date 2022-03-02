DROP TABLE descriptor CASCADE CONSTRAINTS;
CREATE TABLE descriptor
(
	descriptor_id	INT NOT NULL,
	name		VARCHAR2(64),
	description	VARCHAR2(80),
	PRIMARY KEY (descriptor_id)
) TABLESPACE gmdl_meta;
