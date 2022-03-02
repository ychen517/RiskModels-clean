-- Table containing assets which should not be included in extracted files --
DROP TABLE stoxx_product_exclude CASCADE CONSTRAINTS;
CREATE TABLE stoxx_product_exclude
(
	issue_id	CHAR(10) NOT NULL,
	from_dt		DATE NOT NULL,
	thru_dt		DATE NOT NULL,
	REF          	NVARCHAR2(128) NULL,
	add_dt		date null,
	src_id		int null,
	PRIMARY KEY(issue_id, from_dt, thru_dt),
	UNIQUE (issue_id, thru_dt)
) TABLESPACE gmdl_meta;
