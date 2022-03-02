DROP TABLE sub_factor CASCADE CONSTRAINTS;
CREATE TABLE sub_factor
(
	factor_id	INT NOT NULL,
	from_dt		DATE NOT NULL,
	thru_dt		DATE NOT NULL,
	sub_id		INT NOT NULL,
	PRIMARY KEY (sub_id, from_dt, thru_dt)
) TABLESPACE gmdl_meta;

ALTER TABLE sub_factor ADD CONSTRAINT fk_sub_factor FOREIGN KEY (factor_id)
  REFERENCES factor(factor_id);
