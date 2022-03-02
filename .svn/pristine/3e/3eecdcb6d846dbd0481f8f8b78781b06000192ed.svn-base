DROP TABLE classification_constituent CASCADE CONSTRAINTS;
CREATE TABLE classification_constituent
(
  classification_id        INT NOT NULL,
  issue_id                 CHAR(10) NOT NULL,
  weight                   NUMBER NOT NULL,
  change_dt                DATE NOT NULL,
  change_del_flag          CHAR(1) NOT NULL,
  src_id                   INT NOT NULL,
  ref                      VARCHAR2(512),
  rev_dt                   DATE NOT NULL,
  rev_del_flag             CHAR(1) NOT NULL,
  CONSTRAINT pk_class_constituent PRIMARY KEY (issue_id, classification_id,
	change_dt, rev_dt)
) ORGANIZATION INDEX
  TABLESPACE gmdl_classification;

ALTER TABLE classification_constituent ADD (
  CONSTRAINT con_class_constituent_rev CHECK (rev_del_flag IN ('Y', 'N')));

ALTER TABLE classification_constituent ADD (
  CONSTRAINT con_class_constituent_ch
  CHECK (change_del_flag IN ('Y', 'N')));

--ALTER TABLE classification_constituent ADD (
--  CONSTRAINT fk_class_constituent_iid FOREIGN KEY (issue_id)
--    REFERENCES issue (issue_id));

--ALTER TABLE classification_constituent ADD (
--  CONSTRAINT fk_class_constituent_src FOREIGN KEY (src_id)
--    REFERENCES marketdb_global.meta_source (src_id));

ALTER TABLE classification_constituent ADD (
  CONSTRAINT fk_class_constituent_id FOREIGN KEY (classification_id)
    REFERENCES classification_ref (id));

DROP VIEW classification_const_active;
CREATE VIEW classification_const_active AS (
  SELECT classification_id, issue_id, weight, change_dt, change_del_flag,
    src_id, ref, r1.revision_id
  FROM classification_constituent t1 JOIN classification_ref r1
    ON t1.classification_id=r1.id
  WHERE rev_dt=(SELECT MAX(rev_dt) FROM classification_constituent t2 
    JOIN classification_ref r2 ON t2.classification_id=r2.id
    WHERE t1.issue_id=t2.issue_id AND t1.change_dt=t2.change_dt
    AND r1.revision_id=r2.revision_id)
  AND rev_del_flag='N');
