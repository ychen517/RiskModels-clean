DROP TABLE universal_client CASCADE CONSTRAINTS;
CREATE TABLE universal_client
(
    client_id   INT NOT NULL,
    client_name VARCHAR2(256) NOT NULL,
    include_equities    SMALLINT NOT NULL,
    include_commodities SMALLINT NOT NULL,
    from_dt     DATE NOT NULL,
    thru_dt     DATE NOT NULL,
    contact     VARCHAR2(256),
    CONSTRAINT pk_uni_client
               PRIMARY KEY (client_id)
) TABLESPACE gmdl_meta;


DROP TABLE universal_portfolio CASCADE CONSTRAINTS;
CREATE TABLE universal_portfolio
(
  rms_id        INT NOT NULL,
  client_id     INT NOT NULL,
  id_string     VARCHAR2(1000) NOT NULL,
  proxy_id      VARCHAR2(1000),
  additional_ids VARCHAR2(1000),
  axioma_data_id        INT,
  issue_id      CHAR(10),
  status        VARCHAR2(512),
  local         SMALLINT,
  client_comment varchar(1000),
  change_dt	DATE NOT NULL,
  change_del_flag	CHAR(1) NOT NULL,
  src_id	INT NOT NULL,
  rev_dt	DATE NOT NULL,
  rev_del_flag	CHAR(1) NOT NULL,
  CONSTRAINT pk_uni_portfolio
             PRIMARY KEY (rms_id, client_id, change_dt, rev_dt, id_string)
             USING INDEX TABLESPACE gmdl_meta_index
) TABLESPACE gmdl_meta;

ALTER TABLE universal_portfolio ADD(
  CONSTRAINT fk_uni_client_id FOREIGN KEY (client_id)
             REFERENCES universal_client(client_id)
);

--ALTER TABLE universal_portfolio ADD(
--  CONSTRAINT fk_uni_issue_id FOREIGN KEY (issue_id)
--             REFERENCES issue(issue_id)
--);

DROP VIEW uni_portfolio_active;
CREATE VIEW uni_portfolio_active AS (
       SELECT rms_id, client_id, id_string, proxy_id, additional_ids, axioma_data_id, issue_id, status, local, client_comment,
              src_id, change_dt, change_del_flag
  FROM universal_portfolio t1
  WHERE rev_dt=(SELECT MAX(rev_dt) FROM universal_portfolio t2 
    WHERE t1.rms_id=t2.rms_id AND t1.client_id=t2.client_id
    AND t1.id_string=t2.id_string AND t1.change_dt=t2.change_dt)
  AND rev_del_flag='N')
;

DROP VIEW uni_portfolio_active_int;
CREATE VIEW uni_portfolio_active_int AS (
	SELECT rms_id, client_id, id_string, proxy_id, additional_ids, axioma_data_id, issue_id, status, local, client_comment, src_id, CHANGE_DT from_dt, 
         NVL((SELECT MIN(t2.change_dt) 
    FROM uni_portfolio_active t2 WHERE t1.rms_id=t2.rms_id
         AND t1.client_id=t2.client_id
         AND t1.id_string=t2.id_string
         AND NVL(t1.proxy_id,'(null)')=NVL(t2.proxy_id,'(null)')
         and t1.change_dt<t2.change_dt), 
         to_date('9999-12-31', 'YYYY-MM-DD')) thru_dt
 FROM uni_portfolio_active t1
 where change_del_flag = 'N'
)
;

DROP VIEW uni_portfolio_int;
CREATE VIEW uni_portfolio_int AS (
       SELECT rms_id, client_id, id_string, proxy_id, additional_ids, axioma_data_id, issue_id, status, local, client_comment, src_id,
    change_dt from_dt,
            NVL((SELECT MIN(t2.change_dt) 
            FROM universal_portfolio  t2 
            WHERE t1.rms_id=t2.rms_id
            AND t1.client_id=t2.client_id
            AND t1.id_string=t2.id_string
            AND NVL(t1.proxy_id,'(null)')=NVL(t2.proxy_id,'(null)')
            AND t1.change_dt<t2.change_dt  
            AND t2.rev_dt >= t1.rev_dt ), 
    to_date('9999-12-31', 'YYYY-MM-DD')) thru_dt,
    rev_dt trans_from_dt,
    NVL((SELECT MIN(t2.rev_dt) 
        FROM universal_portfolio t2 
        WHERE t1.rms_id=t2.rms_id
        AND t1.client_id=t2.client_id
        AND t1.id_string=t2.id_string
        AND NVL(t1.proxy_id,'(null)')=NVL(t2.proxy_id,'(null)')
        AND t1.change_dt=t2.change_dt
        AND t2.rev_dt > t1.rev_dt), 
    to_date('9999-12-31', 'YYYY-MM-DD')) trans_thru_dt
  FROM universal_portfolio t1 WHERE change_del_flag='N' 
   AND rev_del_flag='N')
 ;
