DROP TABLE content_freeze CASCADE CONSTRAINTS;

CREATE TABLE MODELDB_GLOBAL.CONTENT_FREEZE ( 
    RM_ID   NUMBER(38,0) NOT NULL,
    FROM_DT DATE NOT NULL,
    THRU_DT DATE NOT NULL,
    SRC_ID  NUMBER NOT NULL,
    REF     VARCHAR2(512) NULL,
    REV_DT  DATE NOT NULL,
    CONSTRAINT PK_CONTENT_FREEZE PRIMARY KEY(RM_ID,FROM_DT)
    NOT DEFERRABLE
     VALIDATE
)
GO
ALTER TABLE MODELDB_GLOBAL.CONTENT_FREEZE
    ADD ( CONSTRAINT CON_CONTENT_FREEZEG_DT
    CHECK (from_dt < thru_dt)
    NOT DEFERRABLE INITIALLY IMMEDIATE VALIDATE )
GO
ALTER TABLE MODELDB_GLOBAL.CONTENT_FREEZE
    ADD ( CONSTRAINT FK_CF_MODEL_MNEMONIC
    FOREIGN KEY(RM_ID)
    REFERENCES MODELDB_GLOBAL.RISK_MODEL(MODEL_ID)
    NOT DEFERRABLE INITIALLY IMMEDIATE VALIDATE )
GO