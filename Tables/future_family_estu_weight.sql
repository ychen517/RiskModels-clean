CREATE TABLE
    FUTURE_FAMILY_ESTU_WEIGHT
    (
        FUTURE_FAMILY_ID NVARCHAR2(10) NOT NULL,
        WEIGHT_EXPRESSION_ID INTEGER NOT NULL,
        RMS_ID NUMBER(38) NOT NULL,
        NEST_ID NUMBER(10) NOT NULL,
        PRIMARY KEY (RMS_ID, FUTURE_FAMILY_ID),
        CONSTRAINT FUTURE_FAMILY_ESTU_WEIGHT_FK1 FOREIGN KEY (RMS_ID) REFERENCES RISK_MODEL_SERIE
        (SERIAL_ID),
        CONSTRAINT FUTURE_FAMILY_ESTU_WEIGHT_FK2 FOREIGN KEY (NEST_ID) REFERENCES NEST (ID)
    );