CREATE TABLE
    NEST_IO
    (
        RMS_ID NUMBER(38) NOT NULL,
        NEST_ID NUMBER(10) NOT NULL,
        FACTOR_ID NUMBER(38) NOT NULL,
        INPUT_OUTPUT CHAR(1) NOT NULL,
        PRIMARY KEY (RMS_ID, NEST_ID, FACTOR_ID, INPUT_OUTPUT),
        FOREIGN KEY (RMS_ID) REFERENCES RISK_MODEL_SERIE (SERIAL_ID),
        FOREIGN KEY (NEST_ID) REFERENCES NEST (ID),
        CHECK (INPUT_OUTPUT IN ('I',
                                'O'))
    );
COMMENT ON TABLE NEST_IO
IS
    'Stores factors required by a nested a regression, and those output by it.';
COMMENT ON COLUMN NEST_IO.INPUT_OUTPUT
IS
    'I: Factor must be available before regression is carried; O: Factor will estimated in regression.'
    ;
