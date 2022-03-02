CREATE TABLE
    NEST
    (
        ID NUMBER(10) NOT NULL,
        NAME VARCHAR2(30) NOT NULL,
        PRIMARY KEY (ID)
    );
COMMENT ON TABLE NEST
IS
    'This lists nested regressions';
COMMENT ON COLUMN NEST.NAME
IS
    'Human-readable name for nested regression';
