-----------------------------------------------
-- Create Expression infrastructure
-----------------------------------------------

-- Create sequence for expression IDs with all defaults
CREATE SEQUENCE EXPRESSION_SEQ;

-- Create tables
CREATE TABLE UNARY_OPERATOR (
        NAME VARCHAR2(10) PRIMARY KEY, 
        DESCRIPTION VARCHAR2(200)
);
CREATE TABLE UNARY_EXPRESSION (
        ID INTEGER PRIMARY KEY,
        OPERATOR_NAME VARCHAR2(10) NOT NULL REFERENCES UNARY_OPERATOR(NAME), 
        ARGUMENT1_EXPRESSION_ID INTEGER NOT NULL
);
CREATE TABLE BINARY_OPERATOR (
        NAME VARCHAR2(10) PRIMARY KEY, 
        DESCRIPTION VARCHAR2(200)
);
CREATE TABLE BINARY_EXPRESSION (
        ID INTEGER PRIMARY KEY,
        OPERATOR_NAME VARCHAR2(10) NOT NULL REFERENCES BINARY_OPERATOR(NAME),
        ARGUMENT1_EXPRESSION_ID INTEGER NOT NULL, 
        ARGUMENT2_EXPRESSION_ID INTEGER NOT NULL
);
CREATE TABLE EXPRESSION (
        ID INTEGER PRIMARY KEY,
        CONSTANT BINARY_DOUBLE, 
        UNARY_EXPRESSION_ID INTEGER REFERENCES UNARY_EXPRESSION(ID),
        BINARY_EXPRESSION_ID INTEGER REFERENCES BINARY_EXPRESSION(ID), 
        VARIABLE VARCHAR2(10) UNIQUE,
        DESCRIPTION VARCHAR2(100)
);
ALTER TABLE UNARY_EXPRESSION ADD FOREIGN KEY (ARGUMENT1_EXPRESSION_ID) REFERENCES EXPRESSION(ID);
ALTER TABLE BINARY_EXPRESSION ADD FOREIGN KEY (ARGUMENT1_EXPRESSION_ID) REFERENCES EXPRESSION(ID);
ALTER TABLE BINARY_EXPRESSION ADD FOREIGN KEY (ARGUMENT2_EXPRESSION_ID) REFERENCES EXPRESSION(ID);

-- Populate tables
INSERT INTO UNARY_OPERATOR(NAME, DESCRIPTION) 
select 'cos', 'cos(x)' from dual
union all
select 'delta', 'x == 0 ? 1 : 0' from dual
union all
select 'exp', 'exp(x)' from dual
union all
select 'sin', 'sin(x)' from dual
union all
select 'uminus', '-x' from dual
;
INSERT INTO BINARY_OPERATOR(NAME, DESCRIPTION) 
select 'minus', 'x - y' from dual
union all
select 'plus', 'x + y' from dual
union all
select 'times', 'x * y' from dual
union all
select 'max', 'max(x,y)' from dual
union all 
select 'min', 'min(x,y)' from dual
;
