-- Table defining the map between MarketDB --
-- and ModelDB issue IDs --
DROP TABLE currency_instrument_map CASCADE CONSTRAINTS;
CREATE TABLE currency_instrument_map
(
    currency_code   CHAR(3) NOT NULL,
    instrument_name varCHAR(11) NOT NULL,
    from_dt     DATE not null,
    thru_dt     DATE not null,
    PRIMARY KEY (currency_code, from_dt, thru_dt),
    UNIQUE (currency_code, thru_dt)
);
