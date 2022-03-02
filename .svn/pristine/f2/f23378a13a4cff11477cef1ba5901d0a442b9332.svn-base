DROP MATERIALIZED view rmg_calendar;
CREATE MATERIALIZED VIEW rmg_calendar
TABLESPACE GMDL_META
build immediate
AS SELECT dt, rmg_id, seq sequence
  FROM marketdb_global.meta_trading_calendar_active cal
  JOIN modeldb_global.risk_model_group rmg ON cal.ISO_CTRY_CODE=rmg.MNEMONIC;

CREATE INDEX idx_rmg_calendar
  ON rmg_calendar (dt, rmg_id)
  TABLESPACE gmdl_meta_index;