--- Temporary tables for ModelDB ---
DROP TABLE temp_issues CASCADE CONSTRAINTS;
CREATE GLOBAL TEMPORARY TABLE temp_issues
(
	id		CHAR(10),
	idx		INTEGER,
	PRIMARY KEY (id)
) ON COMMIT PRESERVE ROWS;

DROP TABLE temp_subfactors CASCADE CONSTRAINTS;
CREATE GLOBAL TEMPORARY TABLE temp_subfactors
(
	id		INTEGER,
	idx		INTEGER,
	PRIMARY KEY (id)
) ON COMMIT PRESERVE ROWS;

DROP TABLE temp_dates CASCADE CONSTRAINTS;
CREATE GLOBAL TEMPORARY TABLE temp_dates
(
	dt		DATE,
	idx		INTEGER,
	PRIMARY KEY (dt)
) ON COMMIT PRESERVE ROWS;