--- Estimation estuerses of the risk model instances ---
DROP TABLE rmi_coverage_universe CASCADE CONSTRAINTS;
CREATE TABLE rmi_coverage_universe
(
	rms_id		INT NOT NULL,
	dt		DATE NOT NULL,
	sub_issue_id	CHAR(12) NOT NULL,
	PRIMARY KEY (rms_id, dt, sub_issue_id)
) ORGANIZATION INDEX
  NOLOGGING
  PARTITION BY LIST (rms_id)
(
        PARTITION p_covuniv_rm43 VALUES (-43) TABLESPACE gmdl_rms_main_rm43,
        PARTITION p_covuniv_rm42 VALUES (-42) TABLESPACE gmdl_rms_main_rm42,
        PARTITION p_covuniv_rm41 VALUES (-41) TABLESPACE gmdl_rms_main_rm41,
        PARTITION p_covuniv_rm23 VALUES (-23) TABLESPACE gmdl_rms_main_rm23,
        PARTITION p_covuniv_rm22 VALUES (-22) TABLESPACE gmdl_rms_main_rm22,
        PARTITION p_covuniv_rm21 VALUES (-21) TABLESPACE gmdl_rms_main_rm21,
        PARTITION p_covuniv_rm13 VALUES (-13) TABLESPACE gmdl_rms_main_rm13,
        PARTITION p_covuniv_rm12 VALUES (-12) TABLESPACE gmdl_rms_main_rm12,
        PARTITION p_covuniv_rm11 VALUES (-11) TABLESPACE gmdl_rms_main_rm11,
        PARTITION p_covuniv_rm10 VALUES (-10) TABLESPACE gmdl_rms_main_rm10,
        PARTITION p_covuniv_rm06 VALUES (-6) TABLESPACE gmdl_rms_main_rm06,
        PARTITION p_covuniv_rm05 VALUES (-5) TABLESPACE gmdl_rms_main_rm05,
        PARTITION p_covuniv_rm04 VALUES (-4) TABLESPACE gmdl_rms_main_rm04,
        PARTITION p_covuniv_rm03 VALUES (-3) TABLESPACE gmdl_rms_main_rm04,
        PARTITION p_covuniv_rm02 VALUES (-2) TABLESPACE gmdl_rms_main_rm04,
        PARTITION p_covuniv_rm01 VALUES (-1) TABLESPACE gmdl_rms_main_rm04,
        PARTITION p_covuniv_catchall VALUES (DEFAULT) TABLESPACE gmdl_rms_main
);
