DROP TABLE rmi_factor_exposure CASCADE CONSTRAINTS;
CREATE TABLE rmi_factor_exposure
(
	rms_id		INT NOT NULL,
	dt		DATE NOT NULL,
	sub_factor_id	INT NOT NULL,
	sub_issue_id	CHAR(12) NOT NULL,
	value		NUMBER NOT NULL,
	PRIMARY KEY (rms_id, dt, sub_factor_id, sub_issue_id)
) ORGANIZATION INDEX
  NOLOGGING
  PARTITION BY LIST (rms_id)
(
        PARTITION p_exposure_rm72 VALUES (-72) TABLESPACE gmdl_rms_exposures_rm72,
        PARTITION p_exposure_rm71 VALUES (-71) TABLESPACE gmdl_rms_exposures_rm71,
        PARTITION p_exposure_rm43 VALUES (-43) TABLESPACE gmdl_rms_exposures_rm43,
        PARTITION p_exposure_rm42 VALUES (-42) TABLESPACE gmdl_rms_exposures_rm42,
        PARTITION p_exposure_rm41 VALUES (-41) TABLESPACE gmdl_rms_exposures_rm41,
        PARTITION p_exposure_rm23 VALUES (-23) TABLESPACE gmdl_rms_exposures_rm23,
        PARTITION p_exposure_rm22 VALUES (-22) TABLESPACE gmdl_rms_exposures_rm22,
        PARTITION p_exposure_rm21 VALUES (-21) TABLESPACE gmdl_rms_exposures_rm21,
        PARTITION p_exposure_rm13 VALUES (-13) TABLESPACE gmdl_rms_exposures_rm13,
        PARTITION p_exposure_rm12 VALUES (-12) TABLESPACE gmdl_rms_exposures_rm12,
        PARTITION p_exposure_rm11 VALUES (-11) TABLESPACE gmdl_rms_exposures_rm11,
        PARTITION p_exposure_rm10 VALUES (-10) TABLESPACE gmdl_rms_exposures_rm10,
        PARTITION p_exposure_rm06 VALUES (-6) TABLESPACE gmdl_rms_exposures_rm06,
        PARTITION p_exposure_rm05 VALUES (-5) TABLESPACE gmdl_rms_exposures_rm05,
        PARTITION p_exposure_rm04 VALUES (-4) TABLESPACE gmdl_rms_exposures_rm04,
        PARTITION p_exposure_rm03 VALUES (-3) TABLESPACE gmdl_rms_exposures_rm03,
        PARTITION p_exposure_rm02 VALUES (-2) TABLESPACE gmdl_rms_exposures_rm02,
        PARTITION p_exposure_rm01 VALUES (-1) TABLESPACE gmdl_rms_exposures_rm01,
        PARTITION p_exposure_r10 VALUES (10) TABLESPACE gmdl_rms_exposures_r10,
        PARTITION p_exposure_r11 VALUES (11) TABLESPACE gmdl_rms_exposures_r11,
        PARTITION p_exposure_r12 VALUES (12) TABLESPACE gmdl_rms_exposures_r12,
        PARTITION p_exposure_r13 VALUES (13) TABLESPACE gmdl_rms_exposures_r13,
        PARTITION p_exposure_r16 VALUES (16) TABLESPACE gmdl_rms_exposures_r16,
        PARTITION p_exposure_r17 VALUES (17) TABLESPACE gmdl_rms_exposures_r17,
        PARTITION p_exposure_r18 VALUES (18) TABLESPACE gmdl_rms_exposures_r18,
        PARTITION p_exposure_r19 VALUES (19) TABLESPACE gmdl_rms_exposures_r19,
        PARTITION p_exposure_r20 VALUES (20) TABLESPACE gmdl_rms_exposures_r20,
        PARTITION p_exposure_r21 VALUES (21) TABLESPACE gmdl_rms_exposures_r21,
        PARTITION p_exposure_r22 VALUES (22) TABLESPACE gmdl_rms_exposures_r22,
        PARTITION p_exposure_r25 VALUES (25) TABLESPACE gmdl_rms_exposures_r25,
        PARTITION p_exposure_r26 VALUES (26) TABLESPACE gmdl_rms_exposures_r26,
        PARTITION p_exposure_r27 VALUES (27) TABLESPACE gmdl_rms_exposures_r27,
        PARTITION p_exposure_r28 VALUES (28) TABLESPACE gmdl_rms_exposures_r28,
        PARTITION p_exposure_catchall VALUES (DEFAULT) TABLESPACE gmdl_rms_exposures
);
