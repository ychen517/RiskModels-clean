DROP TABLE rms_specific_return CASCADE CONSTRAINTS;
CREATE TABLE rms_specific_return
(
	rms_id		INT NOT NULL,
	sub_issue_id	CHAR(12) NOT NULL,
	dt		DATE NOT NULL,
	value		NUMBER NOT NULL,
	PRIMARY KEY (rms_id, dt, sub_issue_id)
) ORGANIZATION INDEX
  NOLOGGING
  PARTITION BY LIST (rms_id)
(
        PARTITION p_specret_rm72 VALUES (-72) TABLESPACE gmdl_rms_returns_rm72,
        PARTITION p_specret_rm71 VALUES (-71) TABLESPACE gmdl_rms_returns_rm71,
        PARTITION p_specret_rm43 VALUES (-43) TABLESPACE gmdl_rms_returns_rm43,
        PARTITION p_specret_rm42 VALUES (-42) TABLESPACE gmdl_rms_returns_rm42,
        PARTITION p_specret_rm41 VALUES (-41) TABLESPACE gmdl_rms_returns_rm41,
        PARTITION p_specret_rm23 VALUES (-23) TABLESPACE gmdl_rms_returns_rm23,
        PARTITION p_specret_rm22 VALUES (-22) TABLESPACE gmdl_rms_returns_rm22,
        PARTITION p_specret_rm21 VALUES (-21) TABLESPACE gmdl_rms_returns_rm21,
        PARTITION p_specret_rm13 VALUES (-13) TABLESPACE gmdl_rms_returns_rm13,
        PARTITION p_specret_rm12 VALUES (-12) TABLESPACE gmdl_rms_returns_rm12,
        PARTITION p_specret_rm11 VALUES (-11) TABLESPACE gmdl_rms_returns_rm11,
        PARTITION p_specret_rm10 VALUES (-10) TABLESPACE gmdl_rms_returns_rm10,
        PARTITION p_specret_rm06 VALUES (-6) TABLESPACE gmdl_rms_returns_rm06,
        PARTITION p_specret_rm05 VALUES (-5) TABLESPACE gmdl_rms_returns_rm05,
        PARTITION p_specret_rm04 VALUES (-4) TABLESPACE gmdl_rms_returns_rm04,
        PARTITION p_specret_rm03 VALUES (-3) TABLESPACE gmdl_rms_returns_rm03,
        PARTITION p_specret_rm02 VALUES (-2) TABLESPACE gmdl_rms_returns_rm02,
        PARTITION p_specret_rm01 VALUES (-1) TABLESPACE gmdl_rms_returns_rm01,
        PARTITION p_specret_r10 VALUES (10) TABLESPACE gmdl_rms_returns_r10,
        PARTITION p_specret_r11 VALUES (11) TABLESPACE gmdl_rms_returns_r11,
        PARTITION p_specret_r12 VALUES (12) TABLESPACE gmdl_rms_returns_r12,
        PARTITION p_specret_r13 VALUES (13) TABLESPACE gmdl_rms_returns_r13,
        PARTITION p_specret_r16 VALUES (16) TABLESPACE gmdl_rms_returns_r16,
        PARTITION p_specret_r17 VALUES (17) TABLESPACE gmdl_rms_returns_r17,
        PARTITION p_specret_r18 VALUES (18) TABLESPACE gmdl_rms_returns_r18,
        PARTITION p_specret_r19 VALUES (19) TABLESPACE gmdl_rms_returns_r19,
        PARTITION p_specret_r20 VALUES (20) TABLESPACE gmdl_rms_returns_r20,
        PARTITION p_specret_r21 VALUES (21) TABLESPACE gmdl_rms_returns_r21,
        PARTITION p_specret_r22 VALUES (22) TABLESPACE gmdl_rms_returns_r22,
        PARTITION p_specret_r25 VALUES (25) TABLESPACE gmdl_rms_returns_r25,
        PARTITION p_specret_r26 VALUES (26) TABLESPACE gmdl_rms_returns_r26,
        PARTITION p_specret_r27 VALUES (27) TABLESPACE gmdl_rms_returns_r27,
        PARTITION p_specret_r28 VALUES (28) TABLESPACE gmdl_rms_returns_r28,
        PARTITION p_specret_catchall VALUES (DEFAULT) TABLESPACE gmdl_rms_returns
);
