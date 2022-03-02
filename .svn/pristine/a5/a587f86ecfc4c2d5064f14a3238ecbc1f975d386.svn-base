DROP TABLE rmi_covariance CASCADE CONSTRAINTS;
CREATE TABLE rmi_covariance
(
	rms_id		INT NOT NULL,
	dt		DATE NOT NULL,
	sub_factor1_id	INT NOT NULL,
	sub_factor2_id	INT NOT NULL,
	value		NUMBER NOT NULL,
	PRIMARY KEY	(rms_id, dt, sub_factor1_id, sub_factor2_id)
) ORGANIZATION INDEX
  NOLOGGING
  PARTITION BY LIST (rms_id)
(
        PARTITION p_fcov_rm72 VALUES (-72) TABLESPACE gmdl_rms_main_rm72,
        PARTITION p_fcov_rm71 VALUES (-71) TABLESPACE gmdl_rms_main_rm71,
        PARTITION p_fcov_rm53 VALUES (-53) TABLESPACE gmdl_rms_main_rm5x,
        PARTITION p_fcov_rm52 VALUES (-52) TABLESPACE gmdl_rms_main_rm5x,
        PARTITION p_fcov_rm51 VALUES (-51) TABLESPACE gmdl_rms_main_rm5x,
        PARTITION p_fcov_rm43 VALUES (-43) TABLESPACE gmdl_rms_main_rm43,
        PARTITION p_fcov_rm42 VALUES (-42) TABLESPACE gmdl_rms_main_rm42,
        PARTITION p_fcov_rm41 VALUES (-41) TABLESPACE gmdl_rms_main_rm41,
        PARTITION p_fcov_rm23 VALUES (-23) TABLESPACE gmdl_rms_main_rm23,
        PARTITION p_fcov_rm22 VALUES (-22) TABLESPACE gmdl_rms_main_rm22,
        PARTITION p_fcov_rm21 VALUES (-21) TABLESPACE gmdl_rms_main_rm21,
        PARTITION p_fcov_rm13 VALUES (-13) TABLESPACE gmdl_rms_main_rm13,
        PARTITION p_fcov_rm12 VALUES (-12) TABLESPACE gmdl_rms_main_rm12,
        PARTITION p_fcov_rm11 VALUES (-11) TABLESPACE gmdl_rms_main_rm11,
        PARTITION p_fcov_rm10 VALUES (-10) TABLESPACE gmdl_rms_main_rm10,
        PARTITION p_fcov_rm06 VALUES (-6) TABLESPACE gmdl_rms_main_rm06,
        PARTITION p_fcov_rm05 VALUES (-5) TABLESPACE gmdl_rms_main_rm05,
        PARTITION p_fcov_rm04 VALUES (-4) TABLESPACE gmdl_rms_main_rm04,
        PARTITION p_fcov_rm03 VALUES (-3) TABLESPACE gmdl_rms_main_rm03,
        PARTITION p_fcov_rm02 VALUES (-2) TABLESPACE gmdl_rms_main_rm02,
        PARTITION p_fcov_rm01 VALUES (-1) TABLESPACE gmdl_rms_main_rm01,
        PARTITION p_fcov_r10 VALUES (10) TABLESPACE gmdl_rms_main_r10,
        PARTITION p_fcov_r11 VALUES (11) TABLESPACE gmdl_rms_main_r11,
        PARTITION p_fcov_r12 VALUES (12) TABLESPACE gmdl_rms_main_r12,
        PARTITION p_fcov_r13 VALUES (13) TABLESPACE gmdl_rms_main_r13,
        PARTITION p_fcov_r14 VALUES (14) TABLESPACE gmdl_rms_main_rm5x,
        PARTITION p_fcov_r15 VALUES (15) TABLESPACE gmdl_rms_main_rm5x,
        PARTITION p_fcov_r16 VALUES (16) TABLESPACE gmdl_rms_main_r16,
        PARTITION p_fcov_r17 VALUES (17) TABLESPACE gmdl_rms_main_r17,
        PARTITION p_fcov_r18 VALUES (18) TABLESPACE gmdl_rms_main_r18,
        PARTITION p_fcov_r19 VALUES (19) TABLESPACE gmdl_rms_main_r19,
        PARTITION p_fcov_r20 VALUES (20) TABLESPACE gmdl_rms_main_r20,
        PARTITION p_fcov_r21 VALUES (21) TABLESPACE gmdl_rms_main_r21,
        PARTITION p_fcov_r22 VALUES (22) TABLESPACE gmdl_rms_main_r22,
        PARTITION p_fcov_r25 VALUES (25) TABLESPACE gmdl_rms_main_r25,
        PARTITION p_fcov_r26 VALUES (26) TABLESPACE gmdl_rms_main_r26,
        PARTITION p_fcov_r27 VALUES (27) TABLESPACE gmdl_rms_main_r27,
        PARTITION p_fcov_r28 VALUES (28) TABLESPACE gmdl_rms_main_r28,
        PARTITION p_fcov_catchall VALUES (DEFAULT) TABLESPACE gmdl_rms_main
);
