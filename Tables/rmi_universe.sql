--- Universes of exposure matrices in the risk model instances ---
DROP TABLE rmi_universe CASCADE CONSTRAINTS;
CREATE TABLE rmi_universe
(
	rms_id		INT NOT NULL,
	dt		DATE NOT NULL,
	sub_issue_id	CHAR(12) NOT NULL,
        qualify         INT,
	PRIMARY KEY (rms_id, dt, sub_issue_id)
) ORGANIZATION INDEX
  NOLOGGING
  PARTITION BY LIST (rms_id)
(
        PARTITION p_univ_rm72 VALUES (-72) TABLESPACE gmdl_rms_main_rm72,
        PARTITION p_univ_rm71 VALUES (-71) TABLESPACE gmdl_rms_main_rm71,
        PARTITION p_univ_rm43 VALUES (-43) TABLESPACE gmdl_rms_main_rm43,
        PARTITION p_univ_rm42 VALUES (-42) TABLESPACE gmdl_rms_main_rm42,
        PARTITION p_univ_rm41 VALUES (-41) TABLESPACE gmdl_rms_main_rm41,
        PARTITION p_univ_rm23 VALUES (-23) TABLESPACE gmdl_rms_main_rm23,
        PARTITION p_univ_rm22 VALUES (-22) TABLESPACE gmdl_rms_main_rm22,
        PARTITION p_univ_rm21 VALUES (-21) TABLESPACE gmdl_rms_main_rm21,
        PARTITION p_univ_rm13 VALUES (-13) TABLESPACE gmdl_rms_main_rm13,
        PARTITION p_univ_rm12 VALUES (-12) TABLESPACE gmdl_rms_main_rm12,
        PARTITION p_univ_rm11 VALUES (-11) TABLESPACE gmdl_rms_main_rm11,
        PARTITION p_univ_rm10 VALUES (-10) TABLESPACE gmdl_rms_main_rm10,
        PARTITION p_univ_rm06 VALUES (-6) TABLESPACE gmdl_rms_main_rm06,
        PARTITION p_univ_rm05 VALUES (-5) TABLESPACE gmdl_rms_main_rm05,
        PARTITION p_univ_rm04 VALUES (-4) TABLESPACE gmdl_rms_main_rm04,
        PARTITION p_univ_rm03 VALUES (-3) TABLESPACE gmdl_rms_main_rm03,
        PARTITION p_univ_rm02 VALUES (-2) TABLESPACE gmdl_rms_main_rm02,
        PARTITION p_univ_rm01 VALUES (-1) TABLESPACE gmdl_rms_main_rm01,
        PARTITION p_univ_r10 VALUES (10) TABLESPACE gmdl_rms_main_r10,
        PARTITION p_univ_r11 VALUES (11) TABLESPACE gmdl_rms_main_r11,
        PARTITION p_univ_r12 VALUES (12) TABLESPACE gmdl_rms_main_r12,
        PARTITION p_univ_r13 VALUES (13) TABLESPACE gmdl_rms_main_r13,
        PARTITION p_univ_r16 VALUES (16) TABLESPACE gmdl_rms_main_r16,
        PARTITION p_univ_r17 VALUES (17) TABLESPACE gmdl_rms_main_r17,
        PARTITION p_univ_r18 VALUES (18) TABLESPACE gmdl_rms_main_r18,
        PARTITION p_univ_r19 VALUES (19) TABLESPACE gmdl_rms_main_r19,
        PARTITION p_univ_r20 VALUES (20) TABLESPACE gmdl_rms_main_r20,
        PARTITION p_univ_r21 VALUES (21) TABLESPACE gmdl_rms_main_r21,
        PARTITION p_univ_r22 VALUES (22) TABLESPACE gmdl_rms_main_r22,
        PARTITION p_univ_r25 VALUES (25) TABLESPACE gmdl_rms_main_r25,
        PARTITION p_univ_r26 VALUES (26) TABLESPACE gmdl_rms_main_r26,
        PARTITION p_univ_r27 VALUES (27) TABLESPACE gmdl_rms_main_r27,
        PARTITION p_univ_r28 VALUES (28) TABLESPACE gmdl_rms_main_r28,
        PARTITION p_univ_catchall VALUES (DEFAULT) TABLESPACE gmdl_rms_main
);

-- Index to speed up the query to get min date in rmi_universe for Asset Profile
CREATE INDEX idx_ru_sub_issue_id ON rmi_universe(sub_issue_id)
	TABLESPACE gmdl_subissue;