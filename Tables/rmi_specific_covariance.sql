--- specific covariances of the risk model instances ---
DROP TABLE rmi_specific_covariance CASCADE CONSTRAINTS;
CREATE TABLE rmi_specific_covariance
(
  rms_id        INT NOT NULL,
  dt            DATE NOT NULL,
  sub_issue1_id CHAR(12) NOT NULL,
  sub_issue2_id CHAR(12) NOT NULL,
  value         NUMBER NOT NULL,
  PRIMARY KEY (rms_id, dt, sub_issue1_id, sub_issue2_id)
) ORGANIZATION INDEX
  PARTITION BY LIST (rms_id)
(
        PARTITION p_speccov_r12 VALUES(12) TABLESPACE gmdl_rms_main_r12,
        PARTITION p_speccov_r17 VALUES(17) TABLESPACE gmdl_rms_main_r17,
        PARTITION p_speccov_r25 VALUES(25) TABLESPACE gmdl_rms_main_r25,
        PARTITION p_speccov_r26 VALUES(26) TABLESPACE gmdl_rms_main_r26,
        PARTITION p_speccov_r27 VALUES(27) TABLESPACE gmdl_rms_main_r27,
        PARTITION p_speccov_r28 VALUES(28) TABLESPACE gmdl_rms_main_r28,
        PARTITION p_speccov_r29 VALUES(29) TABLESPACE gmdl_rms_main_r29,
        PARTITION p_speccov_r30 VALUES(30) TABLESPACE gmdl_rms_main_r30,
        PARTITION p_speccov_r33 VALUES(33) TABLESPACE gmdl_rms_main_r33,
        PARTITION p_speccov_r34 VALUES(34) TABLESPACE gmdl_rms_main_r34,
        PARTITION p_speccov_r39 VALUES(39) TABLESPACE gmdl_rms_main_r39,
        PARTITION p_speccov_r40 VALUES(40) TABLESPACE gmdl_rms_main_r40,
        PARTITION p_speccov_r41 VALUES(41) TABLESPACE gmdl_rms_main_r41,
        PARTITION p_speccov_r42 VALUES(42) TABLESPACE gmdl_rms_main_r42,
        PARTITION p_speccov_r43 VALUES(43) TABLESPACE gmdl_rms_main_r43,
        PARTITION p_speccov_r44 VALUES(44) TABLESPACE gmdl_rms_main_r44,
        PARTITION p_speccov_r45 VALUES(45) TABLESPACE gmdl_rms_main_r45,
        PARTITION p_speccov_r46 VALUES(46) TABLESPACE gmdl_rms_main_r46,
        PARTITION p_speccov_r47 VALUES(47) TABLESPACE gmdl_rms_main_r47,
        PARTITION p_speccov_r48 VALUES(48) TABLESPACE gmdl_rms_main_r48,
        PARTITION p_speccov_r49 VALUES(49) TABLESPACE gmdl_rms_main_r49,
        PARTITION p_speccov_r50 VALUES(50) TABLESPACE gmdl_rms_main_r50,
        PARTITION p_speccov_r51 VALUES(51) TABLESPACE gmdl_rms_main_r51,
        PARTITION p_speccov_r52 VALUES(52) TABLESPACE gmdl_rms_main_r52,
        PARTITION p_speccov_r53 VALUES(53) TABLESPACE gmdl_rms_main_r53,
        PARTITION p_speccov_r54 VALUES(54) TABLESPACE gmdl_rms_main_r54,
        PARTITION p_speccov_r55 VALUES(55) TABLESPACE gmdl_rms_main_r55,
        PARTITION p_speccov_r58 VALUES(58) TABLESPACE gmdl_rms_main_r58,
        PARTITION p_speccov_r59 VALUES(59) TABLESPACE gmdl_rms_main_r59,
        PARTITION p_speccov_r60 VALUES(60) TABLESPACE gmdl_rms_main_r60,
        PARTITION p_speccov_r61 VALUES(61) TABLESPACE gmdl_rms_main_r61,
        PARTITION p_speccov_r62 VALUES(62) TABLESPACE gmdl_rms_main_r62,
        PARTITION p_speccov_r63 VALUES(63) TABLESPACE gmdl_rms_main_r63,
        PARTITION p_speccov_r64 VALUES(64) TABLESPACE gmdl_rms_main_r64,
        PARTITION p_speccov_catchall VALUES (DEFAULT) TABLESPACE gmdl_rms_main
);
