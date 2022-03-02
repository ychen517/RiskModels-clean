DROP TABLE amp_industry_return CASCADE CONSTRAINTS;
CREATE TABLE amp_industry_return
(
        dt                   DATE NOT NULL,
        mdl_port_member_id   INT NOT NULL,
        revision_id          INT NOT NULL,    
        ref_id               INT NOT NULL,
        value                NUMBER,
        CONSTRAINT PK_AMP_INDUSTRY_RETURN PRIMARY KEY (dt, mdl_port_member_id, revision_id, ref_id)
                USING INDEX TABLESPACE gmdl_meta_index
) TABLESPACE gmdl_meta;
