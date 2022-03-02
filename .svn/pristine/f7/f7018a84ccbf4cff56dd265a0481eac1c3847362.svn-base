-- Views similar to what exists on marketdb_global
CREATE VIEW MODELDB_GLOBAL.CLASSIFICATION_ACTIVE ( CLASSIFICATION_ID, ISSUE_ID, CHANGE_DT, CHANGE_DEL_FLAG, SRC_ID, REF, NAME, CLASSIFICATION_TYPE_NAME )
AS
SELECT t1.classification_id, t1.issue_id, t1.change_dt, t1.change_del_flag,
    t1.src_id, t1.ref, cref.name,
    cm.name classification_type_name
  FROM modeldb_global.classification_const_active t1, modeldb_global.classification_ref cref,
  modeldb_global.classification_revision crev, modeldb_global.classification_member cm 
  WHERE crev.MEMBER_ID = cm.id
  and cref.REVISION_ID = crev.id
  and t1.CLASSIFICATION_ID = cref.id;

CREATE VIEW MODELDB_GLOBAL.CLASSIFICATION_ACTIVE_INT ( ISSUE_ID, CLASSIFICATION_ID, CLASSIFICATION_TYPE_NAME, NAME, FROM_DT, THRU_DT )
AS
select t1.issue_id, t1.classification_id, t1.classification_type_name, 
    t1.name, t1.change_dt from_dt, 
    NVL((SELECT MIN(t2.change_dt) 
        FROM modeldb_global.CLASSIFICATION_ACTIVE t2 
        WHERE t1.issue_id=t2.issue_id
        and t1.change_dt<t2.change_dt
        and t2.classification_type_name = t1.classification_type_name ), 
        to_date('9999-12-31', 'YYYY-MM-DD')) thru_dt
from modeldb_global.classification_active t1
where change_del_flag = 'N';
