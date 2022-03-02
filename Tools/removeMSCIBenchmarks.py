# Remove all revisions and constituents of the MSCI benchmarks
# from the research database.
# To be run after each sync of the research database with production.

import cx_Oracle
import datetime

def deleteMSCIBenchmarksPriorTo(dbCursor, date):
    # Get constituent and revision counts
    dbCursor.execute("""SELECT count(*) FROM index_constituent
       WHERE revision_id IN (SELECT id FROM index_revision
          WHERE index_id IN (SELECT mem.id FROM index_member mem
             JOIN index_family fam ON mem.family_id=fam.id
             WHERE fam.name like '%MSCI%')
          AND dt < :date_arg)""", date_arg=date)
    numConstituents = dbCursor.fetchone()[0]
    dbCursor.execute("""SELECT count(*) FROM index_revision
       WHERE index_id IN
          (SELECT mem.id FROM index_member mem
             JOIN index_family fam ON mem.family_id=fam.id
             WHERE fam.name like '%MSCI%')
       AND dt < :date_arg""", date_arg=date)
    numRevs = dbCursor.fetchone()[0]
    print('%s: Deleting %d revisions with %d constituents prior to %s' % (
        datetime.datetime.now(), numRevs, numConstituents, date))
    # Now delete them
    dbCursor.execute("""DELETE FROM index_constituent
       WHERE revision_id IN (SELECT id FROM index_revision
          WHERE index_id IN (SELECT mem.id FROM index_member mem
             JOIN index_family fam ON mem.family_id=fam.id
             WHERE fam.name like '%MSCI%')
          AND dt < :date_arg)""", date_arg=date)
    dbCursor.execute("""DELETE FROM index_revision
       WHERE index_id IN
          (SELECT mem.id FROM index_member mem
             JOIN index_family fam ON mem.family_id=fam.id
             WHERE fam.name like '%MSCI%')
       AND dt < :date_arg""", date_arg=date)

def main():
    dbConnection = cx_Oracle.connect(
            'marketdb_global', 'marketdb_global', 'research')
    dbCursor = dbConnection.cursor()
    dbCursor.execute("""SELECT count(*) FROM index_member mem
       JOIN index_family fam ON mem.family_id=fam.id
       WHERE fam.name like '%MSCI%'""")
    numMembers = dbCursor.fetchone()[0]
    dbCursor.execute("""SELECT count(*) FROM index_revision WHERE index_id IN
       (SELECT mem.id FROM index_member mem
          JOIN index_family fam ON mem.family_id=fam.id
          WHERE fam.name like '%MSCI%')""")
    numRevs = dbCursor.fetchone()[0]
    print('%d MSCI benchmarks with %d revisions present in research DB' % (numMembers, numRevs))
    yearInc = datetime.timedelta(365)
    year = datetime.date(2000, 1, 1)
    endDate = datetime.date.today()
    #endDate = datetime.date(2002, 1, 1)
    while year < endDate:
        deleteMSCIBenchmarksPriorTo(dbCursor, year)
        year += yearInc
    deleteMSCIBenchmarksPriorTo(dbCursor, year)
    dbConnection.commit()

if __name__ == '__main__':
    main()
