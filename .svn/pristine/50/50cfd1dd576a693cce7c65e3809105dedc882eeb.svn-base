
import cx_Oracle
import sys

def main():
    sid = sys.argv[1]
    mktdb = cx_Oracle.connect('marketdb_global', 'marketdb_global', sid)
    dbCursor = mktdb.cursor()
    dbCursor.execute("SELECT table_name, partitioned FROM user_all_tables")
    for (tableName,partitioned) in sorted(dbCursor.fetchall()):
        try:
            dbCursor.execute('SELECT count(*) from %s' % tableName)
            print('%s : %d' % (tableName, dbCursor.fetchone()[0]))
            if partitioned == 'YES':
                dbCursor.execute("SELECT partition_name FROM user_tab_partitions WHERE table_name = :table_arg",
                                 table_arg=tableName)
                for (partitionName,) in sorted(dbCursor.fetchall()):
                    dbCursor.execute('SELECT count(*) from %s PARTITION (%s)' % (tableName, partitionName))
                    print('%s.%s : %d' % (tableName, partitionName, dbCursor.fetchone()[0]))
        except cx_Oracle.DatabaseError as ex:
            print(ex)
            print('Error counting %s' % tableName)
    dbCursor.close()
    mktdb.close()

if __name__ == '__main__':
    main()
