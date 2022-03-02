
import datetime
import logging
from riskmodels import ModelDB
from riskmodels import Utilities

def toDate(str):
    if str == 'MAXVALUE':
        return datetime.date(2999, 12, 31)
    return Utilities.parseISODate(str[10:20])

def main():
    mdl = ModelDB.ModelDB(user='modeldb_global', passwd='modeldb_global',
                          sid='research')
    cur = mdl.dbCursor
    tableName = 'SUB_ISSUE_DATA'
    cur.execute("""SELECT max(high_value_length)
       FROM user_tab_partitions WHERE table_name=:table_arg""", table_arg=tableName)
    maxLength = cur.fetchone()[0]
    cur.setoutputsize(maxLength, 2)
    cur.execute("""SELECT partition_name, high_value, partition_position
       FROM user_tab_partitions WHERE table_name=:table_arg
       ORDER BY partition_position ASC""", table_arg=tableName)
    partitions = cur.fetchall()
    print("Current partitions")
    partitions = [(name, toDate(highVal), position)
                  for (name, highVal, position)
                  in partitions]
    for (name, highVal, position) in partitions:
        print(name, highVal)
    startYear = 2000
    numMonths = 48
    newBreakPoints = [datetime.date(startYear + (i / 12), 1 + (i % 12), 1)
                      for i in range(numMonths)]
    print('Adding %d breakpoints starting %d' % (numMonths, startYear))
    for breakPoint in newBreakPoints:
        for (idx, (partName, highVal, position)) in enumerate(partitions):
            if highVal > breakPoint:
                break
        if idx > 0 and breakPoint == partitions[idx-1][1]:
            print('Skipping existing breakpoint %s' % breakPoint)
        else:
            print('Creating new partition with breakpoint %s' % breakPoint)
            newPartName = 'P_%s_%d%02d%02d' % (
                tableName, breakPoint.year, breakPoint.month, breakPoint.day)
            query = """ALTER TABLE %(table)s SPLIT PARTITION %(oldPart)s
               AT (TO_DATE('%(date)s', 'YYYY-MM-DD'))
               INTO (PARTITION %(newPart)s, PARTITION %(oldPart)s)""" % {
                'table': tableName, 'date':breakPoint,
                'oldPart': partName,
                'newPart': newPartName }
            print(query, breakPoint, partName, newPartName)
            cur.execute(query)
    mdl.finalize()

if __name__ == '__main__':
    main()
