#!/usr/lib/nagios/py-venv/bin/python

import psycopg2
import argparse
from datetime import timedelta, datetime


class Status:
    OK = 0
    WARNING = 1
    CRITICAL = 2
    UNKNOWN = 3


class DBCursor():
    _conn = None
    _cursor = None

    def __init__(self, cursor_factory=None):
        self._cursor_factory = cursor_factory

    def __enter__(self):
        self._conn = psycopg2.connect(host='akutan.avo.alaska.edu', database='so2data',
                                      cursor_factory=self._cursor_factory,
                                      user="vvupload", password = "vvupload")
        self._cursor = self._conn.cursor()
        return self._cursor

    def __exit__(self, *args, **kwargs):
        try:
            self._conn.rollback()
        except AttributeError:
            return  # No connection
        self._conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Check VolcView Upload status for all sectors")
    parser.add_argument('-w', dest = "warning", help = "WARNING time, in hours", required = True,
                        metavar = "Warning", type = int)
    parser.add_argument('-c', dest = "critical", help = "CRITICAL time, in hours", required = True,
                        metavar = "Critical", type = int)
    parser.add_argument('-t', dest = "type", help = "Type of sensor", required = False,
                        metavar = "Type", default = 'TROPOMI', type = str)
    parser.add_argument('-v', '--verbose', action = 'store_const', const = True,
                        default = False, help = 'Show verbose debugging output')
    args = parser.parse_args()

    warn_time = timedelta(hours = args.warning)
    warn_hours = int(warn_time.total_seconds() / 60 / 60)
    crit_time = timedelta(hours = args.critical)
    crit_hours = int(crit_time.total_seconds() / 60 / 60)
    perf_data = f"|{warn_hours*60*60}s;{crit_hours*60*60}s"

    check_sql = """
    SELECT
        sector,
        last_update,
        LPAD(FLOOR(EXTRACT(epoch FROM (NOW() - last_update)) / 3600)::text, 2, '0')
        || ':' ||
        LPAD(FLOOR((EXTRACT(epoch FROM (NOW() - last_update)) %% 3600) / 60)::text, 2, '0')
    FROM vvstatus
    WHERE last_update<now()-%s
        AND type=%s;
    """

    with DBCursor() as cursor:
        cursor.execute(check_sql, (crit_time, args.type))
        critical = cursor.fetchall()
        if critical:
            print(f"CRITICAL: One or more VolcView {args.type} sectors is outdated (last upload older than {round(crit_hours)} hours){perf_data}")
            for sector, utime, age in critical:
                isinstance(utime, datetime)
                utime = utime.astimezone(tz = None)
                print(f"Sector: {sector}. Last upload: {utime.strftime('%m/%d/%y %H:%M:%S')} Local ({age})")
            exit(Status.CRITICAL)

        cursor.execute(check_sql, (warn_time, args.type))
        warnings = cursor.fetchall()
        if warnings:
            print(f"WARNING: One or more VolcView {args.type} sectors is stale (last upload older than {round(warn_hours)} hours){perf_data}")
            for sector, utime, age in warnings:
                isinstance(utime, datetime)
                utime = utime.astimezone(tz = None)
                print(f"Sector: {sector}. Last upload: {utime.strftime('%m/%d/%y %H:%M:%S')} Local ({age})")
            exit(Status.WARNING)

    # If we get here, then both CRITICAL and WARNING checks returned no results, so we are OK
    print(f"OK: All {args.type} sectors are up-to-date{perf_data}")
    exit(Status.OK)
