#*.db may be opened by using SQLite. For installing this:
# 1. Download precompiled binaries for Windows
# 2. Create a folder sqlite and unzip files in this folder
# 3. Optionally add sqlite folder in your PATH environment variable
# 4. Example of usage from cmd:
# sqlite3 C:\jira.db
# .schema Jira

import sqlite3
import csv

db_path = r"C:\jira.db"
jira_csv_path = r"C:\data.csv"

with open(jira_csv_path, 'r', encoding='utf-8') as jira_file, sqlite3.connect(db_path) as connection:
    records = csv.DictReader(jira_file, delimiter=';')

    cursor = connection.cursor()
    cursor.execute("DROP TABLE IF EXISTS Jira;")
    cursor.execute("""
                   CREATE TABLE Jira(
                   issue_id TEXT,
                   issue_key TEXT,
                   project_code TEXT,
                   req_type TEXT,
                   req_type_id TEXT,
                   req_type_service_desk_id TEXT,
                   issue_type TEXT,
                   service_code TEXT,
                   status TEXT,
                   reporter_login TEXT,
                   reporter_name TEXT,
                   datetime_creation TEXT,
                   description TEXT,
                   kz_problem_description TEXT,
                   kz_suggest_improv TEXT,
                   kz_result_expect TEXT,
                   cu_education_topic TEXT,
                   position TEXT,
                   department TEXT
                   );
    """)

    for record in records:
        data = tuple(list(record.values())[1:])
        cursor.execute("INSERT INTO Jira VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);", data)


with sqlite3.connect(db_path) as connection:
    cursor = connection.cursor()
    cursor.execute("""
                  SELECT issue_key 
                  FROM Jira
                  WHERE reporter_login = 'name.second_name'
                  """)
    for record in cursor.fetchall():
        print(record)
