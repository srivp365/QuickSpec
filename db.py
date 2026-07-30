# a file to interact with my db and make one time changes
import sqlite3

# establish connection to db
conn = sqlite3.connect("data/db/chunks.db")
cur = conn.cursor()
