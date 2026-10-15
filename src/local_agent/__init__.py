"""Local Agent package."""

import sqlite3
import sys

# Chroma 需要 SQLite >= 3.35；旧 Linux 系统在包入口使用随环境安装的后端。
if sqlite3.sqlite_version_info < (3, 35, 0):
    import pysqlite3

    sys.modules["sqlite3"] = pysqlite3
