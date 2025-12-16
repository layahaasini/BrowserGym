#!/bin/bash
set -e

mysql -u root -p"$MYSQL_ROOT_PASSWORD" <<-EOSQL
    DROP DATABASE IF EXISTS osclass;
    CREATE DATABASE osclass;
    USE osclass;
    source /docker-entrypoint-initdb.d/osclass_craigslist.sql;
EOSQL