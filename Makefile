PROJECT_NAME=llm

e ?= local

ENV_FILE=~/src/secrets/${PROJECT_NAME}/${e}.env

MIGRATION_PATH=migrations
MIGRATION_TABLE=schema_migrations

include ${ENV_FILE}
export $(shell sed 's/=.*//' ${ENV_FILE})


run_sql:
	psql -U $(POSTGRES_USER) $(POSTGRES_DB) -h $(POSTGRES_HOST) -f  ${sql_path}

migrate_create:
	migrate create -ext sql -dir ${MIGRATION_PATH} -seq ${seq}

migrate_up:
	unset PGSERVICEFILE && migrate -database "${POSTGRES_GO_DSN}&x-migrations-table=${MIGRATION_TABLE}" -path ${MIGRATION_PATH} up

migrate_down:
	unset PGSERVICEFILE &&migrate -database "${POSTGRES_GO_DSN}&x-migrations-table=${MIGRATION_TABLE}" -path ${MIGRATION_PATH} down


reset_db:
	psql -U $(POSTGRES_USER) $(POSTGRES_DB) -h $(POSTGRES_HOST) -f ./compose/psql/reset.sql


wait_for_db:
	echo "environment: ${e}" && \
	echo "Waiting for PostgreSQL to be ready..." && \
    until docker-compose -f compose/local.yml exec -T postgres pg_isready -U ${POSTGRES_USER} > /dev/null 2>&1; do \
        echo "PostgreSQL is not ready yet..."; \
        sleep 1; \
    done; \
    echo "PostgreSQL is ready!"
