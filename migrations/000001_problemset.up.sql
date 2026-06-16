-- @block Create problemset table
CREATE TABLE problemset (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    problem TEXT NOT NULL,
    answer TEXT NOT NULL,
    problem_type TEXT NOT NULL,
    solution TEXT,
    time_to_solve INT, --time taken by llm in seconds
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
