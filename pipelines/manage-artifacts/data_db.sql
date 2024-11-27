DROP TABLE IF EXISTS Sentiment;
CREATE TABLE Sentiment (
    comment TEXT,
    sentiment VARCHAR(100)
);

INSERT INTO Sentiment (comment, sentiment)
VALUES
    ('   I am worried about it.   ', 'negative'),
    ('This is wonderful', 'positive'),
    ('This is terrible, do not even try', 'negative'),
    ('This is amazing!', 'positive'),
    ('That is not my cup of tea.', 'negative'),
    ('I like it so much   .', 'positive');

GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO developer;
