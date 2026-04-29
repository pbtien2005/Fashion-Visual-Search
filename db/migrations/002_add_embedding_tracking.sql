ALTER TABLE fashion_items
ADD COLUMN IF NOT EXISTS embedding_model TEXT DEFAULT 'patrickjohncyh/fashion-clip',
ADD COLUMN IF NOT EXISTS embedding_version INTEGER DEFAULT 2;

UPDATE fashion_items
SET embedding_model = 'patrickjohncyh/fashion-clip'
WHERE embedding_model IS NULL;

UPDATE fashion_items
SET embedding_version = 2
WHERE embedding_version IS NULL;
