ALTER TABLE fashion_items
ADD COLUMN IF NOT EXISTS dominant_color_rgb INTEGER[],
ADD COLUMN IF NOT EXISTS dominant_color_name TEXT;
