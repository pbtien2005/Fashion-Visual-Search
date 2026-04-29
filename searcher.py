from AI.vectorizer import extract_vector, fetch_image_for_ai
from db.database import search_similar_by_vector


def search_similar_fashion(image_url: str, top_k: int = 5):
    print("1. Downloading and analyzing input image...")
    img = fetch_image_for_ai(image_url)
    if not img:
        print("Could not load image. Check the URL and try again.")
        return []

    print("2. Extracting CLIP vector...")
    query_vector = extract_vector(img)
    if not query_vector:
        return []

    print("3. Searching PostgreSQL + pgvector...")
    results = search_similar_by_vector(query_vector, top_k=top_k)

    print("\n=== SIMILAR FASHION ITEMS ===")
    for i, item in enumerate(results, start=1):
        similarity = item.get("similarity") or 0
        title = item.get("title") or ""
        print(f"Top {i} (similarity: {similarity * 100:.2f}%):")
        print(f"- ID:     {item.get('id')}")
        print(f"- Title:  {title[:60]}...")
        print(f"- Price:  ${item.get('price')} | Brand: {item.get('brand')}")
        print(f"- Image:  {item.get('image_url')}\n")

    return results


if __name__ == "__main__":
    test_image_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSVUuFRc992mmYkLPowCctNKMQGDCzGES6GTQ&s"

    print("=== SIMILAR PRODUCT SEARCH ===")
    print(f"Input image: {test_image_url}\n")

    search_similar_fashion(image_url=test_image_url, top_k=3)
