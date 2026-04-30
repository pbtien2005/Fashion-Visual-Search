import {
  formatColorLabel,
  formatPercent,
  formatRgb,
  formatVariantScores,
  isPublicImageUrl,
  resolveApiUrl,
} from '../utils/formatters'

function ScoreRows({ rows, productColor, productColorLabel }) {
  return (
    <details className="score-panel">
      <summary>Score</summary>
      <div className="score-grid">
        {rows.map(([label, value]) => (
          <div className="score-row" key={label}>
            <span>{label}</span>
            <strong>{value}</strong>
          </div>
        ))}
        {productColorLabel && (
          <div className="score-row">
            <span>Product color</span>
            <strong className="color-value">
              {productColor && <span className="color-swatch" style={{ backgroundColor: productColor }} />}
              {productColorLabel}
            </strong>
          </div>
        )}
      </div>
    </details>
  )
}

function ProductCardContent({ product }) {
  const imageUrl = resolveApiUrl(product.usable_image_url)
  const similarity = formatPercent(product.similarity)
  const productColor = formatRgb(product.dominant_color_rgb)
  const productColorLabel = formatColorLabel(product.dominant_color_name, product.dominant_color_rgb)
  const scoreRows = [
    ['Image', formatPercent(product.image_similarity)],
    ['Category', formatPercent(product.category_score)],
    ['Color', formatPercent(product.color_score)],
    ['Text/style', formatPercent(product.text_style_score)],
    ['Text', formatPercent(product.text_score)],
    ['Style', formatPercent(product.style_score)],
    ['Cat penalty', formatPercent(product.category_penalty)],
    ['Color penalty', formatPercent(product.color_penalty)],
    ['Final', formatPercent(product.final_score)],
    ['Variant', product.best_crop_variant],
    ['Variants', formatVariantScores(product.variant_scores)],
    ['Query style', product.query_style_label],
    ['Product styles', Array.isArray(product.product_styles) ? product.product_styles.join(', ') : null],
    ['Reason', product.match_reason],
  ].filter(([, value]) => value)

  return (
    <>
      <div className="product-media">
        {imageUrl ? (
          <img src={imageUrl} alt={product.title || product.id || 'Product'} />
        ) : (
          <div className="product-placeholder">No public image</div>
        )}
      </div>
      <div className="product-body">
        <h3>{product.title || product.name || product.id || 'Untitled product'}</h3>
        <div className="product-meta">
          {product.brand && <span>{product.brand}</span>}
          {product.price !== null && product.price !== undefined && <span>${product.price}</span>}
          {similarity && <span>{similarity} match</span>}
        </div>
        {product.category && <p className="category">{product.category}</p>}
        {scoreRows.length > 0 && (
          <ScoreRows rows={scoreRows} productColor={productColor} productColorLabel={productColorLabel} />
        )}
        {!imageUrl && product.image_url && <p className="raw-url">{product.image_url}</p>}
      </div>
    </>
  )
}

export function ProductCard({ product }) {
  const productUrl = isPublicImageUrl(product.product_url) ? product.product_url : null
  const content = <ProductCardContent product={product} />

  if (productUrl) {
    return (
      <a className="product-card product-link" href={productUrl} target="_blank" rel="noreferrer">
        {content}
      </a>
    )
  }

  return <article className="product-card">{content}</article>
}
