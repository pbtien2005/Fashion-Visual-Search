import {
  formatColorLabel,
  formatPercent,
  formatRgb,
  formatVariantScores,
  isPublicImageUrl,
  resolveApiUrl,
} from '../utils/formatters'
import { ChevronIcon, ExternalIcon, ImageIcon } from './Icons'

function ScoreRows({ rows, productColor, productColorLabel }) {
  return (
    <details className="disclosure score-panel">
      <summary>
        <span>Score details</span>
        <ChevronIcon size={15} />
      </summary>
      <div className="kv-grid">
        {rows.map(([label, value]) => (
          <div className="kv-row" key={label}>
            <span>{label}</span>
            <strong>{value}</strong>
          </div>
        ))}
        {productColorLabel && (
          <div className="kv-row">
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

function ProductCardContent({ product, linked }) {
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
  const hasPrice = product.price !== null && product.price !== undefined

  return (
    <>
      <div className="product-media">
        {imageUrl ? (
          <img src={imageUrl} alt={product.title || product.id || 'Product'} loading="lazy" />
        ) : (
          <div className="product-placeholder">
            <ImageIcon size={28} />
            <span>No public image</span>
          </div>
        )}
        {similarity && <span className="match-badge">{similarity} match</span>}
        {linked && (
          <span className="open-badge" aria-hidden="true">
            <ExternalIcon size={15} />
          </span>
        )}
      </div>
      <div className="product-body">
        {product.brand && <span className="product-brand">{product.brand}</span>}
        <h3>{product.title || product.name || product.id || 'Untitled product'}</h3>
        <div className="product-foot">
          {hasPrice && <span className="product-price">${product.price}</span>}
          {product.category && <span className="product-category">{product.category}</span>}
        </div>
        {productColor && (
          <span className="color-value product-color">
            <span className="color-swatch" style={{ backgroundColor: productColor }} />
            {product.dominant_color_name || 'Color'}
          </span>
        )}
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

  if (productUrl) {
    return (
      <div className="product-card product-link">
        <ProductCardContent product={product} linked />
        <a className="product-open" href={productUrl} target="_blank" rel="noreferrer" aria-label="Open product page">
          <span>View product</span>
          <ExternalIcon size={14} />
        </a>
      </div>
    )
  }

  return (
    <article className="product-card">
      <ProductCardContent product={product} />
    </article>
  )
}
