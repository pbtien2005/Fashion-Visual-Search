import { formatColorLabel, formatPercent, formatRgb } from '../utils/formatters'
import { ChevronIcon } from './Icons'
import { ProductCard } from './ProductCard'

function DetectionDebug({ item, debugRows, queryColor }) {
  return (
    <details className="disclosure debug-panel">
      <summary>
        <span>Detection details</span>
        <ChevronIcon size={15} />
      </summary>
      <div className="kv-grid kv-wide">
        {debugRows.map(([label, value]) => (
          <div className="kv-row" key={label}>
            <span>{label}</span>
            <strong className={label === 'Query color' ? 'color-value' : undefined}>
              {label === 'Query color' && queryColor && (
                <span className="color-swatch" style={{ backgroundColor: queryColor }} />
              )}
              {value}
            </strong>
          </div>
        ))}
      </div>
      {item.crop_variants?.length > 0 && (
        <div className="crop-variants">
          {item.crop_variants.map((variant) => (
            <figure className="crop-variant" key={`${variant.name}-${variant.crop_strategy}`}>
              <img src={`data:image/jpeg;base64,${variant.image_base64}`} alt={variant.name} />
              <figcaption>{variant.name}</figcaption>
            </figure>
          ))}
        </div>
      )}
    </details>
  )
}

function ProductResults({ products }) {
  return (
    <div className="product-grid">
      {products.map((product, index) => (
        <ProductCard key={`${product.id || 'product'}-${index}`} product={product} />
      ))}
    </div>
  )
}

export function DetectedItem({ item, index = 0 }) {
  const confidence = formatPercent(item.confidence)
  const confidenceWidth = typeof item.confidence === 'number' ? Math.max(0, Math.min(1, item.confidence)) * 100 : null
  const cropSrc = `data:image/jpeg;base64,${item.crop_image_base64}`
  const hasProducts = Boolean(item.similar_products?.length)
  const lowConfidenceProducts = item.low_confidence_products || []
  const hasLowConfidenceProducts = Boolean(lowConfidenceProducts.length)
  const noConfidentMatch = item.is_confident_match === false
  const queryColor = formatRgb(item.query_color_rgb)
  const queryColorLabel = formatColorLabel(item.query_color_name, item.query_color_rgb)
  const emptyMessage = noConfidentMatch
    ? item.message || 'Detected, but no confident product match found'
    : 'No similar products found for this item.'
  const productCount = hasProducts ? item.similar_products.length : lowConfidenceProducts.length
  const debugRows = [
    ['Class', item.class_name],
    ['Confidence', confidence],
    ['Original bbox', item.original_bbox?.join(', ') || item.bbox?.join(', ')],
    ['Expanded bbox', item.expanded_bbox?.join(', ')],
    ['Padded bbox', item.padded_bbox?.join(', ')],
    ['Output crop', item.crop_width && item.crop_height ? `${item.crop_width} x ${item.crop_height}` : null],
    [
      'Source crop',
      item.source_crop_width && item.source_crop_height ? `${item.source_crop_width} x ${item.source_crop_height}` : null,
    ],
    ['Strategy', item.crop_strategy],
    ['Search variants', item.search_variants?.join(', ')],
    ['Variant count', item.search_variant_count],
    ['Query color', queryColorLabel],
    ['Query style', item.query_style_label],
    ['Reject reason', item.reject_reason],
    ['Top final score', formatPercent(item.top_final_score)],
    ['Top image score', formatPercent(item.top_image_similarity)],
    ['Final threshold', formatPercent(item.match_score_threshold)],
    ['Image threshold', formatPercent(item.min_image_similarity)],
  ].filter(([, value]) => value)

  return (
    <section className="detected-item">
      <aside className="crop-column">
        <div className="crop-panel">
          <img src={cropSrc} alt={item.class_name_vi || item.class_name} />
          <span className="crop-index">{String(index + 1).padStart(2, '0')}</span>
        </div>
        <div className="crop-meta">
          <h2>{item.class_name_vi || item.class_name}</h2>
          {item.class_name_vi && <p>{item.class_name}</p>}
          {confidence && (
            <div className="confidence" title={`Detection confidence ${confidence}`}>
              <div className="confidence-bar"><span style={{ width: `${confidenceWidth}%` }} /></div>
              <span>{confidence}</span>
            </div>
          )}
          {queryColor && (
            <span className="color-value query-color">
              <span className="color-swatch" style={{ backgroundColor: queryColor }} />
              {item.query_color_name || 'Detected color'}
            </span>
          )}
        </div>
      </aside>

      <div className="item-results">
        <div className="item-header">
          <h3>{hasProducts ? 'Similar products' : hasLowConfidenceProducts ? 'Suggestions' : 'No match'}</h3>
          {productCount > 0 && <span className="pill">{productCount} found</span>}
        </div>

        {hasProducts ? (
          <ProductResults products={item.similar_products} />
        ) : hasLowConfidenceProducts ? (
          <div className="low-confidence-panel">
            <div className="notice notice-warn">
              {emptyMessage}. Showing low confidence suggestions for review.
            </div>
            <ProductResults products={lowConfidenceProducts} />
          </div>
        ) : (
          <div className={noConfidentMatch ? 'notice notice-warn' : 'empty-inline'}>{emptyMessage}</div>
        )}

        <DetectionDebug item={item} debugRows={debugRows} queryColor={queryColor} />
      </div>
    </section>
  )
}
