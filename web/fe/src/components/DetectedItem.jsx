import { formatColorLabel, formatPercent, formatRgb } from '../utils/formatters'
import { ProductCard } from './ProductCard'

function DetectionDebug({ item, debugRows, queryColor }) {
  return (
    <details className="debug-panel">
      <summary>Detection debug</summary>
      <div className="debug-grid">
        {debugRows.map(([label, value]) => (
          <div className="debug-row" key={label}>
            <span>{label}</span>
            <strong className={label === 'Query color' ? 'color-value' : undefined}>
              {label === 'Query color' && queryColor && (
                <span className="color-swatch" style={{ backgroundColor: queryColor }} />
              )}
              {value}
            </strong>
          </div>
        ))}
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
      </div>
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

export function DetectedItem({ item }) {
  const confidence = formatPercent(item.confidence)
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
      <div className="crop-panel">
        <img src={cropSrc} alt={item.class_name_vi || item.class_name} />
      </div>
      <div className="item-results">
        <div className="item-header">
          <div>
            <h2>{item.class_name_vi || item.class_name}</h2>
            <p>{item.class_name}</p>
          </div>
          <div className="badges">
            {confidence && <span>{confidence}</span>}
            <span>[{item.bbox?.join(', ')}]</span>
          </div>
        </div>
        <DetectionDebug item={item} debugRows={debugRows} queryColor={queryColor} />
        {hasProducts ? (
          <ProductResults products={item.similar_products} />
        ) : hasLowConfidenceProducts ? (
          <div className="low-confidence-panel">
            <div className="threshold-message">
              {emptyMessage}. Showing low confidence suggestions for review.
            </div>
            <div className="low-confidence-title">Low confidence suggestions</div>
            <ProductResults products={lowConfidenceProducts} />
          </div>
        ) : (
          <div className={noConfidentMatch ? 'empty-inline threshold-message' : 'empty-inline'}>{emptyMessage}</div>
        )}
      </div>
    </section>
  )
}
