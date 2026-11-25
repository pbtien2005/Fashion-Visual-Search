import { ClockIcon, TrashIcon } from '../components/Icons'

function formatDate(value) {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ''
  return date.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

export function HistoryView({ history, onClearHistory }) {
  return (
    <section className="view">
      <div className="section-head">
        <div>
          <p className="eyebrow">Saved in this browser</p>
          <h1 className="view-title">Search history</h1>
        </div>
        <button type="button" className="btn btn-ghost" disabled={!history.length} onClick={onClearHistory}>
          <TrashIcon size={16} />
          Clear all
        </button>
      </div>

      {history.length ? (
        <div className="history-list">
          {history.map((entry) => (
            <article className="history-card" key={entry.id}>
              <div className="history-card-head">
                <div>
                  <h2>{entry.file_name || 'Uploaded image'}</h2>
                  <p>{formatDate(entry.created_at)}</p>
                </div>
                <span className="pill">
                  {entry.detected_items.length} {entry.detected_items.length === 1 ? 'item' : 'items'}
                </span>
              </div>
              {entry.detected_items.length ? (
                <div className="history-crops">
                  {entry.detected_items.map((item, index) => (
                    <div className="history-crop" key={`${entry.id}-${item.class_name}-${index}`}>
                      <img src={`data:image/jpeg;base64,${item.crop_image_base64}`} alt={item.class_name_vi || item.class_name} />
                      <div>
                        <strong>{item.class_name_vi || item.class_name}</strong>
                        <span>
                          {(item.similar_products?.length || item.low_confidence_products?.length || 0)} products
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="empty-inline">No clothing items found.</div>
              )}
            </article>
          ))}
        </div>
      ) : (
        <div className="empty-card">
          <span className="empty-icon"><ClockIcon size={24} /></span>
          <h2>No search history yet</h2>
          <p>Your recent searches will show up here.</p>
        </div>
      )}
    </section>
  )
}
