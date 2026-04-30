export function HistoryView({ history, onClearHistory }) {
  return (
    <section className="panel-stack">
      <div className="section-toolbar">
        <div>
          <h1>Search History</h1>
          <p>Recent searches saved in this browser.</p>
        </div>
        <button type="button" disabled={!history.length} onClick={onClearHistory}>Clear</button>
      </div>

      {history.length ? (
        <div className="history-list">
          {history.map((entry) => (
            <article className="history-card" key={entry.id}>
              <div className="history-card-head">
                <div>
                  <h2>{entry.file_name || 'Uploaded image'}</h2>
                  <p>{new Date(entry.created_at).toLocaleString()}</p>
                </div>
                <span>{entry.detected_items.length} items</span>
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
        <div className="status-panel empty-state">No search history yet.</div>
      )}
    </section>
  )
}
