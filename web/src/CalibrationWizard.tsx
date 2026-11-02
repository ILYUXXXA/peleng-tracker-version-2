import { useEffect, useState, type FormEvent } from 'react'

type CalibrationState = { status: { valid: boolean; camera_ids: number[] }; calibration?: unknown }
type CalibrationResult = { quality: { usable_pairs: number; stereo_rms_px: number; camera0_rms_px: number; camera1_rms_px: number; baseline_m: number; image_size: number[] }; note: string }

export function CalibrationWizard() {
  const [status, setStatus] = useState<CalibrationState['status']>({ valid: false, camera_ids: [] })
  const [camera0, setCamera0] = useState('0')
  const [camera1, setCamera1] = useState('1')
  const [columns, setColumns] = useState('9')
  const [rows, setRows] = useState('6')
  const [square, setSquare] = useState('0.025')
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const [result, setResult] = useState<CalibrationResult | null>(null)

  const refresh = async () => {
    const response = await fetch('/api/calibration/3d')
    if (!response.ok) throw new Error('Не удалось прочитать состояние калибровки')
    const data = await response.json() as CalibrationState
    setStatus(data.status)
  }
  useEffect(() => { void refresh().catch(error => setMessage(error.message)) }, [])

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const form = event.currentTarget
    const data = new FormData(form)
    if ((data.getAll('camera0_images') as File[]).length < 12 || (data.getAll('camera1_images') as File[]).length < 12) {
      setMessage('Нужно минимум 12 пар изображений — одинаковое число файлов с обеих камер.')
      return
    }
    setBusy(true); setMessage('Идёт калибровка. Не закрывайте страницу…'); setResult(null)
    try {
      const response = await fetch('/api/calibration/3d/checkerboard', { method: 'POST', body: data })
      const payload = await response.json() as CalibrationResult & { detail?: string; status?: CalibrationState['status'] }
      if (!response.ok) throw new Error(payload.detail ?? 'Калибровка завершилась ошибкой')
      setResult(payload); if (payload.status) setStatus(payload.status)
      setMessage('Калибровка сохранена. Теперь не перемещайте телефоны.')
      form.reset()
    } catch (error) { setMessage(error instanceof Error ? error.message : 'Ошибка калибровки') }
    finally { setBusy(false) }
  }

  return <section className="panel calibration-wizard">
    <p className="eyebrow">Stereo calibration · metric 3D</p>
    <h2>{status.valid ? 'Камеры откалиброваны' : 'Калибровка двух телефонов'}</h2>
    <p className="muted">Статус: {status.valid ? `камеры ${status.camera_ids.join(', ')} · 3D готово` : 'нет калибровки — 3D координаты недоступны'}.</p>
    <ol className="calibration-steps">
      <li>Распечатайте шахматную доску с <b>9 × 6 внутренними углами</b>. Измерьте сторону клетки линейкой.</li>
      <li>Закрепите телефоны неподвижно. Снимайте доску одновременно обоими телефонами с разных расстояний и наклонов.</li>
      <li>Сохраните не менее 12 пар кадров: в каждой паре доска должна быть в одной позе и полностью видна на обоих кадрах. Не меняйте разрешение, фокус и зум.</li>
      <li>Выберите файлы в одинаковом порядке: кадр №1 камеры A соответствует кадру №1 камеры B.</li>
    </ol>
    <form onSubmit={submit}>
      <div className="camera-fields">
        <label>ID камеры A<input name="camera0_id" type="number" min="0" value={camera0} onChange={event => setCamera0(event.target.value)} required /></label>
        <label>ID камеры B<input name="camera1_id" type="number" min="0" value={camera1} onChange={event => setCamera1(event.target.value)} required /></label>
      </div>
      <div className="camera-fields">
        <label>Внутренние углы по ширине<input name="columns" type="number" min="4" max="20" value={columns} onChange={event => setColumns(event.target.value)} required /></label>
        <label>Внутренние углы по высоте<input name="rows" type="number" min="4" max="20" value={rows} onChange={event => setRows(event.target.value)} required /></label>
      </div>
      <label>Сторона клетки, метры<input name="square_size_m" type="number" min="0.002" max="0.5" step="0.001" value={square} onChange={event => setSquare(event.target.value)} required /></label>
      <label>Кадры камеры A — минимум 12<input name="camera0_images" type="file" accept="image/*" multiple required /></label>
      <label>Соответствующие кадры камеры B — в том же порядке<input name="camera1_images" type="file" accept="image/*" multiple required /></label>
      <button className="primary" type="submit" disabled={busy}>{busy ? 'Калибруем…' : 'Вычислить и сохранить калибровку'}</button>
    </form>
    {message && <p className="notice" aria-live="polite">{message}</p>}
    {result && <dl className="calibration-quality">
      <div><dt>Пар использовано</dt><dd>{result.quality.usable_pairs}</dd></div>
      <div><dt>Stereo RMS</dt><dd>{result.quality.stereo_rms_px.toFixed(2)} px</dd></div>
      <div><dt>Baseline</dt><dd>{result.quality.baseline_m.toFixed(3)} m</dd></div>
      <div><dt>Разрешение</dt><dd>{result.quality.image_size.join(' × ')}</dd></div>
    </dl>}
    <p className="hint">Если RMS выше 1–2 px, повторите съёмку с большим разнообразием поз. После калибровки нельзя передвигать камеры. Для изменения расположения нужна новая калибровка.</p>
  </section>
}
