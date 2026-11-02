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
  const [captured0, setCaptured0] = useState<File[]>([])
  const [captured1, setCaptured1] = useState<File[]>([])
  const [capturing, setCapturing] = useState(false)

  useEffect(() => {
    let active = true
    fetch('/api/calibration/3d').then(async response => {
      if (!response.ok) throw new Error('Не удалось прочитать состояние калибровки')
      return response.json() as Promise<CalibrationState>
    }).then(data => { if (active) setStatus(data.status) }).catch(error => {
      if (active) setMessage(error instanceof Error ? error.message : 'Ошибка состояния калибровки')
    })
    return () => { active = false }
  }, [])

  const capturePair = async () => {
    setCapturing(true); setMessage('Снимаем синхронную пару кадров…')
    try {
      const [response0, response1] = await Promise.all([
        fetch(`/api/cameras/phone/${camera0}/snapshot`, { cache: 'no-store' }),
        fetch(`/api/cameras/phone/${camera1}/snapshot`, { cache: 'no-store' }),
      ])
      if (!response0.ok || !response1.ok) throw new Error('Обе камеры должны быть подключены и передавать видео.')
      const time0 = Number(response0.headers.get('X-Captured-At'))
      const time1 = Number(response1.headers.get('X-Captured-At'))
      const skewMs = Math.abs(time0 - time1) * 1000
      if (!Number.isFinite(skewMs) || skewMs > 150) throw new Error(`Кадры камер различаются на ${skewMs.toFixed(0)} мс. Повторите захват пары.`)
      const nextIndex = Math.max(captured0.length, captured1.length) + 1
      const [blob0, blob1] = await Promise.all([response0.blob(), response1.blob()])
      setCaptured0(current => [...current, new File([blob0], `camera-${camera0}-${nextIndex}.jpg`, { type: 'image/jpeg' })])
      setCaptured1(current => [...current, new File([blob1], `camera-${camera1}-${nextIndex}.jpg`, { type: 'image/jpeg' })])
      setMessage(`Сохранена пара ${nextIndex}. Поменяйте положение шахматной доски и снимите следующую.`)
    } catch (error) { setMessage(error instanceof Error ? error.message : 'Не удалось снять пару кадров') }
    finally { setCapturing(false) }
  }

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const form = event.currentTarget
    const data = new FormData(form)
    const files0 = (data.getAll('camera0_images') as File[]).filter(file => file.size > 0)
    const files1 = (data.getAll('camera1_images') as File[]).filter(file => file.size > 0)
    data.delete('camera0_images'); data.delete('camera1_images')
    for (const file of [...files0, ...captured0]) data.append('camera0_images', file)
    for (const file of [...files1, ...captured1]) data.append('camera1_images', file)
    if (files0.length + captured0.length < 12 || files1.length + captured1.length < 12) {
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
      setCaptured0([]); setCaptured1([]); form.reset()
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
      <label>Дополнительные файлы камеры A<input name="camera0_images" type="file" accept="image/*" multiple /></label>
      <label>Дополнительные файлы камеры B в том же порядке<input name="camera1_images" type="file" accept="image/*" multiple /></label>
      <button className="secondary" type="button" disabled={busy || capturing || camera0 === camera1} onClick={capturePair}>Снять синхронную пару с телефонов ({captured0.length}/12)</button>
      <p className="hint">Можно снимать пары прямо из работающих видеопотоков. После каждой пары перемещайте доску; нужно минимум 12 разных поз. Текущие кадры должны отличаться по времени не более чем на 150 мс.</p>
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
