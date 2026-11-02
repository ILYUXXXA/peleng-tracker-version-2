import { useEffect, useRef, useState } from 'react'

export function PhonePublisher() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const timerRef = useRef<number | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const uploadingRef = useRef(false)
  const [cameraId, setCameraId] = useState(0)
  const [name, setName] = useState('Hackathon phone')
  const [state, setState] = useState<'idle' | 'starting' | 'live' | 'error'>('idle')
  const [message, setMessage] = useState('Телефон ещё не подключён')

  const stop = () => {
    if (timerRef.current !== null) window.clearInterval(timerRef.current)
    timerRef.current = null
    streamRef.current?.getTracks().forEach(track => track.stop())
    streamRef.current = null
    setState('idle')
    setMessage('Трансляция остановлена')
  }

  useEffect(() => stop, [])

  const uploadFrame = async () => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas || video.readyState < 2 || uploadingRef.current) return
    const scale = Math.min(1, 1280 / video.videoWidth)
    canvas.width = Math.max(1, Math.round(video.videoWidth * scale))
    canvas.height = Math.max(1, Math.round(video.videoHeight * scale))
    canvas.getContext('2d')?.drawImage(video, 0, 0, canvas.width, canvas.height)
    const blob = await new Promise<Blob | null>(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.78))
    if (!blob) return
    uploadingRef.current = true
    try {
      const form = new FormData()
      form.append('frame', blob, 'frame.jpg')
      const response = await fetch(`/api/cameras/phone/${cameraId}/frame?name=${encodeURIComponent(name)}`, {
        method: 'POST', body: form,
      })
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      setState('live')
      setMessage('Кадры поступают в Peleng Tracker')
    } catch (error) {
      setState('error')
      setMessage(`Ошибка отправки: ${error instanceof Error ? error.message : 'unknown'}`)
    } finally {
      uploadingRef.current = false
    }
  }

  const start = async () => {
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      setState('error')
      setMessage('Браузеру нужен HTTPS. Откройте защищённый адрес Peleng Tracker.')
      return
    }
    stop()
    setState('starting')
    setMessage('Запрашиваем доступ к задней камере…')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }
      timerRef.current = window.setInterval(uploadFrame, 125)
      setState('live')
      setMessage('Камера подключена · отправка до 8 FPS')
    } catch (error) {
      setState('error')
      setMessage(`Не удалось открыть камеру: ${error instanceof Error ? error.message : 'permission denied'}`)
    }
  }

  return <main className="phone-page">
    <header><p className="eyebrow">Peleng Tracker · phone source</p><h1>Камера телефона</h1><p>Оставьте эту страницу открытой во время демонстрации.</p></header>
    <section className="phone-card">
      <div className="phone-preview"><video ref={videoRef} muted playsInline /><canvas ref={canvasRef} hidden /></div>
      <div className="phone-controls">
        <label>Camera ID<input type="number" min="0" value={cameraId} disabled={state === 'live'} onChange={event => setCameraId(Number(event.target.value))} /></label>
        <label>Название<input value={name} disabled={state === 'live'} onChange={event => setName(event.target.value)} /></label>
        <p className={`phone-state ${state}`}>{message}</p>
        {state === 'live' ? <button className="danger" onClick={stop}>Остановить</button> : <button className="primary" onClick={start}>Включить заднюю камеру</button>}
      </div>
    </section>
  </main>
}
