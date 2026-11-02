import { Line, OrbitControls } from '@react-three/drei'
import { Canvas } from '@react-three/fiber'
import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'

type ReplayCamera = { id: number; name: string; video: string }
type CameraCalibration = {
  image_size: [number, number]
  camera_matrix: number[][]
  distortion_coefficients: number[]
  rotation_world_to_camera: number[][]
  translation_world_to_camera: number[]
}
type ReplayInfo = {
  manifest: {
    id: string; title: string; source: string; fps: number; duration_s: number
    object_id: string; observation_source: string; cameras: ReplayCamera[]
  }
  calibration: { cameras: Record<string, CameraCalibration> }
  evaluation: { median_error_m?: number | null; coverage?: number | null }
}
type TrackState3D = {
  frame_index: number; timestamp: number; object_id: string
  position_xyz_m: [number, number, number]
  velocity_xyz_mps: [number, number, number]
  acceleration_xyz_mps2: [number, number, number]
  speed_mps: number; acceleration_mps2: number
  source_cameras: number[]; reprojection_error_px: number | null
  quality: number; status: 'measured' | 'predicted'
}
type ReplayTrajectory = { states: TrackState3D[] }

const OLD_COLOR = new THREE.Color('#173b55')
const CURRENT_COLOR = new THREE.Color('#ffca63')

function scenePoint(point: [number, number, number]): [number, number, number] {
  return [point[0], point[2], -point[1]]
}

function cameraPosition(calibration: CameraCalibration): [number, number, number] {
  const rotation = calibration.rotation_world_to_camera
  const translation = calibration.translation_world_to_camera
  const result = [0, 0, 0]
  for (let row = 0; row < 3; row += 1) {
    result[row] = -(rotation[0][row] * translation[0] + rotation[1][row] * translation[1] + rotation[2][row] * translation[2])
  }
  return scenePoint(result as [number, number, number])
}

function TrajectoryScene({ states, time, cameras, target }: { states: TrackState3D[]; time: number; cameras: Record<string, CameraCalibration>; target: [number, number, number] }) {
  const visible = useMemo(() => states.filter(state => state.timestamp <= time), [states, time])
  const sampled = useMemo(() => {
    const stride = Math.max(1, Math.ceil(visible.length / 1200))
    return visible.filter((_, index) => index % stride === 0 || index === visible.length - 1)
  }, [visible])
  const points = useMemo(() => sampled.map(state => scenePoint(state.position_xyz_m)), [sampled])
  const vertexColors = useMemo(() => sampled.map(state => {
    const oldestTime = sampled[0]?.timestamp ?? time
    const span = Math.max(time - oldestTime, 1e-6)
    const progress = Math.max(0, Math.min(1, 1 - (time - state.timestamp) / span))
    const color = OLD_COLOR.clone().lerp(CURRENT_COLOR, progress)
    return [color.r, color.g, color.b] as [number, number, number]
  }), [sampled, time])
  const current = visible.at(-1)

  return <>
    <color attach="background" args={['#08131b']} />
    <ambientLight intensity={0.7} />
    <directionalLight position={[4, 8, 5]} intensity={1.4} />
    <gridHelper args={[20, 20, '#294555', '#182d3a']} />
    <axesHelper args={[1.5]} />
    {points.length > 1 && <Line points={points} vertexColors={vertexColors} lineWidth={4} />}
    {current && <mesh position={scenePoint(current.position_xyz_m)}>
      <sphereGeometry args={[0.09, 24, 16]} />
      <meshStandardMaterial color="#ffca63" emissive="#7a4f00" emissiveIntensity={0.8} />
    </mesh>}
    {Object.entries(cameras).map(([id, camera]) => <group key={id} position={cameraPosition(camera)}>
      <mesh rotation-x={Math.PI / 2}>
        <coneGeometry args={[0.12, 0.3, 4]} />
        <meshStandardMaterial color="#69d7ff" wireframe />
      </mesh>
    </group>)}
    <OrbitControls makeDefault target={target} enableDamping dampingFactor={0.1} minDistance={0.5} maxDistance={80} />
  </>
}

function project(point: [number, number, number], calibration: CameraCalibration): [number, number] | null {
  const [x, y, z] = point
  const rotation = calibration.rotation_world_to_camera
  const translation = calibration.translation_world_to_camera
  const camera = [
    rotation[0][0] * x + rotation[0][1] * y + rotation[0][2] * z + translation[0],
    rotation[1][0] * x + rotation[1][1] * y + rotation[1][2] * z + translation[1],
    rotation[2][0] * x + rotation[2][1] * y + rotation[2][2] * z + translation[2],
  ]
  if (camera[2] <= 0) return null
  let normalizedX = camera[0] / camera[2]
  let normalizedY = camera[1] / camera[2]
  const [k1 = 0, k2 = 0, p1 = 0, p2 = 0, k3 = 0] = calibration.distortion_coefficients
  const radius2 = normalizedX * normalizedX + normalizedY * normalizedY
  const radial = 1 + k1 * radius2 + k2 * radius2 * radius2 + k3 * radius2 * radius2 * radius2
  const distortedX = normalizedX * radial + 2 * p1 * normalizedX * normalizedY + p2 * (radius2 + 2 * normalizedX * normalizedX)
  const distortedY = normalizedY * radial + p1 * (radius2 + 2 * normalizedY * normalizedY) + 2 * p2 * normalizedX * normalizedY
  normalizedX = distortedX
  normalizedY = distortedY
  const matrix = calibration.camera_matrix
  return [matrix[0][0] * normalizedX + matrix[0][2], matrix[1][1] * normalizedY + matrix[1][2]]
}

function VideoOverlay({ states, time, calibration }: { states: TrackState3D[]; time: number; calibration: CameraCalibration }) {
  const segments = useMemo(() => {
    const visible = states.filter(state => state.timestamp <= time)
    const stride = Math.max(1, Math.ceil(visible.length / 350))
    const projected = visible.filter((_, index) => index % stride === 0 || index === visible.length - 1)
      .map(state => project(state.position_xyz_m, calibration))
    return projected.slice(1).flatMap((end, index) => {
      const start = projected[index]
      if (!start || !end) return []
      const sample = visible[Math.min(visible.length - 1, (index + 1) * stride)]
      const oldestTime = visible[0]?.timestamp ?? time
      const span = Math.max(time - oldestTime, 1e-6)
      const progress = sample ? Math.max(0, Math.min(1, 1 - (time - sample.timestamp) / span)) : 1
      const color = OLD_COLOR.clone().lerp(CURRENT_COLOR, progress).getStyle()
      return [{ start, end, color, opacity: 0.2 + progress * 0.8 }]
    })
  }, [calibration, states, time])
  const currentState = states.filter(state => state.timestamp <= time).at(-1)
  const current = currentState ? project(currentState.position_xyz_m, calibration) : null
  const [width, height] = calibration.image_size
  return <svg className="trajectory-overlay" viewBox={`0 0 ${width} ${height}`} aria-hidden="true">
    {segments.map((segment, index) => <line key={index} x1={segment.start[0]} y1={segment.start[1]} x2={segment.end[0]} y2={segment.end[1]} stroke={segment.color} strokeOpacity={segment.opacity} strokeWidth="5" strokeLinecap="round" />)}
    {current && <circle cx={current[0]} cy={current[1]} r="9" fill="#ffca63" stroke="#08131b" strokeWidth="3" />}
  </svg>
}

function formatVector(vector: [number, number, number] | undefined, unit: string) {
  return vector ? `${vector.map(value => value.toFixed(2)).join(' · ')} ${unit}` : `— ${unit}`
}

export default function ReplayEditor() {
  const [info, setInfo] = useState<ReplayInfo | null>(null)
  const [states, setStates] = useState<TrackState3D[]>([])
  const [error, setError] = useState('')
  const [cameraId, setCameraId] = useState<number | null>(null)
  const [time, setTime] = useState(0)
  const [speed, setSpeed] = useState(1)
  const [playing, setPlaying] = useState(false)
  const videoRef = useRef<HTMLVideoElement | null>(null)

  useEffect(() => {
    Promise.all([
      fetch('/api/replay').then(response => { if (!response.ok) throw new Error('Replay bundle не загружен'); return response.json() as Promise<ReplayInfo> }),
      fetch('/api/replay/trajectory').then(response => { if (!response.ok) throw new Error('Траектория ещё не обработана'); return response.json() as Promise<ReplayTrajectory> }),
    ]).then(([nextInfo, trajectory]) => {
      setInfo(nextInfo)
      setStates(trajectory.states)
      setCameraId(nextInfo.manifest.cameras[0]?.id ?? null)
    }).catch(reason => setError(reason instanceof Error ? reason.message : 'Не удалось открыть replay'))
  }, [])

  useEffect(() => {
    if (videoRef.current) videoRef.current.playbackRate = speed
  }, [speed, cameraId])

  useEffect(() => {
    if (!playing) return
    let animationFrame = 0
    let lastUpdate = 0
    const updateClock = (timestamp: number) => {
      if (videoRef.current && timestamp - lastUpdate >= 32) {
        setTime(videoRef.current.currentTime)
        lastUpdate = timestamp
      }
      animationFrame = window.requestAnimationFrame(updateClock)
    }
    animationFrame = window.requestAnimationFrame(updateClock)
    return () => window.cancelAnimationFrame(animationFrame)
  }, [playing, cameraId])

  const duration = Math.max(info?.manifest.duration_s ?? 0, states.at(-1)?.timestamp ?? 0)
  const current = useMemo(() => states.reduce<TrackState3D | undefined>((selected, state) => state.timestamp <= time ? state : selected, undefined), [states, time])
  const camera = info?.manifest.cameras.find(item => item.id === cameraId)
  const calibration = cameraId === null ? undefined : info?.calibration.cameras[String(cameraId)]
  const sceneView = useMemo(() => {
    if (!states.length) return { center: [0, 0, 0] as [number, number, number], position: [5, 4, 7] as [number, number, number] }
    const points = states.map(state => scenePoint(state.position_xyz_m))
    const minimum = [Math.min(...points.map(point => point[0])), Math.min(...points.map(point => point[1])), Math.min(...points.map(point => point[2]))]
    const maximum = [Math.max(...points.map(point => point[0])), Math.max(...points.map(point => point[1])), Math.max(...points.map(point => point[2]))]
    const center = minimum.map((value, index) => (value + maximum[index]) / 2) as [number, number, number]
    const span = Math.max(2, ...maximum.map((value, index) => value - minimum[index]))
    return { center, position: [center[0] + span * 1.25, center[1] + span, center[2] + span * 1.5] as [number, number, number] }
  }, [states])

  const seek = (nextTime: number) => {
    const clamped = Math.max(0, Math.min(duration, nextTime))
    setTime(clamped)
    if (videoRef.current && Math.abs(videoRef.current.currentTime - clamped) > 0.03) videoRef.current.currentTime = clamped
  }
  const togglePlayback = async () => {
    const video = videoRef.current
    if (!video) return
    if (video.paused) {
      if (video.currentTime >= duration - 0.02) seek(0)
      video.playbackRate = speed
      await video.play()
    } else video.pause()
  }

  if (error) return <section className="panel replay-empty"><p className="eyebrow">3D Replay</p><h2>Нет обработанной записи</h2><p>{error}</p><code>uv run peleng-tracker demo run demo/&lt;bundle&gt;</code></section>
  if (!info) return <section className="panel replay-empty"><p>Загружаем 3D-траекторию…</p></section>

  return <main className="replay-editor">
    <header className="panel replay-header">
      <div><p className="eyebrow">Processed 3D · {info.manifest.observation_source}</p><h2>{info.manifest.title}</h2></div>
      <div className="replay-actions"><a href="/api/replay/export.csv">CSV</a><a href="/api/replay/export.json">JSON</a></div>
    </header>
    <section className="panel scene-panel">
      <div className="section-title"><div><p className="eyebrow">Metric world space</p><h2>3D-траектория</h2></div><span>мышь: вращение · колесо: масштаб</span></div>
      <div className="scene-canvas"><Canvas frameloop="demand" dpr={[1, 1.75]} flat camera={{ position: sceneView.position, fov: 48 }} gl={{ powerPreference: 'high-performance' }}>
        <TrajectoryScene states={states} time={time} cameras={info.calibration.cameras} target={sceneView.center} />
      </Canvas></div>
    </section>
    <section className="panel replay-video-panel">
      <div className="section-title"><div><p className="eyebrow">Fixed calibrated view</p><h2>Проекция на видео</h2></div>
        <select aria-label="Камера" value={cameraId ?? ''} onChange={event => { setPlaying(false); setCameraId(Number(event.target.value)) }}>
          {info.manifest.cameras.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}
        </select>
      </div>
      {camera && calibration && <div className="replay-video-stage" style={{ aspectRatio: `${calibration.image_size[0]} / ${calibration.image_size[1]}` }}>
        <video key={camera.id} ref={videoRef} src={`/api/replay/cameras/${camera.id}/video`} muted playsInline preload="metadata"
          onLoadedMetadata={event => { event.currentTarget.currentTime = time; event.currentTarget.playbackRate = speed }}
          onTimeUpdate={event => setTime(event.currentTarget.currentTime)} onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} onEnded={() => setPlaying(false)} />
        <VideoOverlay states={states} time={time} calibration={calibration} />
      </div>}
    </section>
    <section className="panel replay-metrics">
      <p className="eyebrow">Object · {info.manifest.object_id}</p><h2>{current?.status === 'predicted' ? 'Прогноз' : 'Измерение'}</h2>
      <dl>
        <div><dt>Position</dt><dd>{formatVector(current?.position_xyz_m, 'm')}</dd></div>
        <div><dt>Velocity</dt><dd>{current ? `${current.speed_mps.toFixed(2)} m/s` : '—'}</dd></div>
        <div><dt>Acceleration</dt><dd>{current ? `${current.acceleration_mps2.toFixed(2)} m/s²` : '—'}</dd></div>
        <div><dt>Quality</dt><dd>{current ? `${Math.round(current.quality * 100)}%` : '—'}</dd></div>
        <div><dt>Cameras</dt><dd>{current?.source_cameras.join(', ') || 'prediction'}</dd></div>
        <div><dt>Frame</dt><dd>{current?.frame_index ?? 0}</dd></div>
      </dl>
    </section>
    <section className="panel timeline-panel">
      <div className="timeline-controls">
        <button onClick={() => seek(time - 1 / info.manifest.fps)} aria-label="Предыдущий кадр">−1f</button>
        <button className="play-button" onClick={togglePlayback}>{playing ? 'Пауза' : 'Воспроизвести'}</button>
        <button onClick={() => seek(time + 1 / info.manifest.fps)} aria-label="Следующий кадр">+1f</button>
        <span>{time.toFixed(2)} / {duration.toFixed(2)} s</span>
        <label>Скорость<select value={speed} onChange={event => setSpeed(Number(event.target.value))}>{[0.25, 0.5, 1, 2, 5].map(value => <option key={value} value={value}>{value}×</option>)}</select></label>
      </div>
      <input className="timeline-range" type="range" min="0" max={duration} step={1 / info.manifest.fps} value={time} onChange={event => seek(Number(event.target.value))} aria-label="Время записи" />
    </section>
  </main>
}
