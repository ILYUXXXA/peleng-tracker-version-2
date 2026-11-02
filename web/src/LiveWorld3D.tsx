import { Canvas } from '@react-three/fiber'
import { Line, OrbitControls, Text } from '@react-three/drei'
import type { WorldTrack } from './services/VisionWebSocket'

function Marker({ track }: { track: WorldTrack }) {
  if (!track.position_xyz_m || track.position_status !== 'valid') return null
  const [x, y, z] = track.position_xyz_m
  const trail = track.trajectory.map(point => [point[1], point[3], -point[2]] as [number, number, number])
  return <group position={[x, z, -y]}>
    <mesh>
      <sphereGeometry args={[track.is_target ? 0.14 : 0.09, 20, 16]} />
      <meshStandardMaterial color={track.is_target ? '#ffca63' : '#69d7ff'} emissive={track.is_target ? '#805000' : '#12445a'} emissiveIntensity={0.5} />
    </mesh>
    <Text position={[0, 0.22, 0]} fontSize={0.13} color="white" anchorX="center">{track.class_name} · {track.global_id}</Text>
  </group>
}

export function LiveWorld3D({ tracks, calibrationValid }: { tracks: WorldTrack[]; calibrationValid: boolean }) {
  const measured = tracks.filter(track => track.position_status === 'valid' && track.position_xyz_m)
  return <div className="scene-canvas live-world-scene">
    <Canvas camera={{ position: [5, 4, 7], fov: 48 }}>
      <color attach="background" args={['#08131b']} />
      <ambientLight intensity={0.8} />
      <directionalLight position={[4, 8, 5]} intensity={1.2} />
      <gridHelper args={[20, 20, '#294555', '#182d3a']} />
      <axesHelper args={[1.5]} />
      {measured.map(track => <Marker key={track.global_id} track={track} />)}
      <OrbitControls makeDefault enableDamping />
    </Canvas>
    {!calibrationValid && <p className="hint">Нет валидной калибровки: реальные XYZ появятся после калибровки обеих камер.</p>}
    {calibrationValid && measured.length === 0 && <p className="hint">Пока нет подтверждённых 3D-позиций. Нужны синхронные наблюдения объекта минимум с двух камер.</p>}
  </div>
}
